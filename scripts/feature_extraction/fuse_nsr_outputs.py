from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

from common import load_config, nsr_class_priority, resolve, save_json, sha256_dataset


def _repair(geometry):
    if geometry is None or geometry.is_empty:
        return GeometryCollection()
    if geometry.is_valid:
        return geometry
    repaired = geometry.buffer(0)
    if not repaired.is_valid:
        # Never drop geometry silently; an unrepaired input invalidates the fused map.
        raise ValueError(f"Geometry could not be repaired with buffer(0): {geometry.wkt[:200]}")
    return repaired


def _union(geometries):
    values = [_repair(value) for value in geometries if value is not None and not value.is_empty]
    return _repair(unary_union(values)) if values else GeometryCollection()


def _parts(geometry):
    geometry = _repair(geometry)
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [value for value in geometry.geoms if not value.is_empty]
    return []


def _read(path: Path, target_crs=None) -> gpd.GeoDataFrame:
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f"Vector has no CRS: {path}")
    if target_crs is not None and frame.crs != target_crs:
        frame = frame.to_crs(target_crs)
    return frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty].copy()


def _class_geometry(path: Path, target_crs, class_field: str, values: set[str], roi):
    frame = _read(path, target_crs)
    if class_field not in frame.columns:
        raise KeyError(f"Class field {class_field!r} is missing from {path}")
    selected = frame.loc[frame[class_field].astype(str).isin(values)]
    return _repair(_union(selected.geometry).intersection(roi))


def _load_selection(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("selection_split") != "validation":
        raise ValueError("Fusion requires a validation-only selection manifest")
    for name in ["Field", "Interstrip", "Water"]:
        if name not in selection.get("selected", {}):
            raise KeyError(f"Selection manifest has no {name} entry")
    return selection


def fuse(args) -> None:
    selection_path = Path(args.selection).resolve()
    roi_path = Path(args.roi).resolve()
    interstrip_path = Path(args.interstrip_region).resolve()
    selection = _load_selection(selection_path)
    protocol = selection.get("evaluation_protocol")
    protocol_hashes = selection.get("evaluation_input_sha256")
    if not protocol or not protocol_hashes:
        raise KeyError("Selection manifest has no validated evaluation protocol")
    args.class_priority = nsr_class_priority(load_config(args.config)[0], args.class_priority)
    if args.class_priority != protocol["class_priority"]:
        raise ValueError("Class priority cannot change after validation selection")
    expected_field_values = sorted({args.field_value, *args.additional_field_value})
    if expected_field_values != sorted(protocol["class_values"]["Field"]):
        raise ValueError("Field class mapping cannot change after validation selection")
    if [args.water_value] != protocol["class_values"]["Water"]:
        raise ValueError("Water class mapping cannot change after validation selection")
    if args.class_field != protocol["class_field"]:
        raise ValueError("Prediction class field cannot change after validation selection")
    selected_paths = {
        name: Path(selection["selected"][name]["prediction"]).resolve()
        for name in ["Field", "Interstrip", "Water"]
    }
    for path in [selection_path, roi_path, interstrip_path, *selected_paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for name, path in selected_paths.items():
        actual_hash = sha256_dataset(path)
        expected_hash = selection["selected"][name].get("prediction_sha256")
        if not expected_hash:
            raise KeyError(f"Selection manifest has no prediction hash for {name}")
        if actual_hash != expected_hash:
            raise ValueError(f"Selected prediction changed after validation: {path}")
    if sha256_dataset(interstrip_path) != protocol_hashes["interstrip_region"]:
        raise ValueError("Interstrip region changed after validation selection")
    validation_roi_path = Path(protocol["roi"]).resolve()
    if not validation_roi_path.is_file():
        raise FileNotFoundError(validation_roi_path)
    if sha256_dataset(validation_roi_path) != protocol_hashes["roi"]:
        raise ValueError("Validation ROI changed after candidate selection")
    if sha256_dataset(roi_path) == protocol_hashes["roi"]:
        raise ValueError("Final fusion ROI must not be the validation ROI used for selection")
    config, base = load_config(args.config)
    registered = {
        key: resolve(base, config["paths"][key])
        for key in ["validation_region", "independent_test_region"]
    }
    if sha256_dataset(validation_roi_path) != sha256_dataset(registered["validation_region"]):
        raise ValueError("Selection was not made on the registered validation region")
    if sha256_dataset(roi_path) != sha256_dataset(registered["independent_test_region"]):
        raise ValueError("Final fusion ROI must be the registered independent test region")

    roi_frame = _read(roi_path)
    if getattr(roi_frame.crs, "is_geographic", False):
        raise ValueError("Fusion requires a projected CRS")
    roi = _union(roi_frame.geometry)
    validation_roi = _union(_read(validation_roi_path, roi_frame.crs).geometry)
    roi_overlap = float(roi.intersection(validation_roi).area)
    tolerance = float(protocol["overlap_tolerance_m2"])
    if roi_overlap > tolerance:
        raise ValueError(
            f"Final ROI overlaps validation ROI by {roi_overlap:.6f} m2, above tolerance {tolerance}"
        )
    interstrip = _repair(_union(_read(interstrip_path, roi_frame.crs).geometry).intersection(roi))
    field_values = {args.field_value, *args.additional_field_value}
    field_outside = _class_geometry(
        selected_paths["Field"], roi_frame.crs, args.class_field, field_values, roi
    ).difference(interstrip)
    field_inside = _class_geometry(
        selected_paths["Interstrip"], roi_frame.crs, args.class_field, field_values, roi
    ).intersection(interstrip)
    raw_field = _repair(unary_union([field_outside, field_inside]))
    raw_water = _class_geometry(
        selected_paths["Water"], roi_frame.crs, args.class_field, {args.water_value}, roi
    )
    raw_overlap = float(raw_field.intersection(raw_water).area)
    if args.class_priority == ["Field", "Water"]:
        field = raw_field
        water = _repair(raw_water.difference(field))
    elif args.class_priority == ["Water", "Field"]:
        water = raw_water
        field = _repair(raw_field.difference(water))
    else:
        raise ValueError("Priority must contain Field and Water exactly once")

    rows = []
    for evaluation_class, semantic_class, geometry, source_class in [
        ("Field", "Field", field.difference(interstrip), "Field"),
        ("Interstrip", "Field", field.intersection(interstrip), "Interstrip"),
        ("Water", "Water", water, "Water"),
    ]:
        selected = selection["selected"][source_class]
        for geometry_part in _parts(geometry):
            rows.append(
                {
                    args.output_class_field: semantic_class,
                    "evaluation_class": evaluation_class,
                    "candidate_id": selected["candidate_id"],
                    "feature": selected["feature"],
                    "scale_px": int(selected["scale_px"]),
                    "geometry": geometry_part,
                }
            )
    output = Path(args.output).resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = gpd.GeoDataFrame(rows, geometry="geometry", crs=roi_frame.crs)
    result["area_m2"] = result.geometry.area.astype(float)
    result.to_file(output, driver="GPKG", layer="objects")
    save_json(
        output.with_suffix(".fusion.json"),
        {
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": sha256_dataset(selection_path),
            "roi": str(roi_path),
            "interstrip_region": str(interstrip_path),
            "validation_roi": str(validation_roi_path),
            "validation_final_roi_overlap_m2": roi_overlap,
            "class_field": args.class_field,
            "class_priority": args.class_priority,
            "selected_inputs": {
                name: {
                    **selection["selected"][name],
                    "sha256": sha256_dataset(path),
                }
                for name, path in selected_paths.items()
            },
            "raw_field_water_overlap_m2": raw_overlap,
            "resolved_field_area_m2": float(field.area),
            "resolved_water_area_m2": float(water.area),
            "output_polygon_count": len(result),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse validation-selected N-SR class outputs")
    parser.add_argument("--config", required=True, help="config.json registering the validation and test regions")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--interstrip-region", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-field", default="class_pred")
    parser.add_argument("--field-value", default="Field")
    parser.add_argument("--additional-field-value", action="append", default=["Greenhouse"])
    parser.add_argument("--water-value", default="Water")
    parser.add_argument("--output-class-field", default="class_pred")
    parser.add_argument(
        "--class-priority",
        nargs=2,
        default=None,
        choices=["Field", "Water"],
        help="Optional; must equal config nsr_evaluation.class_priority (highest first)",
    )
    parser.add_argument("--overwrite", action="store_true")
    fuse(parser.parse_args())


if __name__ == "__main__":
    main()
