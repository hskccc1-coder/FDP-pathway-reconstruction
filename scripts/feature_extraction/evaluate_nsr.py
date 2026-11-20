from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
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
        # Never drop geometry silently; an unrepaired input invalidates the metrics.
        raise ValueError(f"Geometry could not be repaired with buffer(0): {geometry.wkt[:200]}")
    return repaired


def _union(geometries):
    values = [_repair(value) for value in geometries if value is not None and not value.is_empty]
    return _repair(unary_union(values)) if values else GeometryCollection()


def _read(path: Path, target_crs=None) -> gpd.GeoDataFrame:
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f"Vector has no CRS: {path}")
    if target_crs is not None and frame.crs != target_crs:
        frame = frame.to_crs(target_crs)
    frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    frame.geometry = frame.geometry.map(_repair)
    return frame.loc[~frame.geometry.is_empty].reset_index(drop=True)


def _polygon_parts(geometry):
    geometry = _repair(geometry)
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [value for value in geometry.geoms if not value.is_empty]
    return []


def _clip_union(frame: gpd.GeoDataFrame, roi):
    return _repair(_union(frame.geometry).intersection(roi))


def _resolve_pair(first, second, priority: list[str], tolerance: float):
    overlap = float(first.intersection(second).area)
    if overlap <= tolerance:
        return first, second, overlap
    if priority == ["Field", "Water"]:
        second = _repair(second.difference(first))
    elif priority == ["Water", "Field"]:
        first = _repair(first.difference(second))
    else:
        raise ValueError("Priority must contain Field and Water exactly once")
    return first, second, overlap


def _metrics(name: str, truth, prediction) -> dict:
    tp = float(truth.intersection(prediction).area)
    truth_area = float(truth.area)
    prediction_area = float(prediction.area)
    fp = max(0.0, prediction_area - tp)
    fn = max(0.0, truth_area - tp)
    denominator = tp + fp + fn
    precision = tp / (tp + fp) if tp + fp > 0 else math.nan
    recall = tp / (tp + fn) if tp + fn > 0 else math.nan
    # F1 = 2TP/(2TP+FP+FN) is 0 when TP = 0 and is undefined only for an empty class.
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if denominator > 0 else math.nan
    iou = tp / denominator if denominator > 0 else math.nan
    return {
        "class_name": name,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP_area_m2": tp,
        "FP_area_m2": fp,
        "FN_area_m2": fn,
        "truth_area_m2": truth_area,
        "prediction_area_m2": prediction_area,
        "empty_class": denominator == 0,
    }


def _registered_roi(config_value: str, split: str) -> tuple[Path, Path]:
    config, base = load_config(config_value)
    key = {"validation": "validation_region", "test": "independent_test_region"}[split]
    region = config.get("paths", {}).get(key)
    if not region:
        raise KeyError(f"Config does not register paths.{key}")
    path = resolve(base, region)
    if not path.is_file():
        raise FileNotFoundError(path)
    return Path(config_value).expanduser().resolve(), path


def evaluate(args) -> None:
    if args.split == "validation" and args.feature == "fused":
        raise ValueError("Validation candidate evaluation cannot use feature=fused")
    if (args.feature == "fused") != (args.scale == 0):
        raise ValueError("Use feature=fused with scale=0 only for a fused final map")
    prediction_path = Path(args.prediction).resolve()
    field_truth_path = Path(args.field_truth).resolve()
    water_truth_path = Path(args.water_truth).resolve()
    roi_path = Path(args.roi).resolve()
    interstrip_path = Path(args.interstrip_region).resolve()
    for path in [prediction_path, field_truth_path, water_truth_path, roi_path, interstrip_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    # The split label is not trusted on its own: the ROI must be the region registered for it.
    config_path, registered_roi_path = _registered_roi(args.config, args.split)
    if sha256_dataset(roi_path) != sha256_dataset(registered_roi_path):
        raise ValueError(
            f"split={args.split} requires the registered region {registered_roi_path}; got {roi_path}"
        )
    args.class_priority = nsr_class_priority(load_config(args.config)[0], args.class_priority)

    roi_frame = _read(roi_path)
    if getattr(roi_frame.crs, "is_geographic", False):
        raise ValueError("Area evaluation requires a projected CRS")
    roi = _union(roi_frame.geometry)
    interstrip = _clip_union(_read(interstrip_path, roi_frame.crs), roi)

    prediction_frame = _read(prediction_path, roi_frame.crs)
    if args.class_field not in prediction_frame.columns:
        raise KeyError(f"Prediction class field is missing: {args.class_field}")
    raw_prediction_sum = float(prediction_frame.geometry.intersection(roi).area.sum())
    raw_prediction_union = float(_clip_union(prediction_frame, roi).area)
    internal_overlap = max(0.0, raw_prediction_sum - raw_prediction_union)
    if args.require_disjoint_input and internal_overlap > args.overlap_tolerance_m2:
        raise ValueError(
            f"Prediction polygons overlap by {internal_overlap:.6f} m2; run fuse_nsr_outputs.py first"
        )

    labels = prediction_frame[args.class_field].astype(str)
    field_values = {args.field_value, *args.additional_field_value}
    predicted_field = _clip_union(prediction_frame.loc[labels.isin(field_values)], roi)
    predicted_water = _clip_union(prediction_frame.loc[labels == args.water_value], roi)
    predicted_field, predicted_water, prediction_cross_overlap = _resolve_pair(
        predicted_field,
        predicted_water,
        args.class_priority,
        args.overlap_tolerance_m2,
    )

    truth_field = _clip_union(_read(field_truth_path, roi_frame.crs), roi)
    truth_water = _clip_union(_read(water_truth_path, roi_frame.crs), roi)
    # Truth is fixed reference data: an overlap is a labelling error, not something
    # the tunable class priority may rewrite.
    truth_cross_overlap = float(truth_field.intersection(truth_water).area)
    if truth_cross_overlap > args.overlap_tolerance_m2:
        raise ValueError(
            f"Field and Water truth overlap by {truth_cross_overlap:.6f} m2; correct the truth layers first"
        )

    truth = {
        "Field": _repair(truth_field.difference(interstrip)),
        "Interstrip": _repair(truth_field.intersection(interstrip)),
        "Water": _repair(truth_water.difference(interstrip)),
    }
    prediction = {
        "Field": _repair(predicted_field.difference(interstrip)),
        "Interstrip": _repair(predicted_field.intersection(interstrip)),
        "Water": _repair(predicted_water.difference(interstrip)),
    }
    rows = [_metrics(name, truth[name], prediction[name]) for name in ["Field", "Interstrip", "Water"]]
    table = pd.DataFrame(rows)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    overall = {
        "mean_iou": float(table["IoU"].mean()),
        "mean_f1": float(table["F1"].mean()),
    }
    pd.DataFrame([overall]).to_csv(output_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    save_json(
        output_dir / "evaluation.json",
        {
            "candidate_id": args.candidate_id,
            "feature": args.feature,
            "scale_px": args.scale,
            "split": args.split,
            "prediction": str(prediction_path),
            "field_truth": str(field_truth_path),
            "water_truth": str(water_truth_path),
            "roi": str(roi_path),
            "interstrip_region": str(interstrip_path),
            "class_field": args.class_field,
            "class_values": {"Field": sorted(field_values), "Water": [args.water_value]},
            "class_priority": args.class_priority,
            "overlap_tolerance_m2": args.overlap_tolerance_m2,
            "prediction_internal_overlap_m2": internal_overlap,
            "prediction_field_water_overlap_before_resolution_m2": prediction_cross_overlap,
            "truth_field_water_overlap_m2": truth_cross_overlap,
            "config": str(config_path),
            "registered_roi": str(registered_roi_path),
            "input_sha256": {
                "prediction": sha256_dataset(prediction_path),
                "field_truth": sha256_dataset(field_truth_path),
                "water_truth": sha256_dataset(water_truth_path),
                "roi": sha256_dataset(roi_path),
                "interstrip_region": sha256_dataset(interstrip_path),
            },
            "per_class": rows,
            "overall": overall,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlap-safe vector evaluation for N-SR outputs")
    parser.add_argument("--config", required=True, help="config.json registering the validation and test regions")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--field-truth", required=True)
    parser.add_argument("--water-truth", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--interstrip-region", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", required=True, choices=["validation", "test"])
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--feature", required=True, choices=["rgb", "slope", "fused"])
    parser.add_argument("--scale", required=True, type=int, choices=[0, 512, 1024, 2048, 4096])
    parser.add_argument("--class-field", default="class_pred")
    parser.add_argument("--field-value", default="Field")
    parser.add_argument("--additional-field-value", action="append", default=["Greenhouse"])
    parser.add_argument("--water-value", default="Water")
    parser.add_argument(
        "--class-priority",
        nargs=2,
        default=None,
        choices=["Field", "Water"],
        help="Optional; must equal config nsr_evaluation.class_priority (highest first)",
    )
    parser.add_argument("--overlap-tolerance-m2", type=float, default=0.001)
    parser.add_argument(
        "--require-disjoint-input",
        action="store_true",
        help="Reject any overlapping prediction polygons; use for the final fused test map",
    )
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
