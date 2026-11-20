from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject


def _sha256_dataset(path: Path) -> str:
    """Hash a file or all sidecars belonging to one ESRI Shapefile."""
    digest = hashlib.sha256()
    members = [path]
    if path.suffix.lower() == ".shp":
        members = sorted(
            candidate
            for candidate in path.parent.glob(f"{path.stem}.*")
            if candidate.is_file()
        )
    for member in members:
        digest.update(member.name.encode("utf-8"))
        with member.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _rasterize_mask(
    vector_path: Path,
    reference: rasterio.io.DatasetReader,
    field: str | None = None,
    values: list[str] | None = None,
    all_touched: bool = True,
) -> np.ndarray:
    if not vector_path.is_file():
        raise FileNotFoundError(vector_path)
    frame = gpd.read_file(vector_path)
    if frame.crs is None:
        raise ValueError(f"Vector dataset has no CRS: {vector_path}")
    if reference.crs is None:
        raise ValueError(f"Reference raster has no CRS: {reference.name}")
    if frame.crs != reference.crs:
        frame = frame.to_crs(reference.crs)
    if field is not None:
        if field not in frame.columns:
            raise KeyError(
                f"Field {field!r} is missing from {vector_path}; found {sorted(frame.columns)}"
            )
        requested = values or []
        if not requested:
            raise ValueError(f"At least one value is required for {field}")
        frame = frame[frame[field].astype(str).isin([str(value) for value in requested])]
    geometries = [geometry for geometry in frame.geometry if geometry is not None and not geometry.is_empty]
    if not geometries:
        return np.zeros((reference.height, reference.width), dtype=bool)
    return rasterize(
        ((geometry, 1) for geometry in geometries),
        out_shape=(reference.height, reference.width),
        transform=reference.transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    ).astype(bool)


def _aligned_array(path: Path, reference: rasterio.io.DatasetReader) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with rasterio.open(path) as source:
        if source.crs is None or reference.crs is None:
            raise ValueError("Both source and reference rasters must define a CRS")
        fill_value = source.nodata if source.nodata is not None else 0
        result = np.full(
            (reference.height, reference.width), fill_value, dtype=source.dtypes[0]
        )
        reproject(
            source=rasterio.band(source, 1),
            destination=result,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            src_nodata=source.nodata,
            dst_nodata=fill_value,
            resampling=Resampling.nearest,
        )
    return result


def _write_raster(
    path: Path, array: np.ndarray, reference: rasterio.io.DatasetReader, nodata: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = reference.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=nodata,
        compress="lzw",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(path, "w", **profile) as output:
        output.write(array.astype(np.uint8), 1)


def prepare(args: argparse.Namespace) -> None:
    paths = {
        "reference_raster": Path(args.reference_raster).resolve(),
        "truth_raster": Path(args.truth_raster).resolve(),
        "field_vector": Path(args.field_vector).resolve(),
        "interstrip_vector": Path(args.interstrip_vector).resolve(),
        "water_vector": Path(args.water_vector).resolve(),
        "interstrip_region": Path(args.interstrip_region).resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    with rasterio.open(paths["reference_raster"]) as reference:
        transform = reference.transform
        if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
            raise ValueError("The reference must be a north-up projected raster")

        truth = _aligned_array(paths["truth_raster"], reference).astype(np.uint8)
        valid = truth != args.nodata
        observed_truth = set(np.unique(truth[valid]).tolist())
        if not observed_truth.issubset({0, args.field_class_id, args.water_class_id}):
            raise ValueError(f"Unexpected truth classes: {sorted(observed_truth)}")

        zone = _rasterize_mask(
            paths["interstrip_region"], reference, all_touched=args.all_touched
        )
        field_candidate = _rasterize_mask(
            paths["field_vector"],
            reference,
            args.field_field,
            args.field_value,
            args.all_touched,
        )
        interstrip_candidate = _rasterize_mask(
            paths["interstrip_vector"],
            reference,
            args.interstrip_field,
            args.interstrip_value,
            args.all_touched,
        )
        water_candidate = _rasterize_mask(
            paths["water_vector"],
            reference,
            args.water_field,
            args.water_value,
            args.all_touched,
        )

        field = ((field_candidate & ~zone) | (interstrip_candidate & zone)) & valid
        water = water_candidate & valid
        raw_overlap_pixels = int(np.count_nonzero(field & water))
        if args.priority == "field":
            water &= ~field
        else:
            field &= ~water

        prediction = np.zeros(truth.shape, dtype=np.uint8)
        prediction[field] = args.field_class_id
        prediction[water] = args.water_class_id
        prediction[~valid] = args.nodata

        output = Path(args.output).resolve()
        aligned_truth_output = Path(args.aligned_truth_output).resolve()
        _write_raster(output, prediction, reference, args.nodata)
        _write_raster(aligned_truth_output, truth, reference, args.nodata)
        width = reference.width
        height = reference.height

    transform_values = transform.to_gdal()
    pixel_area = abs(transform.a * transform.e)
    metadata = {
        "purpose": "Locked N-SR test prediction raster for paired spatial-block bootstrap",
        "reference_grid": {
            "path": str(paths["reference_raster"]),
            "width": width,
            "height": height,
            "geotransform": list(transform_values),
            "pixel_area_m2": pixel_area,
        },
        "truth_raster": str(paths["truth_raster"]),
        "valid_test_pixels": int(np.count_nonzero(valid)),
        "class_ids": {"background": 0, "field": args.field_class_id, "water": args.water_class_id},
        "class_priority": ["Field", "Water"] if args.priority == "field" else ["Water", "Field"],
        "all_touched": args.all_touched,
        "field_outside_interstrip": {
            "path": str(paths["field_vector"]),
            "field": args.field_field,
            "values": args.field_value,
        },
        "field_inside_interstrip": {
            "path": str(paths["interstrip_vector"]),
            "field": args.interstrip_field,
            "values": args.interstrip_value,
        },
        "water": {
            "path": str(paths["water_vector"]),
            "field": args.water_field,
            "values": args.water_value,
        },
        "interstrip_region": str(paths["interstrip_region"]),
        "raw_field_water_overlap_pixels": raw_overlap_pixels,
        "raw_field_water_overlap_m2": raw_overlap_pixels * pixel_area,
        "output_field_pixels": int(np.count_nonzero(prediction == args.field_class_id)),
        "output_water_pixels": int(np.count_nonzero(prediction == args.water_class_id)),
        "input_sha256": {name: _sha256_dataset(path) for name, path in paths.items()},
        "outputs": {
            "prediction": str(output),
            "aligned_truth": str(aligned_truth_output),
        },
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rasterize and fuse validation-selected N-SR outputs on a locked test grid"
    )
    parser.add_argument("--reference-raster", required=True)
    parser.add_argument("--truth-raster", required=True)
    parser.add_argument("--field-vector", required=True)
    parser.add_argument("--field-field", required=True)
    parser.add_argument("--field-value", action="append", required=True)
    parser.add_argument("--interstrip-vector", required=True)
    parser.add_argument("--interstrip-field", required=True)
    parser.add_argument("--interstrip-value", action="append", required=True)
    parser.add_argument("--water-vector", required=True)
    parser.add_argument("--water-field", required=True)
    parser.add_argument("--water-value", action="append", required=True)
    parser.add_argument("--interstrip-region", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--aligned-truth-output", required=True)
    parser.add_argument("--field-class-id", type=int, default=1)
    parser.add_argument("--water-class-id", type=int, default=2)
    parser.add_argument("--nodata", type=int, default=255)
    parser.add_argument("--priority", choices=["field", "water"], default="field")
    parser.add_argument("--all-touched", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
