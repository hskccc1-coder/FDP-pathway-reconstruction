from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import fiona
import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import MultiPolygon, mapping, shape
from tqdm import tqdm

from common import load_config, resolve, save_json, seed_everything, sha256_file


def iter_windows(width: int, height: int, tile_size: int, overlap: float):
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    columns = list(range(0, max(width - tile_size, 0) + 1, stride))
    rows = list(range(0, max(height - tile_size, 0) + 1, stride))
    final_column = max(width - tile_size, 0)
    final_row = max(height - tile_size, 0)
    if not columns or columns[-1] != final_column:
        columns.append(final_column)
    if not rows or rows[-1] != final_row:
        rows.append(final_row)
    for row in sorted(set(rows)):
        for column in sorted(set(columns)):
            yield Window(
                column,
                row,
                min(tile_size, width - column),
                min(tile_size, height - row),
            )


def rgb_uint8(array: np.ndarray, value_min: float, value_max: float) -> np.ndarray:
    image = np.moveaxis(array[:3], 0, -1).astype(np.float32)
    image = np.nan_to_num(image, nan=value_min, posinf=value_max, neginf=value_min)
    image = np.clip((image - value_min) / (value_max - value_min), 0.0, 1.0)
    return np.round(image * 255.0).astype(np.uint8)


def polygon_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    repaired = geometry.buffer(0)
    if repaired.geom_type == "Polygon":
        return [repaired]
    if repaired.geom_type == "MultiPolygon":
        return list(repaired.geoms)
    return []


def build_generator(config: dict, base: Path):
    settings = config["sam2"]
    repository = resolve(base, config["paths"]["sam2_repository"])
    checkpoint = resolve(base, config["paths"]["sam2_checkpoint"])
    expected_hash = settings.get("checkpoint_sha256", "").strip().upper()
    if expected_hash:
        actual_hash = sha256_file(checkpoint)
        if actual_hash != expected_hash:
            raise ValueError(f"SAM2 checkpoint checksum differs: {actual_hash} != {expected_hash}")
    sys.path.insert(0, str(repository))
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(settings["model_config"], str(checkpoint), device=device)
    generator = SAM2AutomaticMaskGenerator(model, **settings["automatic_mask_generator"])
    return generator, device, repository, checkpoint


def run(config: dict, base: Path, feature: str, scale: int, output: Path, overwrite: bool) -> None:
    settings = config["sam2"]
    allowed_features = list(settings["features"])
    allowed_scales = [int(value) for value in settings["tile_sizes_px"]]
    if feature not in allowed_features:
        raise ValueError(f"Feature must be one of {allowed_features}")
    if scale not in allowed_scales:
        raise ValueError(f"Scale must be one of {allowed_scales}")
    feature_path = resolve(base, config["paths"][f"{feature}_raster"])
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(int(config["seed"]))
    generator, device, repository, checkpoint = build_generator(config, base)
    overlap = float(settings["overlap_fraction"])
    value_min = float(settings["value_min"])
    value_max = float(settings["value_max"])
    minimum_polygon_area_m2 = float(settings.get("minimum_polygon_area_m2", 0.5))
    schema = {
        "geometry": "Polygon",
        "properties": {
            "tile_id": "int",
            "mask_id": "int",
            "feature": "str:16",
            "scale_px": "int",
            "pred_iou": "float",
            "stability": "float",
            "area_m2": "float",
        },
    }
    tile_count = 0
    mask_count = 0
    polygon_count = 0
    with rasterio.open(feature_path) as source:
        windows = list(iter_windows(source.width, source.height, scale, overlap))
        crs_wkt = source.crs.to_wkt() if source.crs is not None else None
        with fiona.open(
            output,
            mode="w",
            driver="GPKG",
            layer="objects",
            schema=schema,
            crs_wkt=crs_wkt,
        ) as sink:
            for tile_id, window in enumerate(tqdm(windows, desc=f"SAM2 {feature} {scale}px")):
                array = source.read([1, 2, 3], window=window, boundless=True, fill_value=0)
                image = rgb_uint8(array, value_min, value_max)
                if not np.any(image):
                    continue
                inference_context = (
                    torch.autocast("cuda", dtype=torch.float16)
                    if device == "cuda" and settings.get("use_fp16", True)
                    else contextlib.nullcontext()
                )
                with torch.inference_mode(), inference_context:
                    masks = generator.generate(image)
                tile_count += 1
                mask_count += len(masks)
                transform = window_transform(window, source.transform)
                for mask_id, record in enumerate(masks):
                    binary = record["segmentation"].astype(np.uint8)
                    for geometry_mapping, value in shapes(binary, mask=binary.astype(bool), transform=transform):
                        if not value:
                            continue
                        for polygon in polygon_parts(shape(geometry_mapping)):
                            if polygon.is_empty or polygon.area < minimum_polygon_area_m2:
                                continue
                            sink.write(
                                {
                                    "geometry": mapping(polygon),
                                    "properties": {
                                        "tile_id": tile_id,
                                        "mask_id": mask_id,
                                        "feature": feature,
                                        "scale_px": scale,
                                        "pred_iou": float(record.get("predicted_iou", np.nan)),
                                        "stability": float(record.get("stability_score", np.nan)),
                                        "area_m2": float(polygon.area),
                                    },
                                }
                            )
                            polygon_count += 1
    save_json(
        output.with_suffix(".run.json"),
        {
            "input": str(feature_path),
            "output": str(output),
            "feature": feature,
            "scale_px": scale,
            "overlap_fraction": overlap,
            "processed_tiles": tile_count,
            "sam2_masks": mask_count,
            "vector_polygons": polygon_count,
            "device": device,
            "sam2_repository": str(repository),
            "sam2_checkpoint": str(checkpoint),
            "automatic_mask_generator": settings["automatic_mask_generator"],
            "minimum_polygon_area_m2": minimum_polygon_area_m2,
            "use_fp16": bool(settings.get("use_fp16", True)),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM2 automatic multi-scale object segmentation")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--feature", required=True, choices=("rgb", "slope"))
    parser.add_argument("--scale", required=True, type=int, choices=(512, 1024, 2048, 4096))
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config, base = load_config(args.config)
    output = (
        Path(args.output).resolve()
        if args.output
        else resolve(base, f"results/segmentation_metrics/runtime/sam2/raw/{args.feature}_{args.scale}.gpkg")
    )
    run(config, base, args.feature, args.scale, output, args.overwrite)


if __name__ == "__main__":
    main()
