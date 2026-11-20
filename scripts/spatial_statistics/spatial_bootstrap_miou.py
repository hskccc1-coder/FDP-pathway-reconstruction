from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_raster(path: Path) -> tuple[np.ndarray, dict]:
    import rasterio

    if not path.is_file():
        raise FileNotFoundError(path)
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise ValueError(f"Raster has no CRS: {path}")
        array = dataset.read(1)
        metadata = {
            "width": dataset.width,
            "height": dataset.height,
            "transform": dataset.transform,
            "geotransform": dataset.transform.to_gdal(),
            "crs": dataset.crs,
            "projection": dataset.crs.to_wkt() if dataset.crs else "",
            "nodata": dataset.nodata,
        }
    return array, metadata


def _assert_same_grid(reference: dict, candidate: dict, name: str) -> None:
    for key in ["width", "height"]:
        if reference[key] != candidate[key]:
            raise ValueError(f"{name} differs from truth grid in {key}")
    if reference["crs"] != candidate["crs"]:
        raise ValueError(f"{name} differs from truth coordinate reference system")
    if not np.allclose(reference["geotransform"], candidate["geotransform"], atol=1e-9):
        raise ValueError(f"{name} differs from truth geotransform")


def _read_prediction_on_grid(path: Path, reference: dict, nodata: int) -> tuple[np.ndarray, bool]:
    """Read a prediction, nearest-neighbour aligning it when inference crops differ."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    array, grid = _read_raster(path)
    try:
        _assert_same_grid(reference, grid, path.name)
        return array, False
    except ValueError:
        pass

    with rasterio.open(path) as source:
        result = np.full(
            (reference["height"], reference["width"]), nodata, dtype=source.dtypes[0]
        )
        reproject(
            source=rasterio.band(source, 1),
            destination=result,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=reference["transform"],
            dst_crs=reference["crs"],
            dst_nodata=nodata,
            resampling=Resampling.nearest,
        )
    return result, True


def confusion_matrix(
    truth: np.ndarray, prediction: np.ndarray, num_classes: int, nodata: int
) -> np.ndarray:
    valid = truth != nodata
    target = truth[valid].astype(np.int64)
    predicted = prediction[valid].astype(np.int64)
    if target.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    if target.min() < 0 or target.max() >= num_classes:
        raise ValueError("Truth contains a class outside the configured range")
    if predicted.min() < 0 or predicted.max() >= num_classes:
        raise ValueError("Prediction contains a class outside the configured range")
    encoded = target * num_classes + predicted
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_confusion(cm: np.ndarray, foreground_ids: list[int]) -> dict[str, object]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    denominator = tp + fp + fn
    iou = np.divide(tp, denominator, out=np.full_like(tp, np.nan), where=denominator > 0)
    selected = iou[foreground_ids]
    mean_iou = float(np.nanmean(selected)) if np.isfinite(selected).any() else math.nan
    return {
        "mean_iou": mean_iou,
        "per_class_iou": iou.tolist(),
        "confusion_matrix": cm.astype(np.int64).tolist(),
    }


def _block_confusions(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    nodata: int,
    num_classes: int,
    block_pixels: int,
    minimum_valid_fraction: float,
) -> tuple[list[tuple[int, int, int, int]], dict[str, np.ndarray], np.ndarray]:
    height, width = truth.shape
    blocks: list[tuple[int, int, int, int]] = []
    valid_counts: list[int] = []
    method_cms: dict[str, list[np.ndarray]] = {name: [] for name in predictions}
    for row_start in range(0, height, block_pixels):
        row_stop = min(height, row_start + block_pixels)
        for col_start in range(0, width, block_pixels):
            col_stop = min(width, col_start + block_pixels)
            target = truth[row_start:row_stop, col_start:col_stop]
            valid_count = int(np.count_nonzero(target != nodata))
            nominal_pixels = block_pixels * block_pixels
            if valid_count < max(1, int(math.ceil(minimum_valid_fraction * nominal_pixels))):
                continue
            blocks.append((row_start, row_stop, col_start, col_stop))
            valid_counts.append(valid_count)
            for name, prediction in predictions.items():
                method_cms[name].append(
                    confusion_matrix(
                        target,
                        prediction[row_start:row_stop, col_start:col_stop],
                        num_classes,
                        nodata,
                    )
                )
    if len(blocks) < 2:
        raise ValueError(
            f"Only {len(blocks)} eligible blocks for block size {block_pixels}; reduce block size "
            "or minimum valid fraction"
        )
    return (
        blocks,
        {name: np.stack(values) for name, values in method_cms.items()},
        np.asarray(valid_counts, dtype=np.int64),
    )


def _percentile_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan, math.nan
    lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def _bootstrap(
    block_cms: dict[str, np.ndarray],
    foreground_ids: list[int],
    iterations: int,
    rng: np.random.Generator,
    proposed_method: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    names = list(block_cms)
    n_blocks = next(iter(block_cms.values())).shape[0]
    method_values = {name: np.empty(iterations, dtype=np.float64) for name in names}
    comparison_values = {
        name: {
            "absolute": np.empty(iterations, dtype=np.float64),
            "relative": np.empty(iterations, dtype=np.float64),
        }
        for name in names
        if name != proposed_method
    }
    for iteration in range(iterations):
        sampled = rng.integers(0, n_blocks, size=n_blocks)
        for name in names:
            cm = block_cms[name][sampled].sum(axis=0)
            method_values[name][iteration] = metrics_from_confusion(cm, foreground_ids)["mean_iou"]
        proposed = method_values[proposed_method][iteration]
        for baseline, values in comparison_values.items():
            baseline_value = method_values[baseline][iteration]
            difference = proposed - baseline_value
            values["absolute"][iteration] = difference
            values["relative"][iteration] = (
                100.0 * difference / baseline_value if baseline_value > 0 else math.nan
            )
    return method_values, comparison_values


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyse(args: argparse.Namespace) -> None:
    truth_path = Path(args.truth).resolve()
    prediction_paths: dict[str, Path] = {}
    for item in args.prediction:
        if "=" not in item:
            raise ValueError(f"Prediction must have NAME=PATH form: {item}")
        name, raw_path = item.split("=", 1)
        if not name or name in prediction_paths:
            raise ValueError(f"Prediction name is empty or duplicated: {name!r}")
        prediction_paths[name] = Path(raw_path).resolve()
    if args.proposed_method not in prediction_paths:
        raise KeyError(f"Proposed method {args.proposed_method!r} is not among predictions")
    for path in [truth_path, *prediction_paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    truth, truth_grid = _read_raster(truth_path)
    nodata = args.nodata if args.nodata is not None else truth_grid["nodata"]
    if nodata is None:
        raise ValueError("Truth has no nodata value; pass --nodata explicitly")
    nodata = int(nodata)
    predictions: dict[str, np.ndarray] = {}
    prediction_alignment: dict[str, bool] = {}
    for name, path in prediction_paths.items():
        prediction, was_aligned = _read_prediction_on_grid(path, truth_grid, nodata)
        missing_valid = int(np.count_nonzero((truth != nodata) & (prediction == nodata)))
        if missing_valid:
            raise ValueError(
                f"{name} has {missing_valid} nodata pixels inside the valid truth region after alignment"
            )
        predictions[name] = prediction
        prediction_alignment[name] = was_aligned

    observed_classes = set(np.unique(truth[truth != nodata]).astype(int).tolist())
    expected_classes = set(range(args.num_classes))
    if not observed_classes.issubset(expected_classes):
        raise ValueError(f"Truth contains unexpected classes {sorted(observed_classes - expected_classes)}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = truth_grid["geotransform"]
    pixel_size_x = abs(float(transform[1]))
    pixel_size_y = abs(float(transform[5]))
    if not math.isclose(pixel_size_x, pixel_size_y, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Square pixels are required for metre-based blocks")

    method_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    metadata_blocks: dict[str, object] = {}
    primary_method_values: dict[str, np.ndarray] | None = None
    primary_comparison_values: dict[str, dict[str, np.ndarray]] | None = None

    for block_size_m in args.block_size_m:
        block_pixels = max(1, int(round(block_size_m / pixel_size_x)))
        blocks, block_cms, valid_counts = _block_confusions(
            truth,
            predictions,
            nodata,
            args.num_classes,
            block_pixels,
            args.minimum_valid_fraction,
        )
        rng = np.random.default_rng(args.seed + int(round(block_size_m * 1000)))
        method_values, comparison_values = _bootstrap(
            block_cms,
            args.foreground_class_id,
            args.iterations,
            rng,
            args.proposed_method,
        )
        if math.isclose(block_size_m, args.primary_block_size_m):
            primary_method_values = method_values
            primary_comparison_values = comparison_values

        observed: dict[str, float] = {}
        for name, cms in block_cms.items():
            metrics = metrics_from_confusion(cms.sum(axis=0), args.foreground_class_id)
            observed[name] = float(metrics["mean_iou"])
            lower, upper = _percentile_interval(method_values[name], args.confidence)
            method_rows.append(
                {
                    "block_size_m": block_size_m,
                    "method": name,
                    "mIoU": observed[name],
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": args.confidence,
                    "bootstrap_iterations": args.iterations,
                    "eligible_blocks": len(blocks),
                }
            )
        proposed = observed[args.proposed_method]
        for baseline, values in comparison_values.items():
            baseline_value = observed[baseline]
            difference = proposed - baseline_value
            absolute_lower, absolute_upper = _percentile_interval(values["absolute"], args.confidence)
            relative_lower, relative_upper = _percentile_interval(values["relative"], args.confidence)
            comparison_rows.append(
                {
                    "block_size_m": block_size_m,
                    "proposed_method": args.proposed_method,
                    "baseline": baseline,
                    "proposed_mIoU": proposed,
                    "baseline_mIoU": baseline_value,
                    "absolute_difference": difference,
                    "absolute_difference_percentage_points": 100.0 * difference,
                    "absolute_ci_lower": absolute_lower,
                    "absolute_ci_upper": absolute_upper,
                    "absolute_ci_lower_percentage_points": 100.0 * absolute_lower,
                    "absolute_ci_upper_percentage_points": 100.0 * absolute_upper,
                    "relative_improvement_percent": 100.0 * difference / baseline_value,
                    "relative_ci_lower_percent": relative_lower,
                    "relative_ci_upper_percent": relative_upper,
                    "confidence_level": args.confidence,
                    "bootstrap_iterations": args.iterations,
                    "eligible_blocks": len(blocks),
                }
            )
        for index, ((row_start, row_stop, col_start, col_stop), valid_count) in enumerate(
            zip(blocks, valid_counts)
        ):
            block_rows.append(
                {
                    "block_size_m": block_size_m,
                    "block_index": index,
                    "row_start": row_start,
                    "row_stop": row_stop,
                    "col_start": col_start,
                    "col_stop": col_stop,
                    "valid_pixels": int(valid_count),
                }
            )
        metadata_blocks[str(block_size_m)] = {
            "block_pixels": block_pixels,
            "eligible_blocks": len(blocks),
            "valid_pixels_min": int(valid_counts.min()),
            "valid_pixels_median": float(np.median(valid_counts)),
            "valid_pixels_max": int(valid_counts.max()),
        }

    if primary_method_values is None or primary_comparison_values is None:
        raise ValueError("--primary-block-size-m must also appear in --block-size-m")
    np.savez_compressed(
        output_dir / "primary_bootstrap_replicates.npz",
        **{f"method__{name}": values for name, values in primary_method_values.items()},
        **{
            f"comparison__{baseline}__{kind}": values
            for baseline, payload in primary_comparison_values.items()
            for kind, values in payload.items()
        },
    )
    _write_csv(output_dir / "method_miou_confidence_intervals.csv", method_rows)
    _write_csv(output_dir / "paired_comparison_confidence_intervals.csv", comparison_rows)
    _write_csv(output_dir / "spatial_blocks.csv", block_rows)

    metadata = {
        "analysis": "paired spatial-block percentile bootstrap",
        "interpretation": (
            "Intervals quantify spatial sampling uncertainty within the locked test region, "
            "conditional on the fitted models; they do not include training-seed uncertainty."
        ),
        "truth": str(truth_path),
        "predictions": {name: str(path) for name, path in prediction_paths.items()},
        "prediction_nearest_neighbour_aligned_to_truth_grid": prediction_alignment,
        "proposed_method": args.proposed_method,
        "foreground_class_ids": args.foreground_class_id,
        "num_classes": args.num_classes,
        "nodata": nodata,
        "confidence_level": args.confidence,
        "bootstrap_iterations": args.iterations,
        "seed": args.seed,
        "block_sizes_m": args.block_size_m,
        "primary_block_size_m": args.primary_block_size_m,
        "minimum_valid_fraction": args.minimum_valid_fraction,
        "grid": {
            "width": truth_grid["width"],
            "height": truth_grid["height"],
            "geotransform": list(truth_grid["geotransform"]),
            "valid_pixels": int(np.count_nonzero(truth != nodata)),
        },
        "blocks": metadata_blocks,
        "input_sha256": {
            "truth": _sha256(truth_path),
            **{name: _sha256(path) for name, path in prediction_paths.items()},
        },
    }
    (output_dir / "bootstrap_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"methods": method_rows, "comparisons": comparison_rows}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired spatial-block bootstrap confidence intervals for foreground mIoU"
    )
    parser.add_argument("--truth", required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Repeat NAME=PATH for every aligned prediction raster",
    )
    parser.add_argument("--proposed-method", default="N-SR")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--foreground-class-id", action="append", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--nodata", type=int)
    parser.add_argument("--block-size-m", action="append", type=float, default=None)
    parser.add_argument("--primary-block-size-m", type=float, default=100.0)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.foreground_class_id is None:
        args.foreground_class_id = [1, 2]
    if args.block_size_m is None:
        args.block_size_m = [50.0, 100.0, 200.0]
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between 0 and 1")
    if args.iterations < 1000:
        parser.error("--iterations must be at least 1000")
    if not 0 <= args.minimum_valid_fraction <= 1:
        parser.error("--minimum-valid-fraction must be between 0 and 1")
    analyse(args)


if __name__ == "__main__":
    main()
