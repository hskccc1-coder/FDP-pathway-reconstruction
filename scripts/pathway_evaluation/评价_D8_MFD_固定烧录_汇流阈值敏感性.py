# -*- coding: utf-8 -*-
"""Evaluate FlowAcc-threshold sensitivity at fixed burning depths.

Fixed parameters: alpha=0.8 m, beta=1.2 m, gamma=0 m.
Thresholds: local scan around 1200 plus logarithmic high-threshold extension.
Primary path-matching tolerance: 1 m.
"""

from __future__ import print_function

import gc
import math
import os

from osgeo import gdal, ogr, osr
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "pathway_metrics",
    "dem_routing_runtime",
    u"汇流累积阈值敏感性_A0p8_B1p2_C0",
)
D8_LINE_DIR = os.path.join(OUT_DIR, "D8_lines")
MFD_FLOWACC = os.environ.get(
    "FDP_MFD_SENSITIVITY_INPUT",
    os.path.join(
        PROJECT_ROOT,
        "results",
        "pathway_metrics",
        "dem_routing_runtime",
        "mfd",
        "FlowAcc_MFD",
        "FlowAcc_MFD_A0p80_B1p20.tif",
    ),
)

ALPHA_M = 0.8
BETA_M = 1.2
GAMMA_M = 0.0
THRESHOLDS = (
    list(range(400, 2401, 200))
    + [3000, 4000, 5000, 6000, 8000, 10000]
    + [15000, 20000, 30000, 50000, 80000, 120000]
)
TOLERANCE_M = 1.0
EVAL_CELL_SIZE_M = 0.5
TRUTH_SAMPLE_INTERVAL_M = 0.5
MIN_INTERNAL_GAP_M = 1.0
MFD_BLOCK_ROWS = 2000
TRUTH_SHP = os.path.join(
    PROJECT_ROOT,
    "data",
    "reference_drainage",
    "pathway_evaluation_truth",
    "pathway_evaluation_truth.shp",
)

RESULT_CSV = os.path.join(OUT_DIR, "D8_MFD_threshold_sensitivity_metrics.csv")
PAIR_CSV = os.path.join(OUT_DIR, "MFD_minus_D8_threshold_sensitivity.csv")
MEAN_CSV = os.path.join(OUT_DIR, "Threshold_algorithm_mean_metrics.csv")
BEST_CSV = os.path.join(OUT_DIR, "Threshold_best_by_metric.csv")


def d8_line_path(threshold):
    return os.path.join(
        D8_LINE_DIR,
        "StreamLine_D8_A0p80_B1p20_C0p00_thr%d.shp" % int(threshold),
    )


def assert_same_crs(vector_path, raster_wkt):
    vector_ds = ogr.Open(vector_path)
    if vector_ds is None:
        raise RuntimeError("Cannot open vector: %s" % vector_path)
    vector_srs = vector_ds.GetLayer(0).GetSpatialRef()
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_wkt)
    same = bool(vector_srs and vector_srs.IsSame(raster_srs))
    vector_ds = None
    if not same:
        raise RuntimeError("CRS mismatch: %s" % vector_path)


def build_grid(sample_raster):
    ds = gdal.Open(sample_raster, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError("Cannot open raster: %s" % sample_raster)
    gt = ds.GetGeoTransform()
    source_cell = abs(gt[1])
    factor_float = EVAL_CELL_SIZE_M / source_cell
    factor = int(round(factor_float))
    if factor < 1 or not np.isclose(factor_float, factor, atol=1e-6):
        raise RuntimeError("Evaluation cell size is not an integer multiple of source cell size.")
    grid = {
        "origin_x": gt[0],
        "origin_y": gt[3],
        "cell": EVAL_CELL_SIZE_M,
        "width": int(math.ceil(ds.RasterXSize / factor)),
        "height": int(math.ceil(ds.RasterYSize / factor)),
        "geotransform": (gt[0], EVAL_CELL_SIZE_M, 0.0, gt[3], 0.0, -EVAL_CELL_SIZE_M),
        "projection": ds.GetProjection(),
        "source_width": ds.RasterXSize,
        "source_height": ds.RasterYSize,
        "factor": factor,
    }
    ds = None
    return grid


def rasterize_lines(vector_path, grid):
    vector_ds = ogr.Open(vector_path)
    if vector_ds is None:
        raise RuntimeError("Cannot open vector: %s" % vector_path)
    layer = vector_ds.GetLayer(0)
    mem = gdal.GetDriverByName("MEM").Create("", grid["width"], grid["height"], 1, gdal.GDT_Byte)
    mem.SetGeoTransform(grid["geotransform"])
    mem.SetProjection(grid["projection"])
    band = mem.GetRasterBand(1)
    band.Fill(0)
    error = gdal.RasterizeLayer(mem, [1], layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
    if error != 0:
        raise RuntimeError("Rasterization failed: %s" % vector_path)
    result = band.ReadAsArray().astype(bool)
    vector_ds = None
    mem = None
    return result


def aggregate_mfd_threshold(mfd_path, grid, threshold):
    ds = gdal.Open(mfd_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError("Cannot open MFD raster: %s" % mfd_path)
    factor = grid["factor"]
    width = ds.RasterXSize
    height = ds.RasterYSize
    out = np.zeros((grid["height"], grid["width"]), dtype=bool)
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    for y0 in range(0, height, MFD_BLOCK_ROWS):
        read_rows = min(MFD_BLOCK_ROWS, height - y0)
        data = band.ReadAsArray(0, y0, width, read_rows)
        valid = np.isfinite(data)
        if nodata is not None:
            valid &= data != nodata
        fine = valid & (data > float(threshold))
        del data
        pad_rows = (-fine.shape[0]) % factor
        pad_cols = (-fine.shape[1]) % factor
        if pad_rows or pad_cols:
            fine = np.pad(fine, ((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=False)
        coarse = fine.reshape(
            fine.shape[0] // factor,
            factor,
            fine.shape[1] // factor,
            factor,
        ).any(axis=(1, 3))
        out_row = y0 // factor
        out[out_row:out_row + coarse.shape[0], :coarse.shape[1]] = coarse
        del fine, coarse
    ds = None
    return out


def xy_to_row_col(x, y, grid):
    col = int(math.floor((x - grid["origin_x"]) / grid["cell"]))
    row = int(math.floor((grid["origin_y"] - y) / grid["cell"]))
    return min(max(row, 0), grid["height"] - 1), min(max(col, 0), grid["width"] - 1)


def geometry_parts(geometry):
    geometry_type = ogr.GT_Flatten(geometry.GetGeometryType())
    if geometry_type == ogr.wkbLineString:
        return [geometry]
    if geometry_type == ogr.wkbMultiLineString:
        return [geometry.GetGeometryRef(index) for index in range(geometry.GetGeometryCount())]
    return []


def sample_truth(grid, truth_id=None):
    ds = ogr.Open(TRUTH_SHP)
    if ds is None:
        raise RuntimeError("Cannot open truth shapefile: %s" % TRUTH_SHP)
    layer = ds.GetLayer(0)
    all_rows = []
    all_cols = []
    feature_slices = []
    feature_lengths = []
    cursor = 0

    for feature in layer:
        if truth_id is not None and int(feature.GetField("Id")) != int(truth_id):
            continue
        ogr_geom = feature.GetGeometryRef()
        if ogr_geom is None or ogr_geom.IsEmpty():
            continue
        parts = geometry_parts(ogr_geom)
        for part in parts:
            length = float(part.Length())
            if length <= 0:
                continue
            distances = np.arange(0.0, length, TRUTH_SAMPLE_INTERVAL_M, dtype=float)
            if len(distances) == 0 or distances[-1] < length:
                distances = np.append(distances, length)
            rows = np.empty(len(distances), dtype=np.int32)
            cols = np.empty(len(distances), dtype=np.int32)
            for index, distance in enumerate(distances):
                point = part.Value(float(distance))
                rows[index], cols[index] = xy_to_row_col(point.GetX(), point.GetY(), grid)
            all_rows.append(rows)
            all_cols.append(cols)
            feature_slices.append(slice(cursor, cursor + len(rows)))
            feature_lengths.append(length)
            cursor += len(rows)

    ds = None
    if not all_rows:
        raise RuntimeError("No truth samples for Id=%s" % str(truth_id))
    return {
        "rows": np.concatenate(all_rows),
        "cols": np.concatenate(all_cols),
        "feature_slices": feature_slices,
        "feature_lengths": np.asarray(feature_lengths, dtype=float),
    }


def count_internal_gaps(sample_distances, feature_slices, tolerance_m):
    gap_count = 0
    for feature_slice in feature_slices:
        matched = sample_distances[feature_slice] <= tolerance_m
        if len(matched) < 3 or matched.all() or (~matched).all():
            continue
        padded = np.concatenate(([True], matched, [True])).astype(np.int8)
        changes = np.diff(padded)
        starts = np.where(changes == -1)[0]
        ends = np.where(changes == 1)[0]
        for start, end in zip(starts, ends):
            if start <= 0 or end >= len(matched):
                continue
            gap_length = (end - start) * TRUTH_SAMPLE_INTERVAL_M
            if gap_length + 1e-9 < MIN_INTERNAL_GAP_M:
                continue
            gap_count += 1
    return {
        "internal_gap_count": int(gap_count),
        "estimated_breakpoint_count": int(2 * gap_count),
    }


def evaluate_mask(algorithm, threshold, pred_mask, truth_context):
    pred_pixels = int(pred_mask.sum())
    if pred_pixels <= 0:
        raise RuntimeError("Empty prediction for %s threshold=%s" % (algorithm, threshold))

    dist_to_pred = distance_transform_edt(
        ~pred_mask,
        sampling=(EVAL_CELL_SIZE_M, EVAL_CELL_SIZE_M),
    ).astype(np.float32)

    truth_samples = truth_context["all_samples"]
    all_distances = dist_to_pred[truth_samples["rows"], truth_samples["cols"]]
    pred_to_truth = truth_context["dist_to_truth"][pred_mask]
    recall = float(np.mean(all_distances <= TOLERANCE_M))
    precision = float(np.mean(pred_to_truth <= TOLERANCE_M))
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    pred_buffer = dist_to_pred <= TOLERANCE_M
    truth_buffer = truth_context["truth_buffer"]
    intersection = int(np.count_nonzero(pred_buffer & truth_buffer))
    union = int(np.count_nonzero(pred_buffer | truth_buffer))
    iou = float(intersection) / float(union) if union else 0.0
    dice_denominator = int(pred_buffer.sum()) + int(truth_buffer.sum())
    dice = 2.0 * float(intersection) / float(dice_denominator) if dice_denominator else 0.0

    all_gaps = count_internal_gaps(
        all_distances,
        truth_samples["feature_slices"],
        TOLERANCE_M,
    )

    row = {
        "algorithm": algorithm,
        "alpha_field_burn_m": ALPHA_M,
        "beta_channel_burn_m": BETA_M,
        "gamma_pond_burn_m": GAMMA_M,
        "flowacc_threshold_cells": int(threshold),
        "tolerance_m": TOLERANCE_M,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "buffer_overlap_iou": iou,
        "buffer_overlap_dice": dice,
        "truth_to_pred_mean_offset_m": float(all_distances.mean()),
        "pred_to_truth_mean_offset_m": float(pred_to_truth.mean()),
        "symmetric_mean_offset_m": 0.5 * (
            float(all_distances.mean()) + float(pred_to_truth.mean())
        ),
        "estimated_breakpoint_count": int(all_gaps["estimated_breakpoint_count"]),
        "pred_path_pixels_0p5m": pred_pixels,
        "pred_path_area_m2_0p5m": pred_pixels * EVAL_CELL_SIZE_M ** 2,
    }

    class_outputs = []
    for class_name, truth_id, prefix in [
        ("Field ditch", 0, "field"),
        ("Channel", 1, "channel"),
    ]:
        samples = truth_context["class_samples"][truth_id]
        distances = dist_to_pred[samples["rows"], samples["cols"]]
        gaps = count_internal_gaps(
            distances,
            samples["feature_slices"],
            TOLERANCE_M,
        )
        truth_length = float(samples["feature_lengths"].sum())
        class_values = {
            "recall": float(np.mean(distances <= TOLERANCE_M)),
            "truth_to_pred_mean_offset_m": float(distances.mean()),
            "estimated_breakpoint_count": int(gaps["estimated_breakpoint_count"]),
            "breakpoints_per_km": float(gaps["estimated_breakpoint_count"]) / truth_length * 1000.0,
        }
        for metric, value in class_values.items():
            row["%s_%s" % (prefix, metric)] = value
        class_outputs.append(class_values)

    row["macro_recall"] = 0.5 * (class_outputs[0]["recall"] + class_outputs[1]["recall"])
    row["macro_truth_to_pred_mean_offset_m"] = 0.5 * (
        class_outputs[0]["truth_to_pred_mean_offset_m"]
        + class_outputs[1]["truth_to_pred_mean_offset_m"]
    )
    row["macro_breakpoints_per_km"] = 0.5 * (
        class_outputs[0]["breakpoints_per_km"] + class_outputs[1]["breakpoints_per_km"]
    )

    del dist_to_pred, all_distances, pred_to_truth, pred_buffer
    gc.collect()
    return row


def make_outputs(result):
    result = result.sort_values(["flowacc_threshold_cells", "algorithm"]).reset_index(drop=True)
    result.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

    metric_columns = [
        "macro_recall",
        "recall",
        "precision",
        "f1",
        "buffer_overlap_iou",
        "symmetric_mean_offset_m",
        "estimated_breakpoint_count",
        "macro_truth_to_pred_mean_offset_m",
        "macro_breakpoints_per_km",
        "pred_path_pixels_0p5m",
    ]
    d8 = result[result["algorithm"] == "D8"][["flowacc_threshold_cells"] + metric_columns].copy()
    mfd = result[result["algorithm"] == "MFD"][["flowacc_threshold_cells"] + metric_columns].copy()
    d8 = d8.rename(columns={metric: "D8_%s" % metric for metric in metric_columns})
    mfd = mfd.rename(columns={metric: "MFD_%s" % metric for metric in metric_columns})
    pair = d8.merge(mfd, on="flowacc_threshold_cells", how="inner")
    for metric in metric_columns:
        pair["MFD_minus_D8_%s" % metric] = pair["MFD_%s" % metric] - pair["D8_%s" % metric]
    pair.to_csv(PAIR_CSV, index=False, encoding="utf-8-sig")

    mean_metrics = [
        "macro_recall",
        "precision",
        "f1",
        "buffer_overlap_iou",
        "symmetric_mean_offset_m",
        "estimated_breakpoint_count",
        "macro_breakpoints_per_km",
        "pred_path_pixels_0p5m",
    ]
    mean_table = (
        result.groupby("flowacc_threshold_cells", as_index=False)[mean_metrics]
        .mean()
        .rename(columns={metric: "algorithm_mean_%s" % metric for metric in mean_metrics})
    )
    mean_table.to_csv(MEAN_CSV, index=False, encoding="utf-8-sig")

    best_rows = []
    specs = [
        ("max_macro_recall", "macro_recall", False),
        ("max_precision", "precision", False),
        ("max_f1", "f1", False),
        ("max_path_iou", "buffer_overlap_iou", False),
        ("min_symmetric_offset", "symmetric_mean_offset_m", True),
        ("min_estimated_breakpoints", "estimated_breakpoint_count", True),
    ]
    for algorithm in ["D8", "MFD"]:
        subset = result[result["algorithm"] == algorithm]
        for criterion, metric, minimize in specs:
            index = subset[metric].idxmin() if minimize else subset[metric].idxmax()
            chosen = subset.loc[index]
            best_rows.append(
                {
                    "scope": algorithm,
                    "criterion": criterion,
                    "metric": metric,
                    "flowacc_threshold_cells": int(chosen["flowacc_threshold_cells"]),
                    "metric_value": float(chosen[metric]),
                }
            )
    for criterion, metric, minimize in specs:
        mean_metric = "algorithm_mean_%s" % metric
        index = mean_table[mean_metric].idxmin() if minimize else mean_table[mean_metric].idxmax()
        chosen = mean_table.loc[index]
        best_rows.append(
            {
                "scope": "D8_MFD_mean",
                "criterion": criterion,
                "metric": metric,
                "flowacc_threshold_cells": int(chosen["flowacc_threshold_cells"]),
                "metric_value": float(chosen[mean_metric]),
            }
        )
    pd.DataFrame(best_rows).to_csv(BEST_CSV, index=False, encoding="utf-8-sig")


def main():
    gdal.UseExceptions()
    ogr.UseExceptions()
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(MFD_FLOWACC):
        raise FileNotFoundError("Missing MFD FlowAcc raster: %s" % MFD_FLOWACC)
    missing_d8 = [path for path in [d8_line_path(value) for value in THRESHOLDS] if not os.path.exists(path)]
    if missing_d8:
        raise FileNotFoundError("Missing D8 threshold lines; run generator first: %s" % missing_d8[0])

    grid = build_grid(MFD_FLOWACC)
    assert_same_crs(TRUTH_SHP, grid["projection"])
    assert_same_crs(d8_line_path(THRESHOLDS[0]), grid["projection"])

    print("Preparing truth grid and samples...", flush=True)
    truth_mask = rasterize_lines(TRUTH_SHP, grid)
    dist_to_truth = distance_transform_edt(
        ~truth_mask,
        sampling=(EVAL_CELL_SIZE_M, EVAL_CELL_SIZE_M),
    ).astype(np.float32)
    truth_context = {
        "all_samples": sample_truth(grid),
        "class_samples": {
            0: sample_truth(grid, 0),
            1: sample_truth(grid, 1),
        },
        "dist_to_truth": dist_to_truth,
        "truth_buffer": dist_to_truth <= TOLERANCE_M,
    }

    rows = []
    total = len(THRESHOLDS) * 2
    count = 0
    for threshold in THRESHOLDS:
        for algorithm in ("D8", "MFD"):
            count += 1
            print("[%d/%d] %s FlowAcc > %d" % (count, total, algorithm, threshold), flush=True)
            if algorithm == "D8":
                pred_mask = rasterize_lines(d8_line_path(threshold), grid)
            else:
                pred_mask = aggregate_mfd_threshold(MFD_FLOWACC, grid, threshold)
            row = evaluate_mask(algorithm, threshold, pred_mask, truth_context)
            rows.append(row)
            print(
                "  macro_recall=%.4f precision=%.4f f1=%.4f iou=%.4f"
                % (row["macro_recall"], row["precision"], row["f1"], row["buffer_overlap_iou"]),
                flush=True,
            )
            del pred_mask
            gc.collect()

    result = pd.DataFrame(rows)
    make_outputs(result)
    print(RESULT_CSV, flush=True)
    print(PAIR_CSV, flush=True)
    print(MEAN_CSV, flush=True)
    print(BEST_CSV, flush=True)


if __name__ == "__main__":
    main()
