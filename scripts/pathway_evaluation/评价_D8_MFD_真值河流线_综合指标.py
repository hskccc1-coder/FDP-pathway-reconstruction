# -*- coding: utf-8 -*-
"""
用新的真值河流线对 D8 和 MFD 的 121 组烧录结果做统一评价。

评价指标：
1. 路径召回率：真值线等距采样点中，到预测路径距离 <= 容差的比例。
2. 路径精确率：预测路径像元中，到真值路径距离 <= 容差的比例。
3. F1：召回率和精确率的调和平均。
4. 路径重叠率：真值/预测线缓冲区的 IoU 和 Dice。
5. 路径偏移：真值->预测、预测->真值和对称平均偏移，以及中位数/P95。
6. 连续性代理指标：沿每条真值线统计被匹配段前后包围的内部未匹配段。
   internal_gap_count 是内部缺口数，estimated_breakpoint_count = 2 * gap_count。
   这是网络连续性指标，不是有人工断点标签时的断点检出准确率。

为避免内存溢出：
- 评价网格统一为 0.5 m。
- MFD 0.1 m FlowAcc 分块读取，每 5x5 原像元做“任一像元>1200”聚合。
- 每组完成后立即释放距离栅格和预测掩膜。
- CSV 逐组追加，中断后可继续。
"""

import gc
import glob
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr
from scipy.ndimage import distance_transform_edt
from shapely import wkb


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 1. 输入输出
# ============================================================
TRUTH_SHP = os.path.join(
    PROJECT_ROOT,
    "data",
    "reference_drainage",
    "pathway_evaluation_truth",
    "pathway_evaluation_truth.shp",
)
D8_DIR = os.environ.get(
    "FDP_D8_EVAL_INPUT",
    os.path.join(PROJECT_ROOT, "results", "pathway_metrics", "dem_routing_runtime", "d8"),
)
MFD_DIR = os.environ.get(
    "FDP_MFD_EVAL_INPUT",
    os.path.join(PROJECT_ROOT, "results", "pathway_metrics", "dem_routing_runtime", "mfd", "FlowAcc_MFD"),
)
OUT_DIR = os.path.join(
    PROJECT_ROOT, "results", "pathway_metrics", "dem_routing_runtime", "comprehensive_evaluation"
)

DETAIL_CSV = os.path.join(OUT_DIR, "D8_MFD_all_metrics.csv")
PRIMARY_CSV = os.path.join(OUT_DIR, "D8_MFD_primary_tolerance_1m.csv")
BEST_CSV = os.path.join(OUT_DIR, "D8_MFD_best_by_metric.csv")
PAIR_CSV = os.path.join(OUT_DIR, "D8_minus_MFD_same_AB_1m.csv")
METHOD_CSV = os.path.join(OUT_DIR, "D8_MFD_method_summary_1m.csv")


# ============================================================
# 2. 评价参数
# ============================================================
FLOWACC_THRESHOLD = 1200.0
EVAL_CELL_SIZE_M = 0.5
TRUTH_SAMPLE_INTERVAL_M = 0.5
TOLERANCES_M = [0.5, 1.0, 1.5]
PRIMARY_TOLERANCE_M = 1.0
MIN_INTERNAL_GAP_M = 1.0

# MFD 分块读取高度。必须是 5 的倍数（0.5/0.1=5）。
MFD_BLOCK_ROWS = 500


# ============================================================
# 3. 文件收集与参数解析
# ============================================================
def parse_ab_tag(a_tag, b_tag):
    return float(a_tag.replace("p", ".")), float(b_tag.replace("p", "."))


def collect_d8():
    pattern = re.compile(
        r"StreamLine_D8_thr1200_A([0-9]+p[0-9]+)_B([0-9]+p[0-9]+)\.shp$",
        re.IGNORECASE,
    )
    items = []
    for path in glob.glob(os.path.join(D8_DIR, "*.shp")):
        match = pattern.search(os.path.basename(path))
        if match:
            a, b = parse_ab_tag(match.group(1), match.group(2))
            items.append(("D8", a, b, path))
    return sorted(items, key=lambda x: (x[1], x[2]))


def collect_mfd():
    pattern = re.compile(
        r"FlowAcc_MFD_A([0-9]+p[0-9]+)_B([0-9]+p[0-9]+)\.tif$",
        re.IGNORECASE,
    )
    items = []
    for path in glob.glob(os.path.join(MFD_DIR, "*.tif")):
        match = pattern.search(os.path.basename(path))
        if match:
            a, b = parse_ab_tag(match.group(1), match.group(2))
            items.append(("MFD", a, b, path))
    return sorted(items, key=lambda x: (x[1], x[2]))


def completed_keys():
    if not os.path.exists(DETAIL_CSV):
        return set()
    old = pd.read_csv(DETAIL_CSV)
    required = len(TOLERANCES_M)
    grouped = old.groupby(["algorithm", "A", "B"])["tolerance_m"].nunique()
    return {
        (str(key[0]), round(float(key[1]), 10), round(float(key[2]), 10))
        for key, count in grouped.items()
        if int(count) >= required
    }


# ============================================================
# 4. 空间参考和评价网格
# ============================================================
def assert_same_crs(vector_path, raster_wkt):
    vector_ds = ogr.Open(vector_path)
    if vector_ds is None:
        raise RuntimeError(f"无法打开矢量: {vector_path}")
    layer = vector_ds.GetLayer(0)
    vector_srs = layer.GetSpatialRef()
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_wkt)
    same = bool(vector_srs and vector_srs.IsSame(raster_srs))
    vector_ds = None
    if not same:
        raise RuntimeError(f"坐标系不一致: {vector_path}")


def build_grid(sample_mfd_path):
    ds = gdal.Open(sample_mfd_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开 MFD 样例: {sample_mfd_path}")

    gt = ds.GetGeoTransform()
    source_cell_x = abs(gt[1])
    source_cell_y = abs(gt[5])
    if not np.isclose(source_cell_x, source_cell_y, atol=1e-8):
        raise RuntimeError("输入栅格不是正方形像元。")

    factor_float = EVAL_CELL_SIZE_M / source_cell_x
    factor = int(round(factor_float))
    if factor < 1 or not np.isclose(factor_float, factor, atol=1e-6):
        raise RuntimeError(
            f"评价分辨率 {EVAL_CELL_SIZE_M} m 必须是原始分辨率 "
            f"{source_cell_x} m 的整数倍。"
        )

    width = int(math.ceil(ds.RasterXSize / factor))
    height = int(math.ceil(ds.RasterYSize / factor))
    grid = {
        "origin_x": gt[0],
        "origin_y": gt[3],
        "cell": EVAL_CELL_SIZE_M,
        "width": width,
        "height": height,
        "geotransform": (gt[0], EVAL_CELL_SIZE_M, 0.0, gt[3], 0.0, -EVAL_CELL_SIZE_M),
        "projection": ds.GetProjection(),
        "source_width": ds.RasterXSize,
        "source_height": ds.RasterYSize,
        "source_cell": source_cell_x,
        "factor": factor,
    }
    ds = None
    return grid


def rasterize_lines(vector_path, grid):
    vector_ds = ogr.Open(vector_path)
    if vector_ds is None:
        raise RuntimeError(f"无法打开矢量: {vector_path}")
    layer = vector_ds.GetLayer(0)

    mem = gdal.GetDriverByName("MEM").Create(
        "", grid["width"], grid["height"], 1, gdal.GDT_Byte
    )
    mem.SetGeoTransform(grid["geotransform"])
    mem.SetProjection(grid["projection"])
    band = mem.GetRasterBand(1)
    band.Fill(0)

    error = gdal.RasterizeLayer(
        mem,
        [1],
        layer,
        burn_values=[1],
        options=["ALL_TOUCHED=TRUE"],
    )
    if error != 0:
        raise RuntimeError(f"栅格化失败: {vector_path}")

    result = band.ReadAsArray().astype(bool)
    vector_ds = None
    mem = None
    return result


def aggregate_mfd_threshold(mfd_path, grid):
    """0.1 m FlowAcc 分块阈值化，再用 5x5 any 聚合到 0.5 m。"""
    ds = gdal.Open(mfd_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开 MFD: {mfd_path}")

    if ds.RasterXSize != grid["source_width"] or ds.RasterYSize != grid["source_height"]:
        raise RuntimeError(f"MFD 尺寸不一致: {mfd_path}")

    factor = grid["factor"]
    if MFD_BLOCK_ROWS % factor != 0:
        raise RuntimeError("MFD_BLOCK_ROWS 必须是聚合倍数的整数倍。")

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
        fine = valid & (data > FLOWACC_THRESHOLD)
        del data

        pad_rows = (-fine.shape[0]) % factor
        pad_cols = (-fine.shape[1]) % factor
        if pad_rows or pad_cols:
            fine = np.pad(
                fine,
                ((0, pad_rows), (0, pad_cols)),
                mode="constant",
                constant_values=False,
            )

        coarse = fine.reshape(
            fine.shape[0] // factor,
            factor,
            fine.shape[1] // factor,
            factor,
        ).any(axis=(1, 3))

        out_row = y0 // factor
        out[out_row:out_row + coarse.shape[0], :coarse.shape[1]] = coarse
        del fine
        del coarse

    ds = None
    return out


# ============================================================
# 5. 真值等距采样和连续性
# ============================================================
def xy_to_row_col(x, y, grid):
    col = int(math.floor((x - grid["origin_x"]) / grid["cell"]))
    row = int(math.floor((grid["origin_y"] - y) / grid["cell"]))
    col = min(max(col, 0), grid["width"] - 1)
    row = min(max(row, 0), grid["height"] - 1)
    return row, col


def sample_truth_features(grid):
    ds = ogr.Open(TRUTH_SHP)
    if ds is None:
        raise RuntimeError(f"无法打开真值: {TRUTH_SHP}")
    layer = ds.GetLayer(0)

    all_rows = []
    all_cols = []
    feature_slices = []
    feature_lengths = []
    cursor = 0

    for feature in layer:
        ogr_geom = feature.GetGeometryRef()
        if ogr_geom is None or ogr_geom.IsEmpty():
            continue
        geom = wkb.loads(bytes(ogr_geom.ExportToWkb()))
        if geom.is_empty or geom.length <= 0:
            continue

        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            length = float(part.length)
            if length <= 0:
                continue
            distances = np.arange(0.0, length, TRUTH_SAMPLE_INTERVAL_M, dtype=float)
            if len(distances) == 0 or distances[-1] < length:
                distances = np.append(distances, length)

            rows = np.empty(len(distances), dtype=np.int32)
            cols = np.empty(len(distances), dtype=np.int32)
            for i, distance in enumerate(distances):
                point = part.interpolate(float(distance))
                rows[i], cols[i] = xy_to_row_col(point.x, point.y, grid)

            all_rows.append(rows)
            all_cols.append(cols)
            feature_slices.append(slice(cursor, cursor + len(rows)))
            feature_lengths.append(length)
            cursor += len(rows)

    ds = None
    if not all_rows:
        raise RuntimeError("真值线没有可用几何。")

    return {
        "rows": np.concatenate(all_rows),
        "cols": np.concatenate(all_cols),
        "feature_slices": feature_slices,
        "feature_lengths": np.asarray(feature_lengths, dtype=float),
    }


def count_internal_gaps(sample_distances, feature_slices, tolerance_m):
    gap_count = 0
    affected_features = 0
    total_gap_length = 0.0
    max_gap_length = 0.0

    for feature_slice in feature_slices:
        matched = sample_distances[feature_slice] <= tolerance_m
        if len(matched) < 3 or matched.all() or (~matched).all():
            continue

        padded = np.concatenate(([True], matched, [True])).astype(np.int8)
        changes = np.diff(padded)
        starts = np.where(changes == -1)[0]
        ends = np.where(changes == 1)[0]
        feature_has_gap = False

        for start, end in zip(starts, ends):
            # 只计算前后都有匹配段的内部缺口，排除两端末梢遗漏。
            if start <= 0 or end >= len(matched):
                continue
            gap_length = (end - start) * TRUTH_SAMPLE_INTERVAL_M
            if gap_length + 1e-9 < MIN_INTERNAL_GAP_M:
                continue
            gap_count += 1
            feature_has_gap = True
            total_gap_length += gap_length
            max_gap_length = max(max_gap_length, gap_length)

        if feature_has_gap:
            affected_features += 1

    return {
        "internal_gap_count": int(gap_count),
        "estimated_breakpoint_count": int(2 * gap_count),
        "features_with_internal_gap": int(affected_features),
        "internal_gap_total_length_m": float(total_gap_length),
        "internal_gap_max_length_m": float(max_gap_length),
    }


# ============================================================
# 6. 单个预测结果的指标
# ============================================================
def safe_percentile(values, percentile):
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def evaluate_prediction(pred_mask, truth_mask, dist_to_truth, truth_samples, truth_buffers):
    pred_pixels = int(pred_mask.sum())
    if pred_pixels <= 0:
        raise RuntimeError("预测路径为空。")

    dist_to_pred = distance_transform_edt(
        ~pred_mask,
        sampling=(EVAL_CELL_SIZE_M, EVAL_CELL_SIZE_M),
    ).astype(np.float32)

    sample_dist = dist_to_pred[truth_samples["rows"], truth_samples["cols"]]
    pred_to_truth_dist = dist_to_truth[pred_mask]

    truth_to_pred_mean = float(sample_dist.mean())
    pred_to_truth_mean = float(pred_to_truth_dist.mean())
    symmetric_mean = 0.5 * (truth_to_pred_mean + pred_to_truth_mean)

    gap_metrics = count_internal_gaps(
        sample_dist,
        truth_samples["feature_slices"],
        PRIMARY_TOLERANCE_M,
    )

    common = {
        "eval_cell_size_m": EVAL_CELL_SIZE_M,
        "truth_sample_interval_m": TRUTH_SAMPLE_INTERVAL_M,
        "flowacc_threshold": FLOWACC_THRESHOLD,
        "truth_sample_count": int(sample_dist.size),
        "pred_path_pixels": pred_pixels,
        "pred_path_area_m2": pred_pixels * EVAL_CELL_SIZE_M * EVAL_CELL_SIZE_M,
        "truth_to_pred_mean_offset_m": truth_to_pred_mean,
        "truth_to_pred_median_offset_m": safe_percentile(sample_dist, 50),
        "truth_to_pred_p95_offset_m": safe_percentile(sample_dist, 95),
        "pred_to_truth_mean_offset_m": pred_to_truth_mean,
        "pred_to_truth_median_offset_m": safe_percentile(pred_to_truth_dist, 50),
        "pred_to_truth_p95_offset_m": safe_percentile(pred_to_truth_dist, 95),
        "symmetric_mean_offset_m": symmetric_mean,
        "primary_gap_tolerance_m": PRIMARY_TOLERANCE_M,
        "min_internal_gap_m": MIN_INTERNAL_GAP_M,
        **gap_metrics,
    }

    rows = []
    for tolerance in TOLERANCES_M:
        truth_matched = sample_dist <= tolerance
        pred_matched = pred_to_truth_dist <= tolerance
        recall = float(truth_matched.mean())
        precision = float(pred_matched.mean())
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        pred_buffer = dist_to_pred <= tolerance
        truth_buffer = truth_buffers[tolerance]
        intersection = int(np.count_nonzero(pred_buffer & truth_buffer))
        union = int(np.count_nonzero(pred_buffer | truth_buffer))
        pred_buffer_pixels = int(pred_buffer.sum())
        truth_buffer_pixels = int(truth_buffer.sum())
        iou = intersection / union if union else 0.0
        dice_denominator = pred_buffer_pixels + truth_buffer_pixels
        dice = 2.0 * intersection / dice_denominator if dice_denominator else 0.0

        rows.append(
            {
                "tolerance_m": tolerance,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "buffer_overlap_iou": iou,
                "buffer_overlap_dice": dice,
                "truth_matched_samples": int(truth_matched.sum()),
                "pred_matched_pixels": int(pred_matched.sum()),
                "buffer_intersection_pixels": intersection,
                "buffer_union_pixels": union,
                **common,
            }
        )

        del pred_buffer

    del dist_to_pred
    del sample_dist
    del pred_to_truth_dist
    gc.collect()
    return rows


# ============================================================
# 7. 汇总和绘图
# ============================================================
def append_rows(rows):
    header = not os.path.exists(DETAIL_CSV)
    pd.DataFrame(rows).to_csv(
        DETAIL_CSV,
        mode="a",
        header=header,
        index=False,
        encoding="utf-8-sig",
    )


def make_best_table(primary):
    specs = [
        ("max_recall", "recall", False),
        ("max_precision", "precision", False),
        ("max_f1", "f1", False),
        ("max_overlap_iou", "buffer_overlap_iou", False),
        ("min_symmetric_offset", "symmetric_mean_offset_m", True),
        ("min_breakpoints", "estimated_breakpoint_count", True),
    ]
    rows = []
    for algorithm, group in primary.groupby("algorithm"):
        for selection, column, ascending in specs:
            ranked = group.sort_values(
                [column, "f1", "A", "B"],
                ascending=[ascending, False, True, True],
            )
            best = ranked.iloc[0]
            rows.append(
                {
                    "algorithm": algorithm,
                    "selection": selection,
                    "best_A": best["A"],
                    "best_B": best["B"],
                    "recall": best["recall"],
                    "precision": best["precision"],
                    "f1": best["f1"],
                    "buffer_overlap_iou": best["buffer_overlap_iou"],
                    "symmetric_mean_offset_m": best["symmetric_mean_offset_m"],
                    "estimated_breakpoint_count": best["estimated_breakpoint_count"],
                }
            )
    return pd.DataFrame(rows)


def make_pair_comparison(primary):
    d8 = primary[primary["algorithm"] == "D8"].copy()
    mfd = primary[primary["algorithm"] == "MFD"].copy()
    metric_cols = [
        "recall",
        "precision",
        "f1",
        "buffer_overlap_iou",
        "buffer_overlap_dice",
        "truth_to_pred_mean_offset_m",
        "pred_to_truth_mean_offset_m",
        "symmetric_mean_offset_m",
        "internal_gap_count",
        "estimated_breakpoint_count",
    ]
    d8 = d8[["A", "B"] + metric_cols].rename(
        columns={column: f"D8_{column}" for column in metric_cols}
    )
    mfd = mfd[["A", "B"] + metric_cols].rename(
        columns={column: f"MFD_{column}" for column in metric_cols}
    )
    merged = d8.merge(mfd, on=["A", "B"], how="inner")
    for column in metric_cols:
        merged[f"D8_minus_MFD_{column}"] = merged[f"D8_{column}"] - merged[f"MFD_{column}"]
    return merged


def make_method_summary(primary):
    metrics = [
        "recall",
        "precision",
        "f1",
        "buffer_overlap_iou",
        "buffer_overlap_dice",
        "truth_to_pred_mean_offset_m",
        "pred_to_truth_mean_offset_m",
        "symmetric_mean_offset_m",
        "internal_gap_count",
        "estimated_breakpoint_count",
    ]
    rows = []
    for algorithm, group in primary.groupby("algorithm"):
        for metric in metrics:
            values = group[metric].astype(float)
            rows.append(
                {
                    "algorithm": algorithm,
                    "metric": metric,
                    "mean_over_121": values.mean(),
                    "median_over_121": values.median(),
                    "std_over_121": values.std(ddof=1),
                    "min_over_121": values.min(),
                    "max_over_121": values.max(),
                }
            )
    return pd.DataFrame(rows)


def plot_heatmaps(primary):
    plot_specs = [
        ("recall", "Recall", 0.0, 1.0, "viridis"),
        ("precision", "Precision", 0.0, 1.0, "viridis"),
        ("f1", "F1", 0.0, 1.0, "viridis"),
        ("buffer_overlap_iou", "Buffered path IoU", 0.0, 1.0, "viridis"),
        ("symmetric_mean_offset_m", "Symmetric mean offset (m)", None, None, "magma_r"),
        ("estimated_breakpoint_count", "Estimated breakpoint count", None, None, "magma_r"),
    ]

    for algorithm in ["D8", "MFD"]:
        subset = primary[primary["algorithm"] == algorithm]
        for column, label, vmin, vmax, cmap in plot_specs:
            pivot = subset.pivot(index="A", columns="B", values=column).sort_index(ascending=False)
            pivot.to_csv(
                os.path.join(OUT_DIR, f"HeatmapTable_{algorithm}_{column}.csv"),
                encoding="utf-8-sig",
            )
            fig, ax = plt.subplots(figsize=(7.2, 6.1), dpi=180)
            image = ax.imshow(
                pivot.values,
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{algorithm}: {label} (tolerance={PRIMARY_TOLERANCE_M:g} m)")
            ax.set_xlabel("B channel burn depth (m)")
            ax.set_ylabel("A field-ditch burn depth (m)")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([f"{x:g}" for x in pivot.columns], rotation=45, ha="right")
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels([f"{x:g}" for x in pivot.index])
            fig.colorbar(image, ax=ax, label=label)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"Heatmap_{algorithm}_{column}.png"))
            plt.close(fig)


# ============================================================
# 8. 主程序
# ============================================================
def main():
    gdal.UseExceptions()
    ogr.UseExceptions()
    os.makedirs(OUT_DIR, exist_ok=True)

    d8_items = collect_d8()
    mfd_items = collect_mfd()
    if not d8_items:
        raise RuntimeError(f"未找到 D8 SHP: {D8_DIR}")
    if not mfd_items:
        raise RuntimeError(f"未找到 MFD TIF: {MFD_DIR}")

    print(f"D8: {len(d8_items)} groups", flush=True)
    print(f"MFD: {len(mfd_items)} groups", flush=True)
    if len(d8_items) != 121 or len(mfd_items) != 121:
        print("Warning: expected 121 groups for each algorithm.", flush=True)

    grid = build_grid(mfd_items[0][3])
    assert_same_crs(TRUTH_SHP, grid["projection"])
    for _, _, _, path in d8_items[:1]:
        assert_same_crs(path, grid["projection"])

    print(
        f"Evaluation grid: {grid['width']} x {grid['height']}, "
        f"cell={grid['cell']} m, aggregation factor={grid['factor']}",
        flush=True,
    )

    print("Rasterizing truth...", flush=True)
    truth_mask = rasterize_lines(TRUTH_SHP, grid)
    truth_pixels = int(truth_mask.sum())
    if truth_pixels <= 0:
        raise RuntimeError("真值栅格为空。")

    print("Sampling truth lines...", flush=True)
    truth_samples = sample_truth_features(grid)
    print(
        f"Truth features/parts: {len(truth_samples['feature_slices'])}, "
        f"samples: {len(truth_samples['rows'])}, "
        f"length: {truth_samples['feature_lengths'].sum():.3f} m",
        flush=True,
    )

    print("Computing distance-to-truth raster...", flush=True)
    dist_to_truth = distance_transform_edt(
        ~truth_mask,
        sampling=(EVAL_CELL_SIZE_M, EVAL_CELL_SIZE_M),
    ).astype(np.float32)
    truth_buffers = {
        tolerance: dist_to_truth <= tolerance
        for tolerance in TOLERANCES_M
    }

    done = completed_keys()
    all_items = d8_items + mfd_items
    print(f"Completed groups found: {len(done)}", flush=True)

    for index, (algorithm, a, b, path) in enumerate(all_items, start=1):
        key = (algorithm, round(a, 10), round(b, 10))
        print(
            f"[{index}/{len(all_items)}] {algorithm} A={a:g}, B={b:g}",
            flush=True,
        )
        if key in done:
            print("  skip", flush=True)
            continue

        if algorithm == "D8":
            pred_mask = rasterize_lines(path, grid)
        else:
            pred_mask = aggregate_mfd_threshold(path, grid)

        rows = evaluate_prediction(
            pred_mask,
            truth_mask,
            dist_to_truth,
            truth_samples,
            truth_buffers,
        )
        for row in rows:
            row.update(
                {
                    "algorithm": algorithm,
                    "A": a,
                    "B": b,
                    "prediction_file": os.path.basename(path),
                }
            )
        append_rows(rows)
        primary_row = min(rows, key=lambda x: abs(x["tolerance_m"] - PRIMARY_TOLERANCE_M))
        print(
            "  "
            f"R={primary_row['recall']:.4f}, "
            f"P={primary_row['precision']:.4f}, "
            f"F1={primary_row['f1']:.4f}, "
            f"IoU={primary_row['buffer_overlap_iou']:.4f}, "
            f"offset={primary_row['symmetric_mean_offset_m']:.3f} m, "
            f"breaks={primary_row['estimated_breakpoint_count']}",
            flush=True,
        )

        done.add(key)
        del pred_mask
        del rows
        gc.collect()

    detail = pd.read_csv(DETAIL_CSV)
    detail = detail.sort_values(["algorithm", "A", "B", "tolerance_m"])
    detail.to_csv(DETAIL_CSV, index=False, encoding="utf-8-sig")

    primary = detail[np.isclose(detail["tolerance_m"], PRIMARY_TOLERANCE_M)].copy()
    primary.to_csv(PRIMARY_CSV, index=False, encoding="utf-8-sig")

    best = make_best_table(primary)
    best.to_csv(BEST_CSV, index=False, encoding="utf-8-sig")

    pair = make_pair_comparison(primary)
    pair.to_csv(PAIR_CSV, index=False, encoding="utf-8-sig")

    summary = make_method_summary(primary)
    summary.to_csv(METHOD_CSV, index=False, encoding="utf-8-sig")

    plot_heatmaps(primary)

    print("\nBest configurations:", flush=True)
    print(best.to_string(index=False), flush=True)
    print("\nOutputs:", flush=True)
    for path in [DETAIL_CSV, PRIMARY_CSV, BEST_CSV, PAIR_CSV, METHOD_CSV]:
        print(path, flush=True)


if __name__ == "__main__":
    main()
