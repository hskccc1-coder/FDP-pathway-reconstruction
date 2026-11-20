# -*- coding: utf-8 -*-
"""D8/MFD 在 121 组烧录深度下的论文风格可视化。"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(
    PACKAGE_ROOT,
    "results",
    "pathway_metrics",
    "dem_routing",
    "D8_MFD_真值河流线_综合评价_thr1200",
    "D8_MFD_primary_tolerance_1m.csv",
)
OUT_DIR = os.path.join(
    PACKAGE_ROOT,
    "results",
    "pathway_metrics",
    "figures",
    "D8_MFD_烧录深度可视化_thr1200",
)

RECOMMENDED_A = 0.8
RECOMMENDED_B = 1.2
STABLE_A_MIN = 0.8
STABLE_B_MIN = 1.0


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "SimSun", "DejaVu Serif"],
        "font.size": 9.0,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_data():
    data = pd.read_csv(INPUT_CSV)
    expected = 2 * 11 * 11
    if len(data) != expected:
        raise RuntimeError(f"期望 {expected} 行，实际 {len(data)} 行。")
    if set(data["algorithm"]) != {"D8", "MFD"}:
        raise RuntimeError("算法必须同时包含 D8 和 MFD。")
    return data


def matrix(data, algorithm, metric):
    table = (
        data[data["algorithm"] == algorithm]
        .pivot(index="B", columns="A", values=metric)
        .sort_index()
        .sort_index(axis=1)
    )
    return table.index.to_numpy(float), table.columns.to_numpy(float), table.to_numpy(float)


def cell_edges(values):
    step = float(np.median(np.diff(values)))
    return np.r_[values - step / 2.0, values[-1] + step / 2.0]


def text_color_for_value(cmap, norm, value):
    r, g, b, _ = cmap(norm(value))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.58 else "white"


def add_cell_labels(ax, xs, ys, values, cmap, norm, scale=100.0, decimals=1):
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            value = values[row, col]
            ax.text(
                x,
                y,
                f"{value * scale:.{decimals}f}",
                ha="center",
                va="center",
                color=text_color_for_value(cmap, norm, value),
                fontsize=7.1,
                fontweight="bold",
            )


def add_stable_region(ax, xs, ys, label=True):
    step_x = float(np.median(np.diff(xs)))
    step_y = float(np.median(np.diff(ys)))
    left = STABLE_A_MIN - step_x / 2.0
    bottom = STABLE_B_MIN - step_y / 2.0
    right = xs.max() + step_x / 2.0
    top = ys.max() + step_y / 2.0
    ax.add_patch(
        Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            fill=False,
            edgecolor="#d62728",
            linewidth=1.5,
            linestyle=(0, (3, 2)),
            zorder=5,
        )
    )
    if label:
        ax.text(
            right - 0.02,
            top - 0.02,
            "Stable region",
            color="#d62728",
            ha="right",
            va="top",
            fontsize=8.0,
            fontstyle="italic",
            fontweight="bold",
            zorder=6,
        )


def add_selected_cell(ax, xs, ys, a=RECOMMENDED_A, b=RECOMMENDED_B):
    step_x = float(np.median(np.diff(xs)))
    step_y = float(np.median(np.diff(ys)))
    ax.add_patch(
        Rectangle(
            (a - step_x / 2.0, b - step_y / 2.0),
            step_x,
            step_y,
            fill=False,
            edgecolor="#d62728",
            linewidth=2.0,
            zorder=7,
        )
    )


def format_depth_axes(ax, xs, ys, title=None):
    ax.set_xticks(xs)
    ax.set_yticks(ys)
    ax.set_xticklabels([f"{v:g}" for v in xs], rotation=0)
    ax.set_yticklabels([f"{v:g}" for v in ys])
    ax.set_xlabel(r"$A$ field-ditch burning depth (m)")
    ax.set_ylabel(r"$B$ channel burning depth (m)")
    if title:
        ax.set_title(title, pad=6, fontweight="bold")
    ax.set_aspect("equal")
    ax.tick_params(length=3, width=0.8, labelsize=8)


def draw_recall_panel(ax, data, algorithm, vmin, vmax, title):
    ys, xs, z = matrix(data, algorithm, "recall")
    x_edges = cell_edges(xs)
    y_edges = cell_edges(ys)
    cmap = plt.get_cmap("YlGnBu")
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        z,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        edgecolors=(1, 1, 1, 0.18),
        linewidth=0.25,
    )
    add_cell_labels(ax, xs, ys, z, cmap, mesh.norm)
    add_stable_region(ax, xs, ys)
    add_selected_cell(ax, xs, ys)
    format_depth_axes(ax, xs, ys, title)
    return mesh


def save_figure(fig, stem):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=350, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, stem + ".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_recall_figures(data):
    vmin = float(data["recall"].min())
    vmax = float(data["recall"].max())

    for algorithm in ("D8", "MFD"):
        fig, ax = plt.subplots(figsize=(6.2, 5.25))
        mesh = draw_recall_panel(
            ax,
            data,
            algorithm,
            vmin,
            vmax,
            f"{algorithm} recall at FlowAcc > 1200",
        )
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.035)
        cbar.set_label("Recall")
        ax.text(
            RECOMMENDED_A + 0.14,
            RECOMMENDED_B + 0.01,
            r"Selected: $A=0.8$ m, $B=1.2$ m",
            color="#d62728",
            ha="left",
            va="center",
            fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="#d62728", linewidth=0.7, pad=1.2),
            zorder=8,
        )
        fig.tight_layout()
        save_figure(fig, f"Fig01_{algorithm}_Recall_Heatmap")

    fig = plt.figure(figsize=(11.5, 5.1))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.045), wspace=0.10)
    axes = np.empty(2, dtype=object)
    axes[0] = fig.add_subplot(grid[0, 0])
    axes[1] = fig.add_subplot(grid[0, 1], sharex=axes[0], sharey=axes[0])
    color_axis = fig.add_subplot(grid[0, 2])
    meshes = []
    for ax, algorithm in zip(axes, ("D8", "MFD")):
        meshes.append(
            draw_recall_panel(
                ax,
                data,
                algorithm,
                vmin,
                vmax,
                f"({chr(97 + len(meshes))}) {algorithm}",
            )
        )
    axes[1].set_ylabel("")
    cbar = fig.colorbar(meshes[-1], cax=color_axis)
    cbar.set_label("Recall")
    fig.suptitle("Recall across burning-depth combinations", y=0.995, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.12, top=0.90)
    save_figure(fig, "MainFig1_Recall_Parameter_Sensitivity")


def draw_standard_heatmap(ax, data, algorithm, metric, title, cmap, vmin, vmax, percent=True):
    ys, xs, z = matrix(data, algorithm, metric)
    display_scale = 100.0 if percent else 1.0
    display_z = z * display_scale
    mesh = ax.pcolormesh(
        cell_edges(xs),
        cell_edges(ys),
        display_z,
        cmap=cmap,
        vmin=vmin * display_scale,
        vmax=vmax * display_scale,
        shading="flat",
    )
    add_cell_labels(
        ax,
        xs,
        ys,
        display_z,
        plt.get_cmap(cmap),
        mesh.norm,
        scale=1.0,
        decimals=1 if percent else 2,
    )
    add_selected_cell(ax, xs, ys)
    format_depth_axes(ax, xs, ys, title)
    return mesh


def plot_f1_iou(data):
    specifications = [
        ("f1", "F1 score (%)", "YlGnBu"),
        ("buffer_overlap_iou", "Path overlap IoU (%)", "YlGnBu"),
    ]
    fig = plt.figure(figsize=(11.2, 9.2))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.045),
        hspace=0.24,
        wspace=0.12,
    )
    axes = np.empty((2, 2), dtype=object)
    color_axes = []
    for row in range(2):
        axes[row, 0] = fig.add_subplot(grid[row, 0])
        axes[row, 1] = fig.add_subplot(grid[row, 1], sharex=axes[row, 0], sharey=axes[row, 0])
        color_axes.append(fig.add_subplot(grid[row, 2]))
    for row, (metric, label, cmap) in enumerate(specifications):
        vmin = float(data[metric].min())
        vmax = float(data[metric].max())
        row_meshes = []
        for col, algorithm in enumerate(("D8", "MFD")):
            row_meshes.append(
                draw_standard_heatmap(
                    axes[row, col],
                    data,
                    algorithm,
                    metric,
                    f"{algorithm}: {label}",
                    cmap,
                    vmin,
                    vmax,
                )
            )
            if col == 1:
                axes[row, col].set_ylabel("")
        cbar = fig.colorbar(row_meshes[-1], cax=color_axes[row])
        cbar.set_label(label)
    fig.suptitle("Accuracy and path-overlap response", y=0.995, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.08, top=0.94)
    save_figure(fig, "Fig02_F1_IoU_D8_MFD_Heatmaps")


def paired_grid(data, metric, expression):
    pivot = data.pivot(index=["A", "B"], columns="algorithm", values=metric)
    if expression == "MFD-D8":
        values = pivot["MFD"] - pivot["D8"]
    elif expression == "D8-MFD":
        values = pivot["D8"] - pivot["MFD"]
    else:
        raise ValueError(expression)
    return values.unstack("A").sort_index().sort_index(axis=1)


def plot_difference_heatmaps(data):
    # 正值统一表示 MFD 更优。对“越低越好”指标使用 D8-MFD。
    specs = [
        ("recall", "MFD − D8 recall (pp)", "MFD-D8", 100.0),
        ("precision", "MFD − D8 precision (pp)", "MFD-D8", 100.0),
        ("f1", "MFD − D8 F1 (pp)", "MFD-D8", 100.0),
        ("buffer_overlap_iou", "MFD − D8 IoU (pp)", "MFD-D8", 100.0),
        (
            "symmetric_mean_offset_m",
            "MFD offset reduction (m)",
            "D8-MFD",
            1.0,
        ),
        (
            "estimated_breakpoint_count",
            "MFD breakpoint reduction",
            "D8-MFD",
            1.0,
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0), sharex=True, sharey=True)
    for ax, (metric, title, expression, scale) in zip(axes.ravel(), specs):
        table = paired_grid(data, metric, expression)
        xs = table.columns.to_numpy(float)
        ys = table.index.to_numpy(float)
        z = table.to_numpy(float) * scale
        limit = float(np.max(np.abs(z)))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        mesh = ax.pcolormesh(
            cell_edges(xs),
            cell_edges(ys),
            z,
            cmap="RdBu_r",
            norm=norm,
            shading="flat",
        )
        format_depth_axes(ax, xs, ys, title)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.044, pad=0.025)
        cbar.ax.tick_params(labelsize=7)
    for ax in axes[0, :]:
        ax.set_xlabel("")
    for ax in axes[:, 1:].ravel():
        ax.set_ylabel("")
    fig.suptitle(
        "Paired algorithm differences (positive values favor MFD)",
        y=0.995,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.09, top=0.93, hspace=0.24, wspace=0.30)
    save_figure(fig, "Fig03_D8_MFD_Paired_Difference_Heatmaps")


def plot_marginal_response(data):
    metrics = [
        ("recall", "Recall (%)", 100.0),
        ("f1", "F1 score (%)", 100.0),
        ("estimated_breakpoint_count", "Estimated breakpoints", 1.0),
    ]
    dimensions = [
        ("A", r"$A$ field-ditch burning depth (m)"),
        ("B", r"$B$ channel burning depth (m)"),
    ]
    colors = {"D8": "#2166ac", "MFD": "#d6604d"}
    markers = {"D8": "o", "MFD": "s"}

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), sharex="row")
    for row, (dimension, xlabel) in enumerate(dimensions):
        grouped = data.groupby(["algorithm", dimension], as_index=False).mean(numeric_only=True)
        for col, (metric, ylabel, scale) in enumerate(metrics):
            ax = axes[row, col]
            for algorithm in ("D8", "MFD"):
                group = grouped[grouped["algorithm"] == algorithm].sort_values(dimension)
                ax.plot(
                    group[dimension],
                    group[metric] * scale,
                    color=colors[algorithm],
                    marker=markers[algorithm],
                    markersize=4.0,
                    linewidth=1.5,
                    label=algorithm,
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(color="#d9d9d9", linewidth=0.6, alpha=0.75)
            ax.set_xlim(-0.03, 2.03)
            ax.set_xticks(np.arange(0, 2.01, 0.2))
            ax.tick_params(axis="x", labelrotation=45, labelsize=7.5)
            if row == 0:
                ax.set_title(ylabel, fontweight="bold")
            if col == 0:
                ax.legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Marginal response to burning depth (mean over the other depth)",
        y=0.995,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, "Fig04_Burning_Depth_Marginal_Response")


def plot_pair_difference_distributions(data):
    specs = [
        ("recall", "MFD − D8 recall (pp)", "MFD-D8", 100.0),
        ("precision", "MFD − D8 precision (pp)", "MFD-D8", 100.0),
        ("f1", "MFD − D8 F1 (pp)", "MFD-D8", 100.0),
        ("buffer_overlap_iou", "MFD − D8 IoU (pp)", "MFD-D8", 100.0),
        ("symmetric_mean_offset_m", "MFD offset reduction (m)", "D8-MFD", 1.0),
        ("estimated_breakpoint_count", "MFD breakpoint reduction", "D8-MFD", 1.0),
    ]
    rng = np.random.default_rng(20261022)
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.7))
    for ax, (metric, title, expression, scale) in zip(axes.ravel(), specs):
        pivot = data.pivot(index=["A", "B"], columns="algorithm", values=metric)
        if expression == "MFD-D8":
            values = (pivot["MFD"] - pivot["D8"]).to_numpy(float) * scale
        else:
            values = (pivot["D8"] - pivot["MFD"]).to_numpy(float) * scale
        ax.boxplot(
            values,
            vert=True,
            widths=0.32,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="#9ecae1", edgecolor="#2166ac", linewidth=1.0),
            medianprops=dict(color="#b2182b", linewidth=1.5),
            whiskerprops=dict(color="#2166ac"),
            capprops=dict(color="#2166ac"),
        )
        jitter = rng.normal(1.0, 0.045, size=len(values))
        ax.scatter(jitter, values, s=8, color="#4d4d4d", alpha=0.30, linewidths=0)
        ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xticks([])
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.75)
        ax.text(
            0.98,
            0.96,
            f"mean = {values.mean():.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
    fig.suptitle(
        "Distribution of paired differences across 121 burning-depth combinations\n"
        "Positive values favor MFD",
        y=0.995,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "Fig05_Paired_Difference_Distributions")


def plot_main_comparison_summary(data):
    """正文主图2：用四个面板取代多张指标热图。"""
    colors = {"D8": "#2166ac", "MFD": "#d6604d"}
    markers = {"D8": "o", "MFD": "s"}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.7))

    # (a)(b) A/B 的 F1 边际响应，使用同一纵轴便于直接比较效应强弱。
    marginal_tables = {}
    for dimension in ("A", "B"):
        marginal_tables[dimension] = (
            data.groupby(["algorithm", dimension], as_index=False)
            .mean(numeric_only=True)
            .sort_values(["algorithm", dimension])
        )
    f1_min = min(table["f1"].min() for table in marginal_tables.values()) * 100.0
    f1_max = max(table["f1"].max() for table in marginal_tables.values()) * 100.0

    for ax, dimension, panel, stable_start, xlabel in [
        (axes[0, 0], "A", "(a)", STABLE_A_MIN, r"$A$ field-ditch burning depth (m)"),
        (axes[0, 1], "B", "(b)", STABLE_B_MIN, r"$B$ channel burning depth (m)"),
    ]:
        table = marginal_tables[dimension]
        ax.axvspan(stable_start, 2.0, color="#eeeeee", alpha=0.75, zorder=0)
        for algorithm in ("D8", "MFD"):
            group = table[table["algorithm"] == algorithm]
            ax.plot(
                group[dimension],
                group["f1"] * 100.0,
                color=colors[algorithm],
                marker=markers[algorithm],
                markersize=4.2,
                linewidth=1.6,
                label=algorithm,
                zorder=3,
            )
        ax.axvline(stable_start, color="#777777", linewidth=0.9, linestyle="--")
        ax.text(
            stable_start + 0.04,
            f1_min + 0.05 * (f1_max - f1_min),
            "stable",
            color="#666666",
            fontsize=8,
            rotation=90,
            va="bottom",
        )
        ax.set_xlim(-0.03, 2.03)
        ax.set_ylim(f1_min - 0.45, f1_max + 0.35)
        ax.set_xticks(np.arange(0, 2.01, 0.2))
        ax.tick_params(axis="x", labelrotation=45, labelsize=7.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Marginal mean F1 (%)")
        variable_label = "A" if dimension == "A" else "B"
        ax.set_title(f"{panel} Effect of {variable_label} on F1", fontweight="bold")
        ax.grid(color="#dddddd", linewidth=0.6, alpha=0.75, zorder=0)
    axes[0, 0].legend(frameon=False, loc="lower right")

    # (c) 四个同量纲精度指标，显示 121 组配对差值的平均和全范围。
    accuracy_metrics = [
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("f1", "F1"),
        ("buffer_overlap_iou", "Path IoU"),
    ]
    means = []
    lower = []
    upper = []
    labels = []
    for metric, label in accuracy_metrics:
        pivot = data.pivot(index=["A", "B"], columns="algorithm", values=metric)
        difference = (pivot["MFD"] - pivot["D8"]) * 100.0
        mean = float(difference.mean())
        means.append(mean)
        lower.append(mean - float(difference.min()))
        upper.append(float(difference.max()) - mean)
        labels.append(label)

    ax = axes[1, 0]
    y = np.arange(len(labels))
    point_colors = [colors["MFD"] if value >= 0 else colors["D8"] for value in means]
    for i, (mean, lo, hi, color) in enumerate(zip(means, lower, upper, point_colors)):
        ax.errorbar(
            mean,
            i,
            xerr=np.array([[lo], [hi]]),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=5.5,
            capsize=3,
            linewidth=1.5,
        )
        ax.text(
            mean + (0.035 if mean >= 0 else -0.035),
            i,
            f"{mean:+.2f}",
            ha="left" if mean >= 0 else "right",
            va="center",
            fontsize=8,
            color=color,
            fontweight="bold",
        )
    ax.axvline(0.0, color="#777777", linewidth=0.9, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired difference: MFD − D8 (percentage points)")
    ax.set_title("(c) Paired metric differences across 121 pairs", fontweight="bold")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.75)
    ax.text(
        0.02,
        0.03,
        "Whiskers show the full range across A/B combinations",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )

    # (d) 异量纲结构指标改用相对降低率，同时标注原始单位。
    structural_metrics = [
        ("symmetric_mean_offset_m", "Symmetric offset", "0.855 m"),
        ("estimated_breakpoint_count", "Continuity-gap proxy", "34.4 fewer breakpoints"),
    ]
    reductions = []
    structural_labels = []
    raw_labels = []
    for metric, label, raw_label in structural_metrics:
        pivot = data.pivot(index=["A", "B"], columns="algorithm", values=metric)
        reduction = (pivot["D8"] - pivot["MFD"]) / pivot["D8"] * 100.0
        reductions.append(float(reduction.mean()))
        structural_labels.append(label)
        raw_labels.append(raw_label)

    ax = axes[1, 1]
    y = np.arange(len(structural_labels))
    bars = ax.barh(y, reductions, color="#d6604d", height=0.46)
    for bar, value, raw_label in zip(bars, reductions, raw_labels):
        ax.text(
            value + 0.08,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.2f}%  ({raw_label})",
            ha="left",
            va="center",
            fontsize=8.2,
            fontweight="bold",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(structural_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(reductions) * 1.48)
    ax.set_xlabel("Mean relative reduction by MFD (%)")
    ax.set_title("(d) Spatial and continuity improvements", fontweight="bold")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.75)

    fig.suptitle(
        "Algorithm comparison across burning-depth combinations",
        y=0.995,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0, w_pad=1.5)
    save_figure(fig, "MainFig2_Algorithm_Comparison_Summary")


def main():
    data = load_data()
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_recall_figures(data)
    plot_f1_iou(data)
    plot_difference_heatmaps(data)
    plot_marginal_response(data)
    plot_pair_difference_distributions(data)
    plot_main_comparison_summary(data)
    print(f"已输出至: {OUT_DIR}")


if __name__ == "__main__":
    main()
