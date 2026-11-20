# -*- coding: utf-8 -*-
"""Draw publication figures for FlowAcc-threshold sensitivity."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PACKAGE_ROOT / "results" / "pathway_metrics" / "dem_routing" / "汇流累积阈值敏感性_A0p8_B1p2_C0"
OUT_DIR = PACKAGE_ROOT / "results" / "pathway_metrics" / "figures" / "汇流累积阈值敏感性_A0p8_B1p2_C0"
CSV_PATH = DATA_DIR / "D8_MFD_threshold_sensitivity_metrics.csv"

COLORS = {"D8": "#2878B5", "MFD": "#D95F4C"}
MARKERS = {"D8": "o", "MFD": "s"}
CURRENT_THRESHOLD = 1200
F1_OPTIMUM = 50000
TICKS = [400, 1200, 5000, 15000, 50000, 120000]
TICK_LABELS = ["400", "1,200", "5,000", "15,000", "50,000", "120,000"]


def set_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def setup_threshold_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(360, 135000)
    ax.set_xticks(TICKS)
    ax.set_xticklabels(TICK_LABELS)
    ax.minorticks_off()
    ax.grid(True, axis="both", color="#D8D8D8", linewidth=0.65, alpha=0.72)
    ax.axvline(
        CURRENT_THRESHOLD,
        color="#555555",
        linestyle="--",
        linewidth=1.05,
        zorder=1,
    )
    ax.set_xlabel("Flow-accumulation threshold (cells)")


def plot_lines(ax, data, metric, scale=1.0):
    for algorithm in ("D8", "MFD"):
        part = data[data["algorithm"] == algorithm].sort_values(
            "flowacc_threshold_cells"
        )
        ax.plot(
            part["flowacc_threshold_cells"],
            part[metric] * scale,
            color=COLORS[algorithm],
            marker=MARKERS[algorithm],
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.15,
            linewidth=1.7,
            label=algorithm,
            zorder=3,
        )


def save_all(fig, stem):
    for suffix in ("png", "pdf"):
        fig.savefig(OUT_DIR / (stem + "." + suffix), format=suffix)


def draw_accuracy(data):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.1), constrained_layout=True)
    specifications = [
        ("macro_recall", "Macro recall (%)", "(a) Macro-averaged recall"),
        ("precision", "Precision (%)", "(b) Precision"),
        ("f1", "F1 score (%)", "(c) F1 score"),
        ("buffer_overlap_iou", "Path IoU (%)", "(d) Path overlap"),
    ]
    for ax, (metric, ylabel, title) in zip(axes.flat, specifications):
        plot_lines(ax, data, metric, scale=100.0)
        setup_threshold_axis(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=9, fontweight="bold")

    # Explain the two reference choices without obscuring the curves.
    axes[0, 0].text(
        CURRENT_THRESHOLD,
        axes[0, 0].get_ylim()[1],
        "  current: 1,200",
        ha="left",
        va="top",
        color="#444444",
        fontsize=9,
    )
    for ax, metric in ((axes[1, 0], "f1"), (axes[1, 1], "buffer_overlap_iou")):
        for algorithm in ("D8", "MFD"):
            row = data[
                (data["algorithm"] == algorithm)
                & (data["flowacc_threshold_cells"] == F1_OPTIMUM)
            ].iloc[0]
            ax.scatter(
                [F1_OPTIMUM],
                [row[metric] * 100.0],
                s=75,
                marker="*",
                color=COLORS[algorithm],
                edgecolor="white",
                linewidth=0.7,
                zorder=5,
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.025),
    )
    save_all(fig, "Fig1_FlowAcc_Threshold_Sensitivity_D8_MFD")
    plt.close(fig)


def draw_structure(data):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.65), constrained_layout=True)

    plot_lines(axes[0], data, "symmetric_mean_offset_m")
    setup_threshold_axis(axes[0])
    axes[0].set_ylabel("Symmetric mean offset (m)")
    axes[0].set_title("(a) Mean bidirectional path offset", pad=9, fontweight="bold")

    plot_lines(axes[1], data, "estimated_breakpoint_count")
    setup_threshold_axis(axes[1])
    axes[1].set_ylabel("Estimated internal breakpoints")
    axes[1].set_title("(b) Path-continuity interruptions", pad=9, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    save_all(fig, "Fig2_FlowAcc_Threshold_Offset_Breakpoints_D8_MFD")
    plt.close(fig)


def main():
    set_style()
    data = pd.read_csv(CSV_PATH)
    draw_accuracy(data)
    draw_structure(data)
    print(OUT_DIR / "Fig1_FlowAcc_Threshold_Sensitivity_D8_MFD.png")
    print(OUT_DIR / "Fig2_FlowAcc_Threshold_Offset_Breakpoints_D8_MFD.png")


if __name__ == "__main__":
    main()
