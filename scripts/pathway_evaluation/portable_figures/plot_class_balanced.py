# -*- coding: utf-8 -*-
"""Plot the final 0.2 m class-balanced D8/MFD comparison heatmaps."""

from __future__ import print_function

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEPTH_STEP_M = 0.2
STEP_TAG = ("%.1f" % DEPTH_STEP_M).replace(".", "p")
RUN_KIND = u"粗步长" if DEPTH_STEP_M >= 0.4 - 1e-9 else u"全量步长"
DATA_DIR = os.path.join(
    PACKAGE_ROOT,
    "results",
    "pathway_metrics",
    "dem_routing",
    u"D8_MFD_分类宏平均_%s%s_thr1200" % (RUN_KIND, STEP_TAG),
)
OUT_DIR = os.path.join(
    PACKAGE_ROOT,
    "results",
    "pathway_metrics",
    "figures",
    u"D8_MFD_分类宏平均_%s%s_thr1200" % (RUN_KIND, STEP_TAG),
)
SUMMARY_CSV = os.path.join(DATA_DIR, "D8_MFD_macro_summary_step%s.csv" % STEP_TAG)
PAIR_CSV = os.path.join(DATA_DIR, "MFD_minus_D8_class_macro_step%s.csv" % STEP_TAG)
FIGURE_TAG = "Coarse" if DEPTH_STEP_M >= 0.4 - 1e-9 else "Full"
OUT_OVERVIEW = os.path.join(OUT_DIR, "Fig_ClassBalanced_Recall_%s_Overview" % FIGURE_TAG)
OUT_DIFF = os.path.join(OUT_DIR, "Fig_ClassBalanced_Recall_%s_MFD_minus_D8" % FIGURE_TAG)


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def pivot(data, value):
    return data.pivot(index="B", columns="A", values=value).sort_index().sort_index(axis=1)


def label_heatmap(ax, values, percent=True, threshold=None):
    if threshold is None:
        threshold = 0.5 * (np.nanmin(values) + np.nanmax(values))
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            text = "%.1f" % (value * 100.0) if percent else "%+.2f" % value
            color = "white" if value > threshold else "black"
            font_size = 5.2 if values.shape[1] > 7 else 6.6
            ax.text(col, row, text, ha="center", va="center", fontsize=font_size, color=color)


def format_axes(ax, table, title, show_ylabel, show_xlabel):
    ax.set_title(title, pad=4, fontweight="bold")
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels(["%g" % value for value in table.columns])
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels(["%g" % value for value in table.index])
    ax.set_xlabel(r"$A$ field-ditch burning depth (m)" if show_xlabel else "")
    ax.set_ylabel(r"$B$ channel burning depth (m)" if show_ylabel else "")
    if len(table.columns) > 7:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


def save_all(fig, base):
    fig.savefig(base + ".png", dpi=400, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")


def plot_overview(summary):
    specs = [
        ("field_recall", "Field-ditch recall"),
        ("channel_recall", "Channel recall"),
        ("macro_recall", "Macro recall"),
    ]
    values_all = summary[[item[0] for item in specs]].to_numpy(dtype=float)
    vmin = float(np.nanmin(values_all))
    vmax = float(np.nanmax(values_all))
    width = 10.8 if summary["A"].nunique() > 7 else 8.8
    height = 6.8 if summary["A"].nunique() > 7 else 5.65
    fig, axes = plt.subplots(2, 3, figsize=(width, height), dpi=240, constrained_layout=True)
    image = None
    for row_index, algorithm in enumerate(["D8", "MFD"]):
        subset = summary[summary["algorithm"] == algorithm]
        for col_index, (metric, label) in enumerate(specs):
            ax = axes[row_index, col_index]
            table = pivot(subset, metric)
            image = ax.imshow(table.values, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
            label_heatmap(ax, table.values, threshold=0.5 * (vmin + vmax))
            format_axes(
                ax,
                table,
                "(%s) %s: %s" % (chr(97 + row_index * 3 + col_index), algorithm, label),
                show_ylabel=(col_index == 0),
                show_xlabel=(row_index == 1),
            )
    cbar = fig.colorbar(image, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label("Recall (%)")
    cbar.set_ticks(np.linspace(vmin, vmax, 5))
    cbar.set_ticklabels(["%.1f" % (value * 100.0) for value in np.linspace(vmin, vmax, 5)])
    save_all(fig, OUT_OVERVIEW)
    plt.close(fig)


def plot_difference(pair):
    specs = [
        ("MFD_minus_D8_field_recall", "Field-ditch recall"),
        ("MFD_minus_D8_channel_recall", "Channel recall"),
        ("MFD_minus_D8_macro_recall", "Macro recall"),
    ]
    abs_max = max(float(np.nanmax(np.abs(pair[metric].to_numpy(dtype=float)))) for metric, _ in specs)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    width = 10.8 if pair["A"].nunique() > 7 else 8.8
    height = 4.2 if pair["A"].nunique() > 7 else 3.15
    fig, axes = plt.subplots(1, 3, figsize=(width, height), dpi=240, constrained_layout=True)
    image = None
    for index, (metric, label) in enumerate(specs):
        ax = axes[index]
        table = pivot(pair, metric)
        image = ax.imshow(table.values * 100.0, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-abs_max * 100.0, vcenter=0.0, vmax=abs_max * 100.0), aspect="equal")
        for row in range(table.values.shape[0]):
            for col in range(table.values.shape[1]):
                value_pp = table.values[row, col] * 100.0
                color = "white" if abs(value_pp) > 0.55 * abs_max * 100.0 else "black"
                font_size = 5.0 if table.values.shape[1] > 7 else 6.5
                ax.text(col, row, "%+.2f" % value_pp, ha="center", va="center", fontsize=font_size, color=color)
        format_axes(ax, table, "(%s) %s" % (chr(97 + index), label), index == 0, True)
    cbar = fig.colorbar(image, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label("MFD − D8 (percentage points)")
    save_all(fig, OUT_DIFF)
    plt.close(fig)


def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    setup_style()
    summary = pd.read_csv(SUMMARY_CSV)
    pair = pd.read_csv(PAIR_CSV)
    plot_overview(summary)
    plot_difference(pair)
    print(OUT_OVERVIEW + ".png")
    print(OUT_DIFF + ".png")


if __name__ == "__main__":
    main()
