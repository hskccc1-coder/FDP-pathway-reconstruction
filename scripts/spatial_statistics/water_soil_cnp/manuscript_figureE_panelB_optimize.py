from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
FIG_E_DIR = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "manuscript_backhalf_figures_final" / "FigureE_Ntransport"
IMAGE_ONLY = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "manuscript_backhalf_figures_images_only"

BANDS = ["0-1 m", "1-3 m", "3-5 m", "5-10 m"]
DECLINE_LABELS = {
    "0-1 m": "Reference",
    "1-3 m": "-37%",
    "3-5 m": "-49%",
    "5-10 m": "-62%",
}
LETTERS = {"0-1 m": "a", "1-3 m": "b", "3-5 m": "c", "5-10 m": "d"}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = pd.read_csv(FIG_E_DIR / "FigureE_segment_level_stats.csv")
    friedman = pd.read_csv(FIG_E_DIR / "FigureE_Friedman_result.csv")
    stats = stats[(stats["Distance_band"].isin(BANDS)) & (stats["Valid_pixel_N"] > 0)].copy()
    stats["Plot_value"] = np.log10(stats["Median_N_transport"] + 1.0)
    return stats, friedman


def band_arrays(stats: pd.DataFrame) -> list[np.ndarray]:
    return [stats.loc[stats["Distance_band"] == band, "Plot_value"].dropna().to_numpy() for band in BANDS]


def summaries(stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for band in BANDS:
        values = stats.loc[stats["Distance_band"] == band, "Plot_value"].dropna()
        rows.append(
            {
                "Distance_band": band,
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "n_segments": int(values.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def add_background_points(ax, arrays: list[np.ndarray]) -> None:
    rng = np.random.default_rng(20260827)
    for idx, values in enumerate(arrays, start=1):
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=2.2,
            color="#71b88a",
            alpha=0.08,
            linewidths=0,
            zorder=1,
        )


def add_median_iqr(ax, summary: pd.DataFrame, connect: bool) -> None:
    x = np.arange(1, len(BANDS) + 1)
    med = summary["median"].to_numpy()
    q25 = summary["q25"].to_numpy()
    q75 = summary["q75"].to_numpy()
    if connect:
        ax.plot(x, med, color="#006d3c", linewidth=1.15, zorder=4)
    ax.errorbar(
        x,
        med,
        yerr=np.vstack([med - q25, q75 - med]),
        fmt="o",
        color="#006d3c",
        ecolor="#006d3c",
        elinewidth=1.15,
        capsize=3.5,
        capthick=1.15,
        markersize=4.2,
        zorder=5,
    )


def add_annotations(ax, summary: pd.DataFrame) -> None:
    y_max = max(float(summary["q75"].max()) + 0.18, 0.68)
    ax.set_ylim(-0.03, y_max)
    for idx, band in enumerate(BANDS, start=1):
        row = summary.loc[summary["Distance_band"] == band].iloc[0]
        ax.text(idx, y_max - 0.09, DECLINE_LABELS[band], ha="center", va="top", fontsize=7, color="black")
        ax.text(
            idx,
            float(row["q75"]) + 0.022,
            LETTERS[band],
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="black",
        )
    ax.text(
        0.02,
        0.98,
        "Friedman, p < 0.001",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        color="black",
    )


def style_axis(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold", fontsize=8.5)
    ax.set_xticks(np.arange(1, len(BANDS) + 1))
    ax.set_xticklabels(BANDS)
    ax.set_xlabel("Distance from reconstructed runoff pathway (m)")
    ax.set_ylabel("log10(N transport intensity + 1)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", colors="black", width=0.8, length=3.0, direction="out")
    ax.grid(False)


def plot_boxplot_version(stats: pd.DataFrame) -> None:
    arrays = band_arrays(stats)
    summary = summaries(stats)
    fig, ax = plt.subplots(figsize=(4.25, 3.05), constrained_layout=True)
    bp = ax.boxplot(
        arrays,
        positions=np.arange(1, len(BANDS) + 1),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.9},
        boxprops={"facecolor": "#dbeee2", "edgecolor": "black", "linewidth": 0.75},
        whiskerprops={"color": "black", "linewidth": 0.75},
        capprops={"color": "black", "linewidth": 0.75},
    )
    for patch in bp["boxes"]:
        patch.set_alpha(0.76)
    add_background_points(ax, arrays)
    add_median_iqr(ax, summary, connect=False)
    add_annotations(ax, summary)
    style_axis(ax, "(b) N transport intensity across pathway-distance bands")
    fig.savefig(FIG_E_DIR / "FigureE_boxplot_version.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_E_DIR / "FigureE_boxplot_version.pdf", bbox_inches="tight")
    fig.savefig(FIG_E_DIR / "FigureE_boxplot_version.svg", bbox_inches="tight")
    plt.close(fig)


def plot_median_iqr_version(stats: pd.DataFrame) -> None:
    arrays = band_arrays(stats)
    summary = summaries(stats)
    fig, ax = plt.subplots(figsize=(4.25, 3.05), constrained_layout=True)
    add_background_points(ax, arrays)
    add_median_iqr(ax, summary, connect=True)
    add_annotations(ax, summary)
    style_axis(ax, "(b) N transport intensity across pathway-distance bands")
    fig.savefig(FIG_E_DIR / "FigureE_medianIQR_version.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_E_DIR / "FigureE_medianIQR_version.pdf", bbox_inches="tight")
    fig.savefig(FIG_E_DIR / "FigureE_medianIQR_version.svg", bbox_inches="tight")
    plt.close(fig)


def update_caption(friedman: pd.DataFrame) -> None:
    chi_square = float(friedman.loc[0, "chi_square"])
    n_segments = int(friedman.loc[0, "N_segments"])
    k = 4
    kendall_w = chi_square / (n_segments * (k - 1))
    text = f"""# Figure E Panel B Caption Note

Panel (b) summarizes spatial/process consistency between modeled N transport intensity and reconstructed runoff corridors. The y-axis is log10(N transport intensity + 1). Values are segment-level medians from 50 m pathway segments, and raster cells were used only to summarize each segment-band combination.

Distance bands are mutually exclusive: 0-1 m, 1-3 m, 3-5 m, and 5-10 m from the reconstructed runoff pathway. The plotted green points show the median across pathway segments, error bars show the interquartile range, and faint background points show segment-level values with low visual weight. Percentage annotations above the groups are descriptive declines relative to the 0-1 m median and are not additional statistical tests.

Friedman test: chi-square = {chi_square:.4f}, df = 3, p < 0.001, n = {n_segments} complete pathway segments. Kendall's W = {kendall_w:.3f}. All pairwise Wilcoxon signed-rank comparisons were significant after Holm correction (adjusted p < 0.001).

This panel should be interpreted as evidence of spatial/process consistency between modeled N transport intensity and reconstructed runoff corridors, not as validation of an observed or actual N transport pathway.
"""
    (FIG_E_DIR / "FigureE_panelB_caption.md").write_text(text, encoding="utf-8-sig")
    print(f"Kendall_W={kendall_w:.6f}")


def copy_images() -> None:
    IMAGE_ONLY.mkdir(parents=True, exist_ok=True)
    for name in [
        "FigureE_boxplot_version.png",
        "FigureE_boxplot_version.pdf",
        "FigureE_boxplot_version.svg",
        "FigureE_medianIQR_version.png",
        "FigureE_medianIQR_version.pdf",
        "FigureE_medianIQR_version.svg",
    ]:
        src = FIG_E_DIR / name
        if src.exists():
            (IMAGE_ONLY / name).write_bytes(src.read_bytes())


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    stats, friedman = load_data()
    plot_boxplot_version(stats)
    plot_median_iqr_version(stats)
    update_caption(friedman)
    copy_images()
    print(f"Wrote optimized panel B figures to: {FIG_E_DIR}")


if __name__ == "__main__":
    main()
