from __future__ import annotations

import shutil
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy import ndimage, stats


ROOT = Path(__file__).resolve().parents[3]
NLOAD_RASTER = ROOT / "data" / "soil_water_measurements" / "单位侵蚀量" / "水土总氮多流向总氮1.tif"
FLOWACC_RASTER = Path(
    os.environ.get(
        "ECOINFO_FLOWACC_RASTER",
        str(ROOT / "data" / "soil_water_measurements" / "external" / "FlowAcc_MFD_A1p00_B1p20.tif"),
    )
)
OUT_DIR = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "manuscript_backhalf_figures_final" / "FigureE_Ntransport_FlowAccPathway"
IMAGE_ONLY = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "manuscript_backhalf_figures_images_only"

BANDS = [
    ("0-1 m", 0.0, 1.0),
    ("1-3 m", 1.0, 3.0),
    ("3-5 m", 3.0, 5.0),
    ("5-10 m", 5.0, 10.0),
]
BLOCK_SIZE_M = 50
FLOWACC_THRESHOLD_PERCENTILE = 99.5


def read_n_transport() -> tuple[np.ndarray, tuple[float, ...], str]:
    ds = gdal.Open(str(NLOAD_RASTER))
    if ds is None:
        raise FileNotFoundError(NLOAD_RASTER)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(float)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        arr[np.isclose(arr, nodata)] = np.nan
    arr[arr < 0] = np.nan
    return arr, ds.GetGeoTransform(), ds.GetProjection()


def raster_extent(gt: tuple[float, ...], nx: int, ny: int) -> tuple[float, float, float, float]:
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + nx * gt[1]
    ymin = ymax + ny * gt[5]
    return xmin, xmax, ymin, ymax


def read_flowacc_to_n_grid(gt: tuple[float, ...], projection: str, nx: int, ny: int) -> np.ndarray:
    source = gdal.Open(str(FLOWACC_RASTER))
    if source is None:
        raise FileNotFoundError(FLOWACC_RASTER)
    source_nodata = source.GetRasterBand(1).GetNoDataValue()
    bounds = (gt[0], gt[3] + ny * gt[5], gt[0] + nx * gt[1], gt[3])
    warped = gdal.Warp(
        "",
        source,
        format="MEM",
        outputBounds=bounds,
        width=nx,
        height=ny,
        dstSRS=projection,
        srcNodata=source_nodata,
        dstNodata=np.nan,
        resampleAlg="max",
    )
    arr = warped.GetRasterBand(1).ReadAsArray().astype(float)
    arr[arr < -1e30] = np.nan
    return arr


def build_corridor_segments(flowacc_1m: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, pd.DataFrame]:
    valid = np.isfinite(flowacc_1m)
    threshold = float(np.nanpercentile(flowacc_1m[valid], FLOWACC_THRESHOLD_PERCENTILE))
    corridor_mask = valid & (flowacc_1m >= threshold)

    h, w = corridor_mask.shape
    y_blocks = np.arange(h) // BLOCK_SIZE_M
    x_blocks = np.arange(w) // BLOCK_SIZE_M
    segment_raster = np.zeros((h, w), dtype=np.int32)
    segment_lookup: dict[tuple[int, int], int] = {}
    rows = []
    next_id = 1
    ys, xs = np.where(corridor_mask)
    for y, x in zip(ys, xs):
        key = (int(y_blocks[y]), int(x_blocks[x]))
        if key not in segment_lookup:
            segment_lookup[key] = next_id
            rows.append({"Segment_ID": next_id, "Block_row": key[0], "Block_col": key[1]})
            next_id += 1
        segment_raster[y, x] = segment_lookup[key]

    inventory = pd.DataFrame(rows)
    corridor_counts = pd.Series(segment_raster[segment_raster > 0]).value_counts().rename_axis("Segment_ID")
    corridor_counts = corridor_counts.reset_index(name="Corridor_pixel_N")
    inventory = inventory.merge(corridor_counts, on="Segment_ID", how="left")
    inventory["Segment_length_m"] = inventory["Corridor_pixel_N"].astype(float)
    return corridor_mask, segment_raster, threshold, inventory


def assign_band(distance_m: np.ndarray) -> np.ndarray:
    band = np.full(distance_m.shape, "", dtype=object)
    for label, low, high in BANDS:
        if low == 0:
            mask = (distance_m >= low) & (distance_m <= high)
        else:
            mask = (distance_m > low) & (distance_m <= high)
        band[mask] = label
    return band


def segment_level_stats(
    n_transport: np.ndarray,
    distance_m: np.ndarray,
    nearest_segment: np.ndarray,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    band_labels = assign_band(distance_m)
    valid = np.isfinite(n_transport) & (nearest_segment > 0) & (distance_m <= 10.0) & (band_labels != "")
    pixels = pd.DataFrame(
        {
            "Segment_ID": nearest_segment[valid].astype(int),
            "Distance_band": band_labels[valid],
            "N_transport": n_transport[valid].astype(float),
        }
    )
    pixels["N_transport_log10p1"] = np.log10(pixels["N_transport"] + 1.0)
    grouped = pixels.groupby(["Segment_ID", "Distance_band"], sort=False)
    raw = grouped["N_transport"].agg(
        Valid_pixel_N="count",
        Mean_N_transport="mean",
        Median_N_transport="median",
        Q25=lambda x: x.quantile(0.25),
        Q75=lambda x: x.quantile(0.75),
        SD=lambda x: x.std(ddof=1),
    )
    log_stats = grouped["N_transport_log10p1"].agg(Mean_log="mean", Median_log="median")
    out = raw.join(log_stats).reset_index()
    out["Mean_raw"] = out["Mean_N_transport"]
    out["Median_raw"] = out["Median_N_transport"]

    all_rows = pd.MultiIndex.from_product(
        [inventory["Segment_ID"].astype(int), [band[0] for band in BANDS]],
        names=["Segment_ID", "Distance_band"],
    ).to_frame(index=False)
    out = all_rows.merge(out, on=["Segment_ID", "Distance_band"], how="left")
    out = out.merge(inventory[["Segment_ID", "Segment_length_m", "Corridor_pixel_N"]], on="Segment_ID", how="left")
    out["Valid_pixel_N"] = out["Valid_pixel_N"].fillna(0).astype(int)
    return out[
        [
            "Segment_ID",
            "Segment_length_m",
            "Corridor_pixel_N",
            "Distance_band",
            "Valid_pixel_N",
            "Mean_N_transport",
            "Median_N_transport",
            "Q25",
            "Q75",
            "SD",
            "Mean_raw",
            "Median_raw",
            "Mean_log",
            "Median_log",
        ]
    ]


def pairing_qa(stats_table: pd.DataFrame) -> pd.DataFrame:
    counts = stats_table.pivot(index="Segment_ID", columns="Distance_band", values="Valid_pixel_N")
    out = pd.DataFrame({"Segment_ID": counts.index.astype(int)})
    for label, _, _ in BANDS:
        out[f"Has_{label}"] = counts[label].fillna(0).to_numpy() > 0
    out["Complete_for_Friedman"] = out[[f"Has_{band[0]}" for band in BANDS]].all(axis=1)
    return out


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        value = min((m - rank) * p_values[idx], 1.0)
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def run_tests(stats_table: pd.DataFrame, qa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complete = qa.loc[qa["Complete_for_Friedman"], "Segment_ID"].astype(int)
    pivot = (
        stats_table[stats_table["Segment_ID"].isin(complete)]
        .pivot(index="Segment_ID", columns="Distance_band", values="Median_N_transport")
        [[band[0] for band in BANDS]]
        .dropna()
    )
    chi_square, p_value = stats.friedmanchisquare(*[pivot[label].to_numpy() for label, _, _ in BANDS])
    friedman = pd.DataFrame(
        [{"N_segments": int(len(pivot)), "chi_square": float(chi_square), "df": 3, "p": float(p_value)}]
    )
    rows = []
    for i, band1 in enumerate([band[0] for band in BANDS]):
        for band2 in [band[0] for band in BANDS][i + 1 :]:
            pair = pivot[[band1, band2]].dropna()
            try:
                w_stat, raw_p = stats.wilcoxon(pair[band1], pair[band2], zero_method="wilcox")
            except ValueError:
                w_stat, raw_p = 0.0, 1.0
            rows.append({"Band1": band1, "Band2": band2, "N_pairs": int(len(pair)), "W": float(w_stat), "raw_p": float(raw_p)})
    pairwise = pd.DataFrame(rows)
    pairwise["Holm_adjusted_p"] = holm_adjust(pairwise["raw_p"].fillna(1.0).tolist())
    return friedman, pairwise, pivot


def summarize(stats_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, _, _ in BANDS:
        raw_values = stats_table.loc[
            (stats_table["Distance_band"] == label) & (stats_table["Valid_pixel_N"] > 0), "Median_N_transport"
        ].dropna()
        plot_values = np.log10(raw_values + 1.0)
        rows.append(
            {
                "Distance_band": label,
                "N_segments": int(raw_values.shape[0]),
                "Segment_median": float(raw_values.median()),
                "Segment_Q25": float(raw_values.quantile(0.25)),
                "Segment_Q75": float(raw_values.quantile(0.75)),
                "Plot_median": float(plot_values.median()),
                "Plot_Q25": float(plot_values.quantile(0.25)),
                "Plot_Q75": float(plot_values.quantile(0.75)),
            }
        )
    summary = pd.DataFrame(rows)
    reference = float(summary.loc[summary["Distance_band"] == "0-1 m", "Segment_median"].iloc[0])
    summary["Decline_vs_0_1_pct"] = (1.0 - summary["Segment_median"] / reference) * 100.0
    return summary


def add_background_points(ax, stats_table: pd.DataFrame) -> None:
    rng = np.random.default_rng(20260827)
    for idx, label in enumerate([band[0] for band in BANDS], start=1):
        values = stats_table.loc[
            (stats_table["Distance_band"] == label) & (stats_table["Valid_pixel_N"] > 0),
            "Median_N_transport",
        ].dropna()
        values = np.log10(values + 1.0)
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=3.5,
            color="#405463",
            alpha=0.07,
            linewidths=0,
            zorder=1,
        )


def add_panel_annotations(ax, summary: pd.DataFrame, friedman: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    y_top = max(float(summary["Plot_Q75"].max()) + 0.18, 0.56)
    ax.set_ylim(-0.03, y_top)
    letters = compact_letters(pairwise)
    for idx, label in enumerate([band[0] for band in BANDS], start=1):
        row = summary.loc[summary["Distance_band"] == label].iloc[0]
        if label == "0-1 m":
            decline = "Reference"
        else:
            decline = f"-{row['Decline_vs_0_1_pct']:.0f}%"
        ax.text(idx, y_top - 0.075, decline, ha="center", va="top", fontsize=8, color="#283844")
        ax.text(idx, float(row["Plot_Q75"]) + 0.022, letters[label], ha="center", va="bottom", fontsize=8.5, color="#1d2b34")
    ax.text(0.02, 0.98, "Friedman, p < 0.001", transform=ax.transAxes, ha="left", va="top", fontsize=8.8, color="#1d2b34")


def compact_letters(pairwise: pd.DataFrame, alpha: float = 0.05) -> dict[str, str]:
    labels = [band[0] for band in BANDS]
    significant = {(row.Band1, row.Band2): row.Holm_adjusted_p < alpha for row in pairwise.itertuples(index=False)}
    significant.update({(b, a): value for (a, b), value in list(significant.items())})

    candidates: list[set[str]] = []
    for mask in range(1, 2 ** len(labels)):
        group = {labels[i] for i in range(len(labels)) if mask & (1 << i)}
        if all(not significant.get((a, b), False) for a in group for b in group if a != b):
            candidates.append(group)

    required_pairs = [
        (a, b)
        for i, a in enumerate(labels)
        for b in labels[i + 1 :]
        if not significant.get((a, b), False)
    ]

    best: tuple[set[str], ...] | None = None
    from itertools import combinations

    for size in range(1, len(labels) + 2):
        for combo in combinations(candidates, size):
            covered_labels = set().union(*combo)
            if covered_labels != set(labels):
                continue
            if any(not any(a in group and b in group for group in combo) for a, b in required_pairs):
                continue
            best = combo
            break
        if best is not None:
            break

    letter_sets = sorted(best or tuple({label} for label in labels), key=lambda group: min(labels.index(label) for label in group))
    out = {label: "" for label in labels}
    for idx, group in enumerate(letter_sets):
        letter = chr(ord("a") + idx)
        for label in group:
            out[label] += letter
    return out


def style_panel_b(ax) -> None:
    ax.set_title("(b) N transport intensity across pathway-distance bands", loc="left", fontweight="bold", fontsize=10)
    ax.set_xticks(np.arange(1, len(BANDS) + 1))
    ax.set_xticklabels([band[0] for band in BANDS])
    ax.set_xlabel("Distance from reconstructed runoff pathway (m)")
    ax.set_ylabel("log10(N transport intensity + 1)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.33)


def make_boxplot(stats_table: pd.DataFrame, summary: pd.DataFrame, friedman: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    arrays = [
        np.log10(
            stats_table.loc[
                (stats_table["Distance_band"] == label) & (stats_table["Valid_pixel_N"] > 0), "Median_N_transport"
            ].dropna()
            + 1.0
        ).to_numpy()
        for label, _, _ in BANDS
    ]
    fig, ax = plt.subplots(figsize=(4.4, 3.25), constrained_layout=True)
    bp = ax.boxplot(
        arrays,
        positions=np.arange(1, len(BANDS) + 1),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1d2b34", "linewidth": 1.1},
        boxprops={"facecolor": "#d8e5df", "edgecolor": "#577169", "linewidth": 0.85},
        whiskerprops={"color": "#577169", "linewidth": 0.8},
        capprops={"color": "#577169", "linewidth": 0.8},
    )
    for patch in bp["boxes"]:
        patch.set_alpha(0.82)
    add_background_points(ax, stats_table)
    x = np.arange(1, len(BANDS) + 1)
    med = summary["Plot_median"].to_numpy()
    q25 = summary["Plot_Q25"].to_numpy()
    q75 = summary["Plot_Q75"].to_numpy()
    ax.errorbar(x, med, yerr=np.vstack([med - q25, q75 - med]), fmt="o", color="#176f55", ecolor="#176f55", elinewidth=1.5, capsize=3.5, markersize=4.6, zorder=5)
    add_panel_annotations(ax, summary, friedman, pairwise)
    style_panel_b(ax)
    fig.savefig(OUT_DIR / "FigureE_FlowAcc_boxplot_version.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "FigureE_FlowAcc_boxplot_version.pdf", bbox_inches="tight")
    plt.close(fig)


def make_median_iqr(stats_table: pd.DataFrame, summary: pd.DataFrame, friedman: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.25), constrained_layout=True)
    add_background_points(ax, stats_table)
    x = np.arange(1, len(BANDS) + 1)
    med = summary["Plot_median"].to_numpy()
    q25 = summary["Plot_Q25"].to_numpy()
    q75 = summary["Plot_Q75"].to_numpy()
    ax.plot(x, med, color="#176f55", linewidth=1.45, zorder=4)
    ax.errorbar(x, med, yerr=np.vstack([med - q25, q75 - med]), fmt="o", color="#176f55", ecolor="#176f55", elinewidth=1.5, capsize=3.5, markersize=4.6, zorder=5)
    add_panel_annotations(ax, summary, friedman, pairwise)
    style_panel_b(ax)
    fig.savefig(OUT_DIR / "FigureE_FlowAcc_medianIQR_version.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "FigureE_FlowAcc_medianIQR_version.pdf", bbox_inches="tight")
    plt.close(fig)


def make_final_figure(
    n_transport: np.ndarray,
    flowacc_1m: np.ndarray,
    corridor_mask: np.ndarray,
    gt: tuple[float, ...],
    stats_table: pd.DataFrame,
    summary: pd.DataFrame,
    friedman: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    extent = tuple(value * 0.001 for value in raster_extent(gt, n_transport.shape[1], n_transport.shape[0]))
    display = np.log10(n_transport + 1.0)
    display[~np.isfinite(n_transport)] = np.nan
    vmin, vmax = np.nanpercentile(display, [2, 98])
    halo = ndimage.binary_dilation(corridor_mask, iterations=1)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.35, 3.55),
        gridspec_kw={"width_ratios": [1.0, 1.08]},
        constrained_layout=True,
    )
    im = axes[0].imshow(display, extent=extent, origin="upper", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].imshow(
        np.ma.masked_where(~halo, halo),
        extent=extent,
        origin="upper",
        cmap=ListedColormap(["white"]),
        alpha=0.72,
        interpolation="nearest",
    )
    axes[0].imshow(
        np.ma.masked_where(~corridor_mask, corridor_mask),
        extent=extent,
        origin="upper",
        cmap=ListedColormap(["#d7301f"]),
        alpha=0.9,
        interpolation="nearest",
    )
    axes[0].set_title("(a) Modeled N transport intensity", loc="left", fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Easting (km)")
    axes[0].set_ylabel("Northing (km)")
    axes[0].text(0.02, 0.02, "MFD FlowAcc-derived runoff corridor", transform=axes[0].transAxes, ha="left", va="bottom", fontsize=7.4, color="white", bbox={"facecolor": "#263238", "edgecolor": "none", "alpha": 0.72, "pad": 2.2})
    cb = fig.colorbar(im, ax=axes[0], orientation="horizontal", shrink=0.72, pad=0.08)
    cb.set_label("log10(N transport intensity + 1)")

    arrays = [
        np.log10(
            stats_table.loc[
                (stats_table["Distance_band"] == label) & (stats_table["Valid_pixel_N"] > 0), "Median_N_transport"
            ].dropna()
            + 1.0
        ).to_numpy()
        for label, _, _ in BANDS
    ]
    bp = axes[1].boxplot(
        arrays,
        positions=np.arange(1, len(BANDS) + 1),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1d2b34", "linewidth": 1.1},
        boxprops={"facecolor": "#d8e5df", "edgecolor": "#577169", "linewidth": 0.85},
        whiskerprops={"color": "#577169", "linewidth": 0.8},
        capprops={"color": "#577169", "linewidth": 0.8},
    )
    for patch in bp["boxes"]:
        patch.set_alpha(0.82)
    add_background_points(axes[1], stats_table)
    x = np.arange(1, len(BANDS) + 1)
    med = summary["Plot_median"].to_numpy()
    q25 = summary["Plot_Q25"].to_numpy()
    q75 = summary["Plot_Q75"].to_numpy()
    axes[1].errorbar(x, med, yerr=np.vstack([med - q25, q75 - med]), fmt="o", color="#176f55", ecolor="#176f55", elinewidth=1.5, capsize=3.5, markersize=4.6, zorder=5)
    add_panel_annotations(axes[1], summary, friedman, pairwise)
    style_panel_b(axes[1])
    fig.savefig(OUT_DIR / "FigureE_FlowAcc_FINAL.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "FigureE_FlowAcc_FINAL.pdf", bbox_inches="tight")
    plt.close(fig)


def write_notes(
    n_transport: np.ndarray,
    gt: tuple[float, ...],
    flowacc_1m: np.ndarray,
    corridor_mask: np.ndarray,
    threshold: float,
    inventory: pd.DataFrame,
    summary: pd.DataFrame,
    friedman: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    source = gdal.Open(str(FLOWACC_RASTER))
    source_gt = source.GetGeoTransform()
    chi_square = float(friedman.loc[0, "chi_square"])
    n_segments = int(friedman.loc[0, "N_segments"])
    kendall_w = chi_square / (n_segments * (len(BANDS) - 1))
    all_pairwise = bool((pairwise["Holm_adjusted_p"] < 0.001).all())
    text = f"""# Figure E FlowAcc Pathway Definition

This run keeps the same N transport raster and replaces the previous vector river layer with the user-provided MFD FlowAcc raster:

`{FLOWACC_RASTER}`

No N transport model was recalculated. No raw DEM vs burned DEM comparison was performed.

## Raster handling

- N transport raster: `{NLOAD_RASTER.relative_to(ROOT)}`
- N transport resolution: {abs(gt[1]):.3f} m x {abs(gt[5]):.3f} m
- FlowAcc source resolution: {abs(source_gt[1]):.3f} m x {abs(source_gt[5]):.3f} m
- FlowAcc was resampled to the 1 m N transport grid using max resampling.
- FlowAcc threshold percentile for corridor extraction: p{FLOWACC_THRESHOLD_PERCENTILE}
- FlowAcc threshold value: {threshold:.6g}
- Corridor pixels on 1 m grid: {int(corridor_mask.sum())}

## Segment definition

Because the replacement pathway source is a raster rather than a vector line, pathway segments were defined as 50 m spatial corridor blocks containing high-FlowAcc cells. Each N transport raster cell within 10 m was assigned to the nearest high-FlowAcc corridor pixel, and therefore to one corridor block only.

- Total corridor blocks: {len(inventory)}
- Complete blocks used for Friedman: {n_segments}
- Statistical unit: FlowAcc-derived corridor block

## Results

{summary.to_markdown(index=False)}

Friedman test: chi-square = {chi_square:.4f}, df = 3, p = {float(friedman.loc[0, 'p']):.3g}. Kendall's W = {kendall_w:.3f}.

All pairwise Wilcoxon signed-rank comparisons were significant after Holm correction (adjusted p < 0.001): {all_pairwise}.

Interpretation should be limited to spatial/process consistency between modeled N transport intensity and MFD FlowAcc-derived runoff corridors.
"""
    (OUT_DIR / "FigureE_FlowAcc_definition_and_results.md").write_text(text, encoding="utf-8-sig")


def copy_images() -> None:
    IMAGE_ONLY.mkdir(parents=True, exist_ok=True)
    for name in [
        "FigureE_FlowAcc_FINAL.png",
        "FigureE_FlowAcc_FINAL.pdf",
        "FigureE_FlowAcc_boxplot_version.png",
        "FigureE_FlowAcc_boxplot_version.pdf",
        "FigureE_FlowAcc_medianIQR_version.png",
        "FigureE_FlowAcc_medianIQR_version.pdf",
    ]:
        src = OUT_DIR / name
        if src.exists():
            shutil.copy2(src, IMAGE_ONLY / name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    n_transport, gt, projection = read_n_transport()
    flowacc_1m = read_flowacc_to_n_grid(gt, projection, n_transport.shape[1], n_transport.shape[0])
    corridor_mask, segment_raster, threshold, inventory = build_corridor_segments(flowacc_1m)
    distance_m, indices = ndimage.distance_transform_edt(~corridor_mask, sampling=(abs(gt[5]), abs(gt[1])), return_indices=True)
    nearest_segment = segment_raster[indices[0], indices[1]]

    stats_table = segment_level_stats(n_transport, distance_m, nearest_segment, inventory)
    qa = pairing_qa(stats_table)
    friedman, pairwise, _ = run_tests(stats_table, qa)
    summary = summarize(stats_table)

    inventory.to_csv(OUT_DIR / "FigureE_FlowAcc_segment_inventory.csv", index=False, encoding="utf-8-sig")
    stats_table.to_csv(OUT_DIR / "FigureE_FlowAcc_segment_level_stats.csv", index=False, encoding="utf-8-sig")
    qa.to_csv(OUT_DIR / "FigureE_FlowAcc_segment_pairing_QA.csv", index=False, encoding="utf-8-sig")
    friedman.to_csv(OUT_DIR / "FigureE_FlowAcc_Friedman_result.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(OUT_DIR / "FigureE_FlowAcc_pairwise_Wilcoxon_Holm.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "FigureE_FlowAcc_band_summary.csv", index=False, encoding="utf-8-sig")

    make_boxplot(stats_table, summary, friedman, pairwise)
    make_median_iqr(stats_table, summary, friedman, pairwise)
    make_final_figure(n_transport, flowacc_1m, corridor_mask, gt, stats_table, summary, friedman, pairwise)
    write_notes(n_transport, gt, flowacc_1m, corridor_mask, threshold, inventory, summary, friedman, pairwise)
    copy_images()

    print(f"FlowAcc threshold p{FLOWACC_THRESHOLD_PERCENTILE}: {threshold:.6g}")
    print(f"Corridor blocks: {len(inventory)}")
    print(f"Complete blocks: {int(friedman.loc[0, 'N_segments'])}")
    print(summary.to_string(index=False))
    print(pairwise.to_string(index=False))
    print(f"Wrote outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
