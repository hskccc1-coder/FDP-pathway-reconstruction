from __future__ import annotations

import itertools
import math
import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal, ogr
from scipy import ndimage, stats
from shapely import affinity
from shapely.geometry import LineString, MultiLineString
from shapely.ops import substring


ROOT = Path(__file__).resolve().parents[3]
NLOAD_RASTER = ROOT / "data" / "soil_water_measurements" / "单位侵蚀量" / "水土总氮多流向总氮1.tif"
RIVER_SHP = ROOT / "data" / "soil_water_measurements" / "矫正后河流线" / "river.shp"
PACKAGE = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "manuscript_backhalf_figures_final"
FIG_E_DIR = PACKAGE / "FigureE_Ntransport"
TABLE_DIR = PACKAGE / "tables"
IMAGE_ONLY = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "manuscript_backhalf_figures_images_only"

SEGMENT_LENGTH_CANDIDATES = (20.0, 50.0)
FINAL_SEGMENT_LENGTH_M = 50.0
BANDS = [
    ("0-1 m", 0.0, 1.0),
    ("1-3 m", 1.0, 3.0),
    ("3-5 m", 3.0, 5.0),
    ("5-10 m", 5.0, 10.0),
]


def read_raster() -> tuple[np.ndarray, tuple[float, ...], str, float | None]:
    ds = gdal.Open(str(NLOAD_RASTER))
    if ds is None:
        raise FileNotFoundError(NLOAD_RASTER)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(float)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        arr[np.isclose(arr, nodata)] = np.nan
    arr[arr < 0] = np.nan
    return arr, ds.GetGeoTransform(), ds.GetProjection(), nodata


def raster_extent(gt: tuple[float, ...], nx: int, ny: int) -> tuple[float, float, float, float]:
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + nx * gt[1]
    ymin = ymax + ny * gt[5]
    return xmin, xmax, ymin, ymax


def line_parts(geometry) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return [part for part in geometry.geoms if not part.is_empty and part.length > 0]
    return [
        part
        for part in getattr(geometry, "geoms", [])
        if isinstance(part, LineString) and not part.is_empty and part.length > 0
    ]


def split_line(line: LineString, segment_length_m: float) -> list[LineString]:
    pieces: list[LineString] = []
    if line.length <= 0:
        return pieces
    n = int(math.ceil(line.length / segment_length_m))
    for i in range(n):
        start = i * segment_length_m
        end = min((i + 1) * segment_length_m, line.length)
        if end - start <= 0:
            continue
        piece = substring(line, start, end)
        if isinstance(piece, LineString) and piece.length > 0:
            pieces.append(piece)
    return pieces


def build_segments(river: gpd.GeoDataFrame, segment_length_m: float) -> gpd.GeoDataFrame:
    rows = []
    segment_id = 1
    for source_id, geom in enumerate(river.geometry, start=1):
        for component_id, part in enumerate(line_parts(geom), start=1):
            for local_id, piece in enumerate(split_line(part, segment_length_m), start=1):
                rows.append(
                    {
                        "Segment_ID": segment_id,
                        "Source_ID": source_id,
                        "Component_ID": component_id,
                        "Local_ID": local_id,
                        "Segment_length_m": float(piece.length),
                        "geometry": piece,
                    }
                )
                segment_id += 1
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=river.crs)


def rasterize_segments(
    segments: gpd.GeoDataFrame, nx: int, ny: int, gt: tuple[float, ...], projection: str
) -> np.ndarray:
    mem_raster = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Int32)
    mem_raster.SetGeoTransform(gt)
    mem_raster.SetProjection(projection)
    band = mem_raster.GetRasterBand(1)
    band.Fill(0)

    drv = ogr.GetDriverByName("Memory")
    ds = drv.CreateDataSource("segments")
    layer = ds.CreateLayer("segments", geom_type=ogr.wkbLineString)
    field = ogr.FieldDefn("Segment_ID", ogr.OFTInteger)
    layer.CreateField(field)
    layer_defn = layer.GetLayerDefn()
    for row in segments.itertuples(index=False):
        feat = ogr.Feature(layer_defn)
        feat.SetField("Segment_ID", int(row.Segment_ID))
        feat.SetGeometry(ogr.CreateGeometryFromWkb(row.geometry.wkb))
        layer.CreateFeature(feat)
        feat = None
    gdal.RasterizeLayer(mem_raster, [1], layer, options=["ATTRIBUTE=Segment_ID", "ALL_TOUCHED=TRUE"])
    return band.ReadAsArray().astype(np.int32)


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
    arr: np.ndarray, segment_raster: np.ndarray, distance_m: np.ndarray, nearest_segment: np.ndarray, segments: gpd.GeoDataFrame
) -> pd.DataFrame:
    valid = np.isfinite(arr) & (distance_m <= 10.0) & (nearest_segment > 0)
    band_labels = assign_band(distance_m)
    valid &= band_labels != ""
    pixels = pd.DataFrame(
        {
            "Segment_ID": nearest_segment[valid].astype(int),
            "Distance_band": band_labels[valid],
            "N_transport": arr[valid].astype(float),
        }
    )
    pixels["N_transport_log10p1"] = np.log10(pixels["N_transport"] + 1.0)

    grouped = pixels.groupby(["Segment_ID", "Distance_band"], sort=False)
    out = grouped["N_transport"].agg(
        Valid_pixel_N="count",
        Mean_N_transport="mean",
        Median_N_transport="median",
        SD=lambda x: x.std(ddof=1),
        Q25=lambda x: x.quantile(0.25),
        Q75=lambda x: x.quantile(0.75),
    )
    log_stats = grouped["N_transport_log10p1"].agg(Mean_log="mean", Median_log="median")
    out = out.join(log_stats).reset_index()
    out["Mean_raw"] = out["Mean_N_transport"]
    out["Median_raw"] = out["Median_N_transport"]

    all_rows = pd.MultiIndex.from_product(
        [segments["Segment_ID"].astype(int), [band[0] for band in BANDS]],
        names=["Segment_ID", "Distance_band"],
    ).to_frame(index=False)
    out = all_rows.merge(out, on=["Segment_ID", "Distance_band"], how="left")
    lengths = segments[["Segment_ID", "Segment_length_m"]].copy()
    out = out.merge(lengths, on="Segment_ID", how="left")
    out["Valid_pixel_N"] = out["Valid_pixel_N"].fillna(0).astype(int)
    column_order = [
        "Segment_ID",
        "Segment_length_m",
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
    return out[column_order]


def pairing_qa(stats_table: pd.DataFrame) -> pd.DataFrame:
    counts = stats_table.pivot(index="Segment_ID", columns="Distance_band", values="Valid_pixel_N")
    rows = pd.DataFrame({"Segment_ID": counts.index.astype(int)})
    for label, _, _ in BANDS:
        rows[f"Has_{label}"] = counts[label].fillna(0).to_numpy() > 0
    rows["Complete_for_Friedman"] = rows[[f"Has_{band[0]}" for band in BANDS]].all(axis=1)
    return rows


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min((m - rank) * p_values[idx], 1.0)
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def statistical_tests(stats_table: pd.DataFrame, qa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complete_ids = qa.loc[qa["Complete_for_Friedman"], "Segment_ID"].astype(int)
    pivot = (
        stats_table[stats_table["Segment_ID"].isin(complete_ids)]
        .pivot(index="Segment_ID", columns="Distance_band", values="Median_N_transport")
        [[band[0] for band in BANDS]]
        .dropna()
    )
    if len(pivot) >= 2:
        chi_square, p_value = stats.friedmanchisquare(*[pivot[label].to_numpy() for label, _, _ in BANDS])
    else:
        chi_square, p_value = np.nan, np.nan
    friedman = pd.DataFrame(
        [
            {
                "N_segments": int(len(pivot)),
                "chi_square": float(chi_square) if np.isfinite(chi_square) else np.nan,
                "df": len(BANDS) - 1,
                "p": float(p_value) if np.isfinite(p_value) else np.nan,
            }
        ]
    )

    rows = []
    for band1, band2 in itertools.combinations([band[0] for band in BANDS], 2):
        pair = pivot[[band1, band2]].dropna()
        if len(pair) >= 2:
            try:
                w_stat, raw_p = stats.wilcoxon(pair[band1], pair[band2], zero_method="wilcox")
            except ValueError:
                w_stat, raw_p = 0.0, 1.0
        else:
            w_stat, raw_p = np.nan, np.nan
        rows.append(
            {
                "Band1": band1,
                "Band2": band2,
                "N_pairs": int(len(pair)),
                "W": float(w_stat) if np.isfinite(w_stat) else np.nan,
                "raw_p": float(raw_p) if np.isfinite(raw_p) else np.nan,
            }
        )
    pairwise = pd.DataFrame(rows)
    pairwise["Holm_adjusted_p"] = holm_adjust(pairwise["raw_p"].fillna(1.0).tolist())
    return friedman, pairwise, pivot


def compact_letters(pairwise: pd.DataFrame, alpha: float = 0.05) -> dict[str, str]:
    labels = [band[0] for band in BANDS]
    groups = {label: set() for label in labels}
    letters: list[set[str]] = []
    for label in labels:
        placed = False
        for letter_group in letters:
            can_join = True
            for other in letter_group:
                row = pairwise[
                    ((pairwise["Band1"] == label) & (pairwise["Band2"] == other))
                    | ((pairwise["Band1"] == other) & (pairwise["Band2"] == label))
                ]
                if not row.empty and float(row.iloc[0]["Holm_adjusted_p"]) < alpha:
                    can_join = False
                    break
            if can_join:
                letter_group.add(label)
                placed = True
        if not placed:
            letters.append({label})
    for i, letter_group in enumerate(letters):
        letter = chr(ord("a") + i)
        for label in letter_group:
            groups[label].add(letter)
    return {label: "".join(sorted(groups[label])) for label in labels}


def format_p(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p = NA"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def summarize_by_segment(stats_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, _, _ in BANDS:
        values = stats_table.loc[
            (stats_table["Distance_band"] == label) & (stats_table["Valid_pixel_N"] > 0),
            "Median_N_transport",
        ].dropna()
        rows.append(
            {
                "Distance_band": label,
                "N_segments": int(values.shape[0]),
                "Segment_median": float(values.median()) if len(values) else np.nan,
                "Segment_Q25": float(values.quantile(0.25)) if len(values) else np.nan,
                "Segment_Q75": float(values.quantile(0.75)) if len(values) else np.nan,
                "Segment_mean": float(values.mean()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def draw_right_panel(ax, stats_table: pd.DataFrame, friedman: pd.DataFrame, pairwise: pd.DataFrame, log_y: bool) -> None:
    rng = np.random.default_rng(42)
    letters = compact_letters(pairwise)
    summary = summarize_by_segment(stats_table)
    for xi, label in enumerate([band[0] for band in BANDS]):
        values = stats_table.loc[
            (stats_table["Distance_band"] == label) & (stats_table["Valid_pixel_N"] > 0),
            "Median_N_transport",
        ].dropna()
        plot_values = np.log10(values + 1.0) if log_y else values
        jitter = rng.uniform(-0.11, 0.11, size=len(plot_values))
        ax.scatter(
            np.full(len(plot_values), xi) + jitter,
            plot_values,
            s=7,
            color="#6f7f8f",
            alpha=0.26,
            linewidths=0,
            zorder=1,
        )
        center_raw = summary.loc[summary["Distance_band"] == label, "Segment_median"].iloc[0]
        q25_raw = summary.loc[summary["Distance_band"] == label, "Segment_Q25"].iloc[0]
        q75_raw = summary.loc[summary["Distance_band"] == label, "Segment_Q75"].iloc[0]
        center = np.log10(center_raw + 1.0) if log_y else center_raw
        q25 = np.log10(q25_raw + 1.0) if log_y else q25_raw
        q75 = np.log10(q75_raw + 1.0) if log_y else q75_raw
        ax.errorbar(
            xi,
            center,
            yerr=np.array([[center - q25], [q75 - center]]),
            fmt="o",
            color="#1f6f54",
            ecolor="#1f6f54",
            elinewidth=1.5,
            capsize=3,
            markersize=4.5,
            zorder=3,
        )
        y_letter = q75 + max(0.04, abs(q75) * 0.08) if log_y else q75 + max(0.2, abs(q75) * 0.2)
        ax.text(xi, y_letter, letters[label], ha="center", va="bottom", fontsize=8, color="#1f2d3a")
    ax.set_xticks(np.arange(len(BANDS)))
    ax.set_xticklabels([band[0] for band in BANDS])
    ax.set_xlabel("Distance from reconstructed runoff pathway (m)")
    if log_y:
        ax.set_ylabel("log10(N transport intensity + 1)")
    else:
        ax.set_ylabel("Modeled N transport intensity")
    p_text = format_p(float(friedman.loc[0, "p"]))
    ax.text(
        0.02,
        0.98,
        f"Friedman, {p_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#1f2d3a",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)


def make_segment_figures(stats_table: pd.DataFrame, friedman: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    for log_y, suffix in [(False, "linearY"), (True, "logY")]:
        fig, ax = plt.subplots(figsize=(4.1, 3.35), constrained_layout=True)
        draw_right_panel(ax, stats_table, friedman, pairwise, log_y=log_y)
        ax.set_title("N transport intensity across pathway-distance bands", loc="left", fontweight="bold", fontsize=10)
        fig.savefig(FIG_E_DIR / f"FigureE_segment_{suffix}.png", dpi=600, bbox_inches="tight")
        fig.savefig(FIG_E_DIR / f"FigureE_segment_{suffix}.pdf", bbox_inches="tight")
        plt.close(fig)


def make_final_figure(
    arr: np.ndarray,
    gt: tuple[float, ...],
    stats_table: pd.DataFrame,
    friedman: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    river = gpd.read_file(RIVER_SHP)
    river_km = river.copy()
    river_km["geometry"] = river_km.geometry.map(lambda geom: affinity.scale(geom, xfact=0.001, yfact=0.001, origin=(0, 0)))
    extent_m = raster_extent(gt, arr.shape[1], arr.shape[0])
    extent = tuple(value * 0.001 for value in extent_m)
    display = np.log10(arr + 1.0)
    display[~np.isfinite(arr)] = np.nan
    vmin, vmax = np.nanpercentile(display, [2, 98])

    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.35, 3.55),
        gridspec_kw={"width_ratios": [1.0, 1.08]},
        constrained_layout=True,
    )
    im = axes[0].imshow(
        display,
        extent=extent,
        origin="upper",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    river_km.plot(ax=axes[0], color="white", linewidth=1.15, alpha=0.96)
    river_km.plot(ax=axes[0], color="#cf3f2e", linewidth=0.48)
    axes[0].set_title("(a) Modeled N transport intensity", loc="left", fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Easting (km)")
    axes[0].set_ylabel("Northing (km)")
    cb = fig.colorbar(im, ax=axes[0], orientation="horizontal", shrink=0.72, pad=0.08)
    cb.set_label("log10(N transport intensity + 1)")
    axes[0].text(
        0.02,
        0.02,
        "Reconstructed surface-runoff pathway",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="white",
        bbox={"facecolor": "#263238", "edgecolor": "none", "alpha": 0.72, "pad": 2.2},
    )

    draw_right_panel(axes[1], stats_table, friedman, pairwise, log_y=True)
    axes[1].set_title("(b) N transport intensity across pathway-distance bands", loc="left", fontweight="bold", fontsize=10)
    fig.savefig(FIG_E_DIR / "FigureE_Ntransport_FINAL.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_E_DIR / "FigureE_Ntransport_FINAL.pdf", bbox_inches="tight")
    plt.close(fig)


def write_definition_md(
    arr: np.ndarray,
    gt: tuple[float, ...],
    river: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    candidate_counts: dict[float, int],
) -> None:
    nx, ny = arr.shape[1], arr.shape[0]
    x_res, y_res = abs(gt[1]), abs(gt[5])
    footprint_area_km2 = (nx * x_res * ny * y_res) / 1_000_000
    total_length_m = float(river.length.sum())
    density = (total_length_m / 1000.0) / footprint_area_km2
    component_count = sum(len(line_parts(geom)) for geom in river.geometry)
    text = f"""# Figure E Segment Definition

## Input data

- N transport raster: `{NLOAD_RASTER.relative_to(ROOT)}`
- Reconstructed pathway layer: `{RIVER_SHP.relative_to(ROOT)}`
- Raster resolution: {x_res:.3f} m x {y_res:.3f} m
- Raster footprint: {nx} x {ny} pixels ({footprint_area_km2:.3f} km2)
- Total reconstructed pathway length: {total_length_m:.3f} m
- Pathway components: {component_count}
- Pathway density: {density:.3f} km pathway per km2 raster footprint

## Segment-length decision

Candidate segment counts were:

- 20 m: {candidate_counts[20.0]} segments
- 50 m: {candidate_counts[50.0]} segments

The final segment length was set to 50 m. Both candidate definitions exceeded the preferred target of roughly 50-200 valid segments because the reconstructed pathway network is dense relative to the raster footprint. The 20 m definition produced very dense adjacent segments and higher local redundancy. The 50 m definition therefore provides the lower-redundancy option among the requested candidates while retaining local pathway-scale spatial resolution.

## Statistical unit

Raster cells were assigned to the nearest rasterized pathway segment within mutually exclusive distance bands of 0-1 m, 1-3 m, 3-5 m, and 5-10 m. Each raster cell was assigned to at most one segment. Segment-level medians were then used as the repeated statistical unit for the Friedman and paired Wilcoxon-Holm analyses.

## Final segment inventory

- Total generated segments: {len(segments)}
- Segment length target: {FINAL_SEGMENT_LENGTH_M:.1f} m
- Median realized segment length: {segments["Segment_length_m"].median():.3f} m
- Minimum realized segment length: {segments["Segment_length_m"].min():.3f} m
- Maximum realized segment length: {segments["Segment_length_m"].max():.3f} m
"""
    (FIG_E_DIR / "FigureE_segment_definition.md").write_text(text, encoding="utf-8-sig")


def write_caption_md(friedman: pd.DataFrame) -> None:
    text = f"""# Figure E Caption

Figure E evaluates whether modeled nitrogen transport intensity is spatially consistent with the reconstructed surface-runoff pathway layer. Panel (a) maps the user-confirmed nitrogen transport raster (`{NLOAD_RASTER.name}`), displayed as log10(N transport intensity + 1) to reduce the visual dominance of highly skewed high-intensity cells. The overlaid line represents the reconstructed surface-runoff pathway and is used as a spatial reference, not as a directly observed pollutant transport path.

Panel (b) summarizes modeled N transport intensity across mutually exclusive distance bands from the reconstructed pathway: 0-1 m, 1-3 m, 3-5 m, and 5-10 m. Raster cells were assigned to the nearest 50 m pathway segment, so each cell contributed to at most one segment-band combination. Light points represent segment-level median values, and green points with error bars show the median and interquartile range across pathway segments. The displayed n is the number of complete pathway segments used in the repeated-measures test, not the number of raster cells.

The Friedman test was applied to segment-level medians across the four distance bands (n = {int(friedman.loc[0, "N_segments"])} complete segments). Pairwise comparisons used Wilcoxon signed-rank tests with Holm correction. This figure supports a process-consistency interpretation between modeled N transport intensity and reconstructed runoff pathways; it should not be interpreted as direct validation of observed nitrogen transport.
"""
    (FIG_E_DIR / "FigureE_caption.md").write_text(text, encoding="utf-8-sig")


def copy_outputs_to_images_only() -> None:
    IMAGE_ONLY.mkdir(parents=True, exist_ok=True)
    for name in [
        "FigureE_Ntransport_FINAL.png",
        "FigureE_Ntransport_FINAL.pdf",
        "FigureE_segment_linearY.png",
        "FigureE_segment_linearY.pdf",
        "FigureE_segment_logY.png",
        "FigureE_segment_logY.pdf",
    ]:
        src = FIG_E_DIR / name
        if src.exists():
            shutil.copy2(src, IMAGE_ONLY / name)


def main() -> None:
    FIG_E_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    arr, gt, projection, _ = read_raster()
    river = gpd.read_file(RIVER_SHP)
    candidate_counts = {
        candidate: len(build_segments(river, candidate)) for candidate in SEGMENT_LENGTH_CANDIDATES
    }
    segments = build_segments(river, FINAL_SEGMENT_LENGTH_M)

    segment_raster = rasterize_segments(segments, arr.shape[1], arr.shape[0], gt, projection)
    foreground = segment_raster > 0
    distance_m, indices = ndimage.distance_transform_edt(
        ~foreground, sampling=(abs(gt[5]), abs(gt[1])), return_indices=True
    )
    nearest_segment = segment_raster[indices[0], indices[1]]

    stats_table = segment_level_stats(arr, segment_raster, distance_m, nearest_segment, segments)
    qa = pairing_qa(stats_table)
    friedman, pairwise, _ = statistical_tests(stats_table, qa)
    summary = summarize_by_segment(stats_table)

    stats_table.to_csv(FIG_E_DIR / "FigureE_segment_level_stats.csv", index=False, encoding="utf-8-sig")
    qa.to_csv(FIG_E_DIR / "FigureE_segment_pairing_QA.csv", index=False, encoding="utf-8-sig")
    friedman.to_csv(FIG_E_DIR / "FigureE_Friedman_result.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(FIG_E_DIR / "FigureE_pairwise_Wilcoxon_Holm.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(FIG_E_DIR / "FigureE_segment_band_summary.csv", index=False, encoding="utf-8-sig")
    for file_name in [
        "FigureE_segment_level_stats.csv",
        "FigureE_segment_pairing_QA.csv",
        "FigureE_Friedman_result.csv",
        "FigureE_pairwise_Wilcoxon_Holm.csv",
        "FigureE_segment_band_summary.csv",
    ]:
        shutil.copy2(FIG_E_DIR / file_name, TABLE_DIR / file_name)

    make_segment_figures(stats_table, friedman, pairwise)
    make_final_figure(arr, gt, stats_table, friedman, pairwise)
    write_definition_md(arr, gt, river, segments, candidate_counts)
    write_caption_md(friedman)
    copy_outputs_to_images_only()

    print(f"Raster resolution: {abs(gt[1]):.3f} m x {abs(gt[5]):.3f} m")
    print(f"Total pathway length: {float(river.length.sum()):.3f} m")
    print(f"Final segment length: {FINAL_SEGMENT_LENGTH_M:.1f} m")
    print(f"Generated segments: {len(segments)}")
    print(f"Complete segments: {int(friedman.loc[0, 'N_segments'])}")
    print(f"Friedman p: {float(friedman.loc[0, 'p']):.6g}")
    print(f"Wrote Figure E to: {FIG_E_DIR}")


if __name__ == "__main__":
    main()
