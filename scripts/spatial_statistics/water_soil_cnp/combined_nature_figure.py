"""Combine final model coefficients and seasonal observations at publication size."""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator, FormatStrFormatter
import numpy as np
import pandas as pd

from nature_style_recommended_figures import letter_position, x_limits


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "manuscript_backhalf_figures_final"
OUT = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "combined_Nature_figure_20260907_16x9_Times"
STEM = "Figure_combined_models_seasonal_16x9_Times"
SEASONS = ["Autumn", "Winter", "Spring", "Summer"]
COLORS = {"TN": "#2F6F8F", "TP": "#994F78", "Soil": "#617D62"}
PREDICTORS = ["Hydrological distance", "High-intensity agriculture",
              "Natural vegetation", "Flow accumulation"]


def style():
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.serif": ["Times New Roman"],
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 16,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "xtick.major.size": 3,
        "ytick.major.size": 3, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold", "axes.unicode_minus": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def title(fig, ax, label, text):
    pos = ax.get_position()
    fig.text(pos.x0 - 0.008, pos.y1 + 0.016, "({}) {}".format(label, text),
             fontsize=16, va="bottom")


def forest(ax, frame, soil=False):
    entries = []
    if soil:
        for response in ["TN", "TP", "TOC"]:
            entries.append((response, None))
            group = frame[frame.Response == response].set_index("Predictor")
            entries.extend((p, group.loc[p]) for p in PREDICTORS)
            if response != "TOC":
                entries.append(("", None))
    else:
        group = frame.set_index("Predictor")
        entries = [(p, group.loc[p]) for p in
                   ["Hydrological distance", "Dissolved oxygen", "pH"]]
    positions = np.arange(len(entries))[::-1]
    ax.axvline(0, color="#444444", linewidth=0.8, zorder=1)
    ticks, labels = [], []
    for y, (label, row) in zip(positions, entries):
        if not label:
            continue
        ticks.append(y)
        labels.append(label)
        if row is not None:
            value = float(row.Estimate)
            ax.errorbar(value, y,
                        xerr=[[value - row.CI_low], [row.CI_high - value]],
                        fmt="o", color=COLORS["Soil" if soil else "TN"],
                        ecolor="#333333", elinewidth=1.0, capsize=2.5,
                        capthick=1.0, markersize=5.5, zorder=3)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    for tick in ax.get_yticklabels():
        if tick.get_text() in ["TN", "TP", "TOC"]:
            tick.set_fontweight("bold")
            tick.set_fontsize(12.5)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_ylim(-0.75, len(entries) - 0.25)
    ax.set_xlim(*x_limits(frame))
    ax.xaxis.set_major_locator(MultipleLocator(0.2 if soil else 0.05))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f" if soil else "%.2f"))
    ax.spines[["top", "right", "left"]].set_visible(False)


def seasonal(ax, data, summary, tests, letters, pollutant):
    col = pollutant + "_mg_L"
    log_y = pollutant == "TP"
    values = data[col].to_numpy(dtype=float)
    low, high = float(values.min()), float(values.max())
    if log_y:
        assert low > 0, "Log axis requires strictly positive measurements"
        ax.set_yscale("log")
        ax.set_ylim(low * 0.64, high * 2.5)
        ax.yaxis.set_minor_locator(NullLocator())
    else:
        ax.set_ylim(0.15, high + (high - low) * 0.15)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
    sites = sorted(data.Site_ID.unique())
    offsets = dict(zip(sites, np.linspace(-0.055, 0.055, len(sites))))
    for site in sites:
        sub = data[data.Site_ID == site].set_index("Season").reindex(SEASONS)
        x = np.arange(4) + offsets[site]
        ax.plot(x, sub[col], color=COLORS[pollutant], linewidth=0.65,
                alpha=0.14, zorder=1)
        ax.scatter(x, sub[col], s=30, facecolor="white",
                   edgecolor=COLORS[pollutant], linewidth=0.8, zorder=3)
    for i, season in enumerate(SEASONS):
        row = summary[(summary.Pollutant == pollutant) &
                      (summary.Season == season)].iloc[0]
        ax.vlines(i, row.Q25, row.Q75, color="black", linewidth=1.4, zorder=4)
        ax.scatter([i], [row["median"]], color="black", s=40, zorder=5)
        letter = letters[(letters.Pollutant == pollutant) &
                         (letters.Season == season)].compact_letter.iloc[0]
        y = letter_position(data.loc[data.Season == season, col].to_numpy(),
                            row.Q75, low, high, log_y)
        ax.annotate(letter, (i, y), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=13)
    p = tests.loc[tests.Pollutant == pollutant, "p"].iloc[0]
    p_label = r"$P < 0.001$" if p < 0.001 else r"$P = %.3f$" % p
    ax.text(0.02, 0.975, "Friedman, " + p_label, transform=ax.transAxes,
            va="top", fontsize=12.5)
    ax.set_xlim(-0.24, 3.25)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([s + "\n" + str(y) for s, y in
                        zip(SEASONS, [2024, 2024, 2025, 2025])])
    ax.tick_params(axis="x", pad=5)
    ax.set_ylabel(pollutant + r" concentration (mg L$^{-1}$)", labelpad=4)
    ax.spines[["top", "right"]].set_visible(False)


def clear_letter_collisions(fig, axes):
    # Compare rendered text bounds with marker centres at the final print size.
    for ax in axes:
        points = np.concatenate([artist.get_offsets() for artist in ax.collections
                                 if isinstance(artist, matplotlib.collections.PathCollection)])
        pixels = ax.transData.transform(points)
        for annotation in ax.texts:
            if not isinstance(annotation, matplotlib.text.Annotation):
                continue
            for _ in range(40):
                fig.canvas.draw()
                box = annotation.get_window_extent(fig.canvas.get_renderer())
                padding = 4 * fig.dpi / 72
                collision = ((pixels[:, 0] >= box.x0 - padding) &
                             (pixels[:, 0] <= box.x1 + padding) &
                             (pixels[:, 1] >= box.y0 - padding) &
                             (pixels[:, 1] <= box.y1 + padding)).any()
                if not collision:
                    break
                annotation.set_position((0, annotation.get_position()[1] + 3))
            else:
                raise RuntimeError("Could not clear a compact-letter collision")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_data = OUT / "source_data"
    source_data.mkdir(exist_ok=True)
    paths = [SOURCE / "FigureC_seasonal" / ("FigureC_" + name + ".csv")
             for name in ["plot_data", "seasonal_summary", "statistics",
                          "compact_letters", "pairwise_statistics"]]
    paths.append(SOURCE / "FigureD_multivariable" / "FigureD_multivariable_coefficients.csv")
    manifest = []
    for path in paths:
        shutil.copy2(str(path), str(source_data / path.name))
        manifest.append({"source": str(path.relative_to(ROOT)),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    data, summary, tests, letters, pairs, coeffs = [pd.read_csv(p, dtype={"Site_ID": str}) for p in paths]
    assert not data.duplicated(["Site_ID", "Season"]).any()
    assert len(data) == 48 and data.Site_ID.nunique() == 12
    assert data.groupby("Site_ID").Season.nunique().eq(4).all()
    for pollutant in ["TN", "TP"]:
        for season in SEASONS:
            vals = data.loc[data.Season == season, pollutant + "_mg_L"]
            row = summary[(summary.Pollutant == pollutant) & (summary.Season == season)].iloc[0]
            assert len(vals) == row.N
            np.testing.assert_allclose(np.quantile(vals, [0.25, 0.5, 0.75]),
                                       [row.Q25, row["median"], row.Q75])
    plotted = coeffs[coeffs.Plotted_in_FigureD == True]
    water = plotted[plotted.Medium == "Surface water"]
    soil = plotted[plotted.Medium == "Soil"]
    assert len(water) == 3 and len(soil) == 12
    style()
    fig = plt.figure(figsize=(12.8, 7.2))
    a = fig.add_axes([0.190, 0.750, 0.370, 0.170])
    b = fig.add_axes([0.190, 0.105, 0.370, 0.535])
    c = fig.add_axes([0.680, 0.565, 0.295, 0.355])
    d = fig.add_axes([0.680, 0.105, 0.295, 0.355])
    forest(a, water)
    forest(b, soil, soil=True)
    b.set_xlabel(r"Standardized coefficient ($\beta$)", labelpad=7)
    seasonal(c, data, summary, tests, letters, "TN")
    seasonal(d, data, summary, tests, letters, "TP")
    for ax, label, name in [(a, "a", "Surface-water TN"), (b, "b", "Soil nutrients"),
                            (c, "c", "TN"), (d, "d", "TP")]:
        title(fig, ax, label, name)
    clear_letter_collisions(fig, [c, d])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        # Tick locators can create invisible, out-of-range tick labels.
        if any(text in ax.get_xticklabels() + ax.get_yticklabels() for ax in fig.axes):
            if not any(box.overlaps(ax.get_tightbbox(renderer)) for ax in fig.axes):
                continue
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
            outside.append(text.get_text())
    assert not outside, "Text extends outside the canvas: " + str(outside)
    for ext in ["png", "pdf", "svg", "tiff"]:
        kwargs = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(OUT / (STEM + "." + ext), dpi=600, facecolor="white", **kwargs)
    fig.savefig(OUT / (STEM + "_preview.png"), dpi=180, facecolor="white")
    plt.close(fig)
    qa = {"figure_size_mm": [325.12, 182.88], "aspect_ratio": "16:9",
          "font": "Times New Roman", "panel_title_format": "(a) Title",
          "font_size_pt": {"ticks": 12, "labels": 13, "titles": 16, "panels": 16},
          "raster_dpi": 600, "seasonal_rows": len(data), "paired_sites": 12,
          "model_coefficients": len(plotted), "summaries_match_raw_data": True,
          "compact_letters_clear_of_markers": True,
          "text_outside_canvas": outside, "sources": manifest}
    (OUT / "provenance_and_QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    caption = """# Combined Figure: Model Coefficients and Seasonal Water Quality

(a) Continuous-predictor coefficients from the final surface-water TN W3 mixed
model (54 observations, 14 sites). (b) Soil TN, TP and TOC coefficients from the
D2 network-only OLS models with HC3 uncertainty estimates (40 sites). Coloured
points show the reported standardized coefficients; horizontal bars show 95%
confidence intervals. Vertical lines mark zero. Panels a and b use independent
x-axis scales. Surface-water TP retained seasonal terms only and is not shown
in the continuous-predictor coefficient panels.

(c, d) Seasonal surface-water TN and TP for the balanced dataset of 12 sites
sampled in all four seasons (48 observations). Open circles show observations;
faint coloured lines connect the same site. Black points and bars show medians
and interquartile ranges. Letters reproduce the final Holm-adjusted paired
Wilcoxon groupings following the Friedman tests. TN uses a linear axis; TP uses
a logarithmic axis. All measurements, model estimates, confidence intervals,
P values and letter groupings are retained from the final result tables.

The original Figure C and Figure D were regenerated as one four-panel figure.
Typography: Times New Roman, including mathematical text; tick labels 12 pt,
axis labels and grouping letters 13 pt, titles 16 pt with parenthesized panel
letters, e.g. (a) Surface-water TN. Canvas: 12.8 x 7.2 inches (16:9),
325.12 x 182.88 mm. Exports: 600-dpi PNG
and LZW-compressed TIFF; editable vector PDF and SVG; lightweight PNG preview.

Reproduce from the workspace root: `python analysis/combined_nature_figure.py`.
Source CSV snapshots and their SHA-256 hashes accompany the figure.
"""
    (OUT / "Figure_caption_and_notes.md").write_text(caption, encoding="utf-8")
    print(json.dumps(qa, indent=2))
    print("Output: " + str(OUT))


if __name__ == "__main__":
    main()
