from __future__ import annotations

import itertools
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def find_project_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "data" / "soil_water_measurements" / "four_season_TN_TP").exists() and (candidate / "results").exists():
            return candidate
    return Path(__file__).resolve().parents[3]


ROOT = find_project_root()
PACKAGE = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "manuscript_backhalf_figures_final"
FIG_C_DIR = PACKAGE / "FigureC_seasonal"
FIG_D_DIR = PACKAGE / "FigureD_multivariable"
FIG_E_DIR = PACKAGE / "FigureE_Ntransport"
TABLE_DIR = PACKAGE / "tables"
SCRIPT_DIR = PACKAGE / "scripts"
README_DIR = PACKAGE / "README"

SEASONS = ["Autumn", "Winter", "Spring", "Summer"]
SEASON_LABELS = {
    "Autumn": "Autumn\n2024",
    "Winter": "Winter\n2024",
    "Spring": "Spring\n2025",
    "Summer": "Summer\n2025",
}
POLLUTANTS = {"TN": "TN_mg_L", "TP": "TP_mg_L"}
COLORS = {"TN": "#356f8c", "TP": "#8f4c73", "Soil": "#5f7f63"}


def ensure_dirs() -> None:
    for path in [FIG_C_DIR, FIG_D_DIR, FIG_E_DIR, TABLE_DIR, SCRIPT_DIR, README_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def p_text(p: float) -> str:
    if pd.isna(p):
        return "p = NA"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def holm_adjust(p_values: List[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min((m - rank) * p[idx], 1.0)
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def compact_letters(seasons: List[str], pairwise: pd.DataFrame) -> Dict[str, str]:
    sig_pairs = {
        tuple(sorted([row.Season1, row.Season2]))
        for row in pairwise.itertuples(index=False)
        if bool(row.significant)
    }
    nonsig_subsets = []
    for size in range(len(seasons), 0, -1):
        for subset in itertools.combinations(seasons, size):
            if all(tuple(sorted(pair)) not in sig_pairs for pair in itertools.combinations(subset, 2)):
                nonsig_subsets.append(set(subset))
    assigned = {season: "" for season in seasons}
    covered_nonsig = set()
    letters = "abcdefghijklmnopqrstuvwxyz"
    letter_idx = 0
    for subset in nonsig_subsets:
        needed = False
        for season in subset:
            if not assigned[season]:
                needed = True
        for pair in itertools.combinations(sorted(subset), 2):
            if tuple(sorted(pair)) not in sig_pairs and tuple(sorted(pair)) not in covered_nonsig:
                needed = True
        if not needed:
            continue
        letter = letters[letter_idx]
        letter_idx += 1
        for season in subset:
            assigned[season] += letter
        for pair in itertools.combinations(sorted(subset), 2):
            covered_nonsig.add(tuple(sorted(pair)))
    for season in seasons:
        if not assigned[season]:
            assigned[season] = letters[letter_idx]
            letter_idx += 1
    return assigned


def load_figure_c_data() -> pd.DataFrame:
    path = ROOT / "data" / "soil_water_measurements" / "four_season_TN_TP" / "water_TN_TP_four_seasons_balanced_12_sites.csv"
    df = pd.read_csv(path, dtype={"Site_ID": str})
    df = df[df["Season"].isin(SEASONS)].copy()
    df["Season"] = pd.Categorical(df["Season"], categories=SEASONS, ordered=True)
    complete_sites = []
    for site_id, sub in df.groupby("Site_ID"):
        ok = set(sub["Season"].astype(str)) == set(SEASONS)
        ok = ok and sub[["TN_mg_L", "TP_mg_L"]].notna().all(axis=None)
        if ok:
            complete_sites.append(site_id)
    df = df[df["Site_ID"].isin(complete_sites)].copy()
    df = df.sort_values(["Site_ID", "Season"]).reset_index(drop=True)
    return df


def figure_c_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qa_rows = []
    desc_rows = []
    overall_rows = []
    pairwise_rows = []
    letter_rows = []
    for season in SEASONS:
        sub = df[df["Season"] == season]
        qa_rows.append(
            {
                "Season": season,
                "N_TN": int(sub["TN_mg_L"].notna().sum()),
                "N_TP": int(sub["TP_mg_L"].notna().sum()),
                "N_sites": int(sub["Site_ID"].nunique()),
            }
        )
    for pollutant, col in POLLUTANTS.items():
        wide = df.pivot(index="Site_ID", columns="Season", values=col)[SEASONS].dropna()
        stat, p_value = stats.friedmanchisquare(*[wide[season].to_numpy() for season in SEASONS])
        overall_rows.append(
            {
                "Pollutant": pollutant,
                "Test": "Friedman test",
                "N_sites": len(wide),
                "Statistic": stat,
                "p": p_value,
            }
        )
        raw_p = []
        temp_rows = []
        for s1, s2 in itertools.combinations(SEASONS, 2):
            paired = wide[[s1, s2]].dropna()
            result = stats.wilcoxon(paired[s1], paired[s2], zero_method="wilcox", correction=False, alternative="two-sided")
            raw_p.append(result.pvalue)
            temp_rows.append(
                {
                    "Pollutant": pollutant,
                    "Season1": s1,
                    "Season2": s2,
                    "N_pairs": len(paired),
                    "raw_p": result.pvalue,
                }
            )
        adjusted = holm_adjust(raw_p)
        for row, adj in zip(temp_rows, adjusted):
            row["Holm_adjusted_p"] = adj
            row["significant"] = bool(adj < 0.05)
            pairwise_rows.append(row)
        pairwise_pollutant = pd.DataFrame(temp_rows)
        pairwise_pollutant["Holm_adjusted_p"] = adjusted
        pairwise_pollutant["significant"] = pairwise_pollutant["Holm_adjusted_p"] < 0.05
        letters = compact_letters(SEASONS, pairwise_pollutant)
        for season in SEASONS:
            x = df.loc[df["Season"] == season, col].dropna()
            q25, q75 = np.percentile(x, [25, 75])
            desc_rows.append(
                {
                    "Pollutant": pollutant,
                    "Season": season,
                    "N": len(x),
                    "mean": x.mean(),
                    "median": x.median(),
                    "SD": x.std(ddof=1),
                    "Q25": q25,
                    "Q75": q75,
                    "IQR": q75 - q25,
                    "min": x.min(),
                    "max": x.max(),
                }
            )
            letter_rows.append({"Pollutant": pollutant, "Season": season, "compact_letter": letters[season]})

    qa = pd.DataFrame(qa_rows)
    desc = pd.DataFrame(desc_rows)
    overall = pd.DataFrame(overall_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    letters = pd.DataFrame(letter_rows)
    qa.to_csv(FIG_C_DIR / "FigureC_data_QA.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(FIG_C_DIR / "FigureC_seasonal_summary.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(FIG_C_DIR / "FigureC_statistics.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(FIG_C_DIR / "FigureC_pairwise_statistics.csv", index=False, encoding="utf-8-sig")
    letters.to_csv(FIG_C_DIR / "FigureC_compact_letters.csv", index=False, encoding="utf-8-sig")
    for path in [
        "FigureC_data_QA.csv",
        "FigureC_seasonal_summary.csv",
        "FigureC_statistics.csv",
        "FigureC_pairwise_statistics.csv",
        "FigureC_compact_letters.csv",
    ]:
        shutil.copy2(FIG_C_DIR / path, TABLE_DIR / path)
    return qa, desc, overall, pairwise


def letter_position(values: np.ndarray, q75: float, y_min: float, y_max: float, log_y: bool) -> float:
    if log_y:
        log_values = np.log10(values)
        log_span = np.log10(y_max) - np.log10(y_min)
        step = max(log_span * 0.055, 0.035)
        tolerance = max(log_span * 0.03, 0.025)
        candidate = np.log10(q75) + step
        for _ in range(6):
            if not np.any(np.abs(log_values - candidate) < tolerance):
                return float(10**candidate)
            candidate += step
        return float(10**candidate)

    span = y_max - y_min
    step = span * 0.045 if span else 0.05
    tolerance = span * 0.035 if span else 0.04
    candidate = q75 + step
    for _ in range(6):
        if not np.any(np.abs(values - candidate) < tolerance):
            return float(candidate)
        candidate += step
    return float(candidate)


def draw_season_panel(ax, df: pd.DataFrame, desc: pd.DataFrame, overall: pd.DataFrame, pollutant: str, log_y: bool = False) -> None:
    col = POLLUTANTS[pollutant]
    x_positions = np.arange(len(SEASONS))
    if log_y and (df[col].dropna() <= 0).any():
        raise ValueError(f"{pollutant} log-y plot requires strictly positive values.")
    for _, sub in df.groupby("Site_ID"):
        sub = sub.sort_values("Season")
        y = sub[col].to_numpy(dtype=float)
        ax.plot(x_positions, y, color="0.82", linewidth=0.55, alpha=0.38, zorder=1)
    rng = np.random.default_rng(20260827)
    for idx, season in enumerate(SEASONS):
        values = df.loc[df["Season"] == season, col].to_numpy(dtype=float)
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=22,
            facecolor="white",
            edgecolor=COLORS[pollutant],
            linewidth=0.8,
            alpha=0.82,
            zorder=3,
        )
        d = desc[(desc["Pollutant"] == pollutant) & (desc["Season"] == season)].iloc[0]
        q25 = d["Q25"]
        q75 = d["Q75"]
        median = d["median"]
        ax.vlines(idx, q25, q75, color="black", linewidth=1.45, zorder=5)
        ax.scatter([idx], [median], color="black", s=38, zorder=6)
    stat = overall[overall["Pollutant"] == pollutant].iloc[0]
    ax.text(
        0.02,
        0.96,
        f"Friedman, {p_text(stat['p'])}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    letter_table = pd.read_csv(FIG_C_DIR / "FigureC_compact_letters.csv")
    y_max = float(np.nanmax(df[col]))
    y_min = float(np.nanmin(df[col]))
    pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.1
    for idx, season in enumerate(SEASONS):
        letter = letter_table[(letter_table["Pollutant"] == pollutant) & (letter_table["Season"] == season)]["compact_letter"].iloc[0]
        d = desc[(desc["Pollutant"] == pollutant) & (desc["Season"] == season)].iloc[0]
        season_values = df.loc[df["Season"] == season, col].to_numpy(dtype=float)
        letter_y = letter_position(season_values, float(d["Q75"]), y_min, y_max, log_y)
        ax.text(idx, letter_y, letter, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([SEASON_LABELS[s] for s in SEASONS])
    ylabel = f"{pollutant} concentration (mg L$^{{-1}}$)"
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", which="major", linestyle=":", alpha=0.35)
    if log_y:
        lower = max(y_min * 0.72, np.nextafter(0, 1))
        upper = y_max * 1.28
        ax.set_ylim(lower, upper)
    else:
        ax.set_ylim(bottom=max(0, y_min - pad * 0.55), top=y_max + pad * 1.15)


def make_figure_c(df: pd.DataFrame, desc: pd.DataFrame, overall: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    tp_values = df["TP_mg_L"].dropna()
    zero_count = int((tp_values == 0).sum())
    if zero_count:
        raise ValueError(f"TP contains {zero_count} zero values; min TP = {tp_values.min():.6g}.")

    for suffix, tp_log_y in [("linear", False), ("logY", True)]:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
        draw_season_panel(axes[0], df, desc, overall, "TN", log_y=False)
        draw_season_panel(axes[1], df, desc, overall, "TP", log_y=tp_log_y)
        axes[0].set_title("(a) TN", loc="left", fontweight="bold", fontsize=10)
        axes[1].set_title("(b) TP", loc="left", fontweight="bold", fontsize=10)
        fig.savefig(FIG_C_DIR / f"FigureC_seasonal_water_response_{suffix}.png", dpi=600, bbox_inches="tight")
        fig.savefig(FIG_C_DIR / f"FigureC_seasonal_water_response_{suffix}.pdf", bbox_inches="tight")
        if suffix == "linear":
            fig.savefig(FIG_C_DIR / "FigureC_seasonal_water_response.png", dpi=600, bbox_inches="tight")
            fig.savefig(FIG_C_DIR / "FigureC_seasonal_water_response.pdf", bbox_inches="tight")
        plt.close(fig)


def build_figure_d_table() -> pd.DataFrame:
    rows = []
    water = pd.read_csv(ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_phase5_final" / "water" / "water_final_mixed_model_coefficients.csv")
    water_tn = water[
        (water["Response"] == "TN") & (water["Predictor"].isin(["Hydrological distance", "DO", "pH"]))
    ].copy()
    water_tn["Predictor"] = water_tn["Predictor"].replace({"DO": "Dissolved oxygen"})
    for row in water_tn.itertuples(index=False):
        rows.append(
            {
                "Medium": "Surface water",
                "Response": row.Response,
                "Predictor": row.Predictor,
                "Estimate": row.Estimate,
                "Std_Beta": row.Std_Beta,
                "SE": row.SE,
                "CI_low": row.CI_low,
                "CI_high": row.CI_high,
                "p": row.p,
                "N": row.N_obs,
                "N_sites": row.N_sites,
                "Model": "TN W3 MixedLM: Season + distance + DO + pH + (1|Site_ID)",
                "Plotted_in_FigureD": True,
                "Panel_note": "",
            }
        )

    tp_gee = pd.read_csv(
        ROOT
        / "results"
        / "water_soil_multivariable_phase6_statistical_robustness"
        / "water"
        / "water_TP_repeated_measure_sensitivity.csv"
    )
    tp_gee = tp_gee[tp_gee["Method"] == "GEE exchangeable"].copy()
    for row in tp_gee.itertuples(index=False):
        rows.append(
            {
                "Medium": "Surface water",
                "Response": "TP",
                "Predictor": row.Contrast,
                "Estimate": row.Beta,
                "Std_Beta": np.nan,
                "SE": row.SE,
                "CI_low": row.CI_low,
                "CI_high": row.CI_high,
                "p": row.p,
                "N": row.N,
                "N_sites": row.N_sites,
                "Model": "TP GEE exchangeable: Season-only final reporting model",
                "Plotted_in_FigureD": False,
                "Panel_note": "Season-only final model; no additional continuous predictors retained.",
            }
        )

    soil = pd.read_csv(ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_phase5_final" / "soil" / "soil_D2_final_coefficients.csv")
    soil = soil[soil["Term"] != "Intercept"].copy()
    soil["Predictor"] = soil["Predictor"].replace(
        {
            "High-intensity agriculture": "High-intensity agriculture",
            "Natural vegetation": "Natural vegetation",
            "Flow accumulation": "Flow accumulation",
            "Hydrological distance": "Hydrological distance",
        }
    )
    for row in soil.itertuples(index=False):
        rows.append(
            {
                "Medium": "Soil",
                "Response": row.Response,
                "Predictor": row.Predictor,
                "Estimate": row.Estimate,
                "Std_Beta": row.Std_Beta,
                "SE": row.SE,
                "CI_low": row.CI_low,
                "CI_high": row.CI_high,
                "p": row.p,
                "N": row.N_obs,
                "N_sites": row.N_sites,
                "Model": "Soil D2 Network-only OLS-HC3",
                "Plotted_in_FigureD": True,
                "Panel_note": "",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(FIG_D_DIR / "FigureD_multivariable_coefficients.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(FIG_D_DIR / "FigureD_multivariable_coefficients.csv", TABLE_DIR / "FigureD_multivariable_coefficients.csv")
    return out


def add_forest_panel(ax, sub: pd.DataFrame, color: str, title: str, grouped: bool = False) -> None:
    if sub.empty:
        ax.axis("off")
        return
    sub = sub.copy().reset_index(drop=True)
    entries = []
    if grouped:
        for response in ["TN", "TP", "TOC"]:
            group = sub[sub["Response"] == response]
            if group.empty:
                continue
            entries.append({"kind": "header", "label": response, "row": None})
            for _, row in group.iterrows():
                entries.append({"kind": "data", "label": "    " + row["Predictor"], "row": row})
            entries.append({"kind": "spacer", "label": "", "row": None})
        if entries and entries[-1]["kind"] == "spacer":
            entries.pop()
    else:
        for _, row in sub.iterrows():
            entries.append({"kind": "data", "label": row["Predictor"], "row": row})

    y_positions = np.arange(len(entries))[::-1]
    ax.axvline(0, color="0.25", linewidth=1)
    ticks = []
    labels = []
    for y, entry in zip(y_positions, entries):
        if entry["kind"] == "spacer":
            ax.axhline(y, color="0.9", linewidth=0.7, zorder=0)
            continue
        ticks.append(y)
        labels.append(entry["label"])
        if entry["kind"] == "data":
            row = entry["row"]
            estimate = row["Std_Beta"] if pd.notna(row["Std_Beta"]) else row["Estimate"]
            ax.errorbar(
                estimate,
                y,
                xerr=np.array([[estimate - row["CI_low"]], [row["CI_high"] - estimate]]),
                fmt="o",
                color=color,
                ecolor="0.35",
                elinewidth=1.25,
                capsize=3,
                markersize=5,
                zorder=3,
            )
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    for tick, entry in zip(ax.get_yticklabels(), [e for e in entries if e["kind"] != "spacer"]):
        if entry["kind"] == "header":
            tick.set_fontweight("bold")
    ax.set_ylim(-0.7, len(entries) - 0.3)
    ax.set_xlabel("Standardized coefficient (β)")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


def coefficient_xlim(*frames: pd.DataFrame) -> Tuple[float, float]:
    plotted = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    values = pd.concat([plotted["CI_low"], plotted["CI_high"]]).dropna()
    xmin = min(float(values.min()), 0.0)
    xmax = max(float(values.max()), 0.0)
    span = xmax - xmin
    margin = span * 0.12 if span else 0.1
    return xmin - margin, xmax + margin


def make_figure_d(table: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    water = table[(table["Medium"] == "Surface water") & (table["Plotted_in_FigureD"] == True)].copy()  # noqa: E712
    water_order = {"Hydrological distance": 0, "Dissolved oxygen": 1, "pH": 2}
    water["order"] = water["Predictor"].map(water_order)
    water = water.sort_values(["Response", "order"])
    soil = table[(table["Medium"] == "Soil") & (table["Plotted_in_FigureD"] == True)].copy()  # noqa: E712
    response_order = {"TN": 0, "TP": 1, "TOC": 2}
    soil_order = {
        "Hydrological distance": 0,
        "High-intensity agriculture": 1,
        "Natural vegetation": 2,
        "Flow accumulation": 3,
    }
    soil["response_order"] = soil["Response"].map(response_order)
    soil["order"] = soil["Predictor"].map(soil_order)
    soil = soil.sort_values(["response_order", "order"])

    shared_xlim = coefficient_xlim(water, soil)
    for suffix, shared in [("sharedX", True), ("freeX", False)]:
        fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.8), gridspec_kw={"height_ratios": [1.0, 2.65]}, constrained_layout=True)
        add_forest_panel(axes[0], water, COLORS["TN"], "(a) Surface-water TN")
        add_forest_panel(axes[1], soil, COLORS["Soil"], "(b) Soil nutrients", grouped=True)
        axes[0].set_xlabel("")
        if shared:
            axes[0].set_xlim(*shared_xlim)
            axes[1].set_xlim(*shared_xlim)
        fig.savefig(FIG_D_DIR / f"FigureD_water_soil_multivariable_{suffix}.png", dpi=600, bbox_inches="tight")
        fig.savefig(FIG_D_DIR / f"FigureD_water_soil_multivariable_{suffix}.pdf", bbox_inches="tight")
        if shared:
            fig.savefig(FIG_D_DIR / "FigureD_water_soil_multivariable.png", dpi=600, bbox_inches="tight")
            fig.savefig(FIG_D_DIR / "FigureD_water_soil_multivariable.pdf", bbox_inches="tight")
        plt.close(fig)


def write_figure_c_caption(df: pd.DataFrame, overall: pd.DataFrame) -> None:
    tp = df["TP_mg_L"].dropna()
    tn_p = overall.loc[overall["Pollutant"] == "TN", "p"].iloc[0]
    tp_p = overall.loc[overall["Pollutant"] == "TP", "p"].iloc[0]
    text = f"""# Figure C Caption

Seasonal variation in surface-water TN and TP concentrations across the balanced repeated-site dataset. Panels show (a) TN and (b) TP for Autumn 2024, Winter 2024, Spring 2025, and Summer 2025. Open circles are raw observations, faint grey lines connect repeated measurements from the same site, filled black circles indicate seasonal medians, and vertical black bars indicate interquartile ranges. Compact letters show Holm-adjusted paired Wilcoxon groupings after the Friedman test and are positioned from the upper quartile rather than the seasonal maximum. Friedman test results were TN, {p_text(tn_p)}, and TP, {p_text(tp_p)}. The linear version preserves the original concentration axis; the log-y version uses a logarithmic axis for the TP panel to improve readability while retaining all observations. TP minimum = {tp.min():.6g} mg L^-1; TP zero count = {int((tp == 0).sum())}.
"""
    (FIG_C_DIR / "FigureC_caption.md").write_text(text, encoding="utf-8")


def write_figure_d_caption() -> None:
    text = """# Figure D Caption

Multivariable coefficient plots for final water and soil models. Panel (a) shows continuous predictors retained in the final Surface-water TN model. TP was best represented by the seasonal model; therefore, no additional continuous predictors were included in the coefficient panel. Panel (b) shows Soil TN, TP, and TOC D2 Network-only models. Dots indicate standardized coefficients, horizontal bars indicate 95% confidence intervals, and the vertical line indicates zero effect. Response variables were log1p-transformed, and continuous predictors were standardized before model fitting. The shared-x version uses common x-axis limits for direct effect-size comparison; the free-x version allows each panel to use independent x-axis limits.
"""
    (FIG_D_DIR / "FigureD_caption.md").write_text(text, encoding="utf-8")


def scan_nload_inventory() -> Tuple[pd.DataFrame, bool]:
    exact_terms = [
        "n_load",
        "n_transport",
        "n_migration",
        "nitrogen_load",
        "nitrogen_transport",
        "氮负荷",
        "氮迁移",
        "氮输移",
    ]
    broad_terms = exact_terms + ["musle", "scs", "cn", "总氮", "氨氮", "硝酸氮", "tn", "nitrogen"]
    allowed_ext = {".tif", ".img", ".shp", ".csv", ".npy"}
    rows = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed_ext:
            continue
        rel = str(path.relative_to(ROOT))
        lower = rel.lower()
        exact = any(term.lower() in lower for term in exact_terms)
        broad = any(term.lower() in lower for term in broad_terms)
        if not broad:
            continue
        suitable = exact
        variable = "unknown"
        if "musle" in lower:
            variable = "MUSLE erosion/sediment support layer"
            suitable = False
        elif "总氮" in rel or "tn" in lower:
            variable = "TN concentration or TN display layer, not confirmed load"
            suitable = False
        elif "氨氮" in rel:
            variable = "soil ammonium nitrogen concentration, not load"
            suitable = False
        elif "硝酸氮" in rel:
            variable = "soil nitrate nitrogen concentration, not load"
            suitable = False
        elif exact:
            variable = "candidate N transport/load/migration output"
        rows.append(
            {
                "file": rel,
                "variable": variable,
                "unit": "unknown from filename",
                "resolution": "not inspected; no confirmed N-load raster available",
                "CRS": "not inspected",
                "data_source": "project file scan",
                "suitable_for_current_FigureE": suitable,
                "reason": (
                    "filename matches requested N-load/transport/migration terms"
                    if suitable
                    else "supporting or concentration layer only; not a confirmed N-load/transport output"
                ),
            }
        )
    inventory = pd.DataFrame(rows).sort_values(["suitable_for_current_FigureE", "file"], ascending=[False, True])
    inventory.to_csv(FIG_E_DIR / "FigureE_Nload_data_inventory.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(FIG_E_DIR / "FigureE_Nload_data_inventory.csv", TABLE_DIR / "FigureE_Nload_data_inventory.csv")
    found_real = bool(len(inventory) and inventory["suitable_for_current_FigureE"].any())
    return inventory, found_real


def write_figure_e_docs(inventory: pd.DataFrame, found_real: bool) -> None:
    md_rows = []
    for row in inventory.itertuples(index=False):
        md_rows.append(
            f"| `{row.file}` | {row.variable} | {row.unit} | {row.resolution} | {row.CRS} | {row.data_source} | {row.suitable_for_current_FigureE} | {row.reason} |"
        )
    if not md_rows:
        md_rows.append("| No matching project file | NA | NA | NA | NA | project file scan | False | No real N-load/transport/migration file found |")
    inventory_md = "\n".join(
        [
            "# Figure E N-load Data Inventory",
            "",
            "Search terms: N_load, N_transport, N_migration, nitrogen_load, nitrogen_transport, 氮负荷, 氮迁移, 氮输移.",
            "",
            "| file | variable | unit | resolution | CRS | data source | suitable | reason |",
            "|---|---|---|---|---|---|---:|---|",
            *md_rows,
            "",
            "Conclusion: "
            + (
                "At least one candidate file matched the requested N-load/transport/migration terms."
                if found_real
                else "No confirmed real N-load/transport/migration output was found. Figure E numerical analysis was stopped, and no fake N-load figure was generated."
            ),
            "",
        ]
    )
    (FIG_E_DIR / "FigureE_Nload_data_inventory.md").write_text(inventory_md, encoding="utf-8")

    required_inputs = """# Figure E Required Inputs

Figure E was not generated because no confirmed N transport/load/migration raster was found.

Required inputs for a real Figure E:

- A process-model output raster representing N transport intensity, N load, N loss, or N migration intensity.
- Clear unit for the raster values.
- Raster CRS, resolution, nodata value, and valid watershed mask.
- Reconstructed runoff pathway vector in the same or transformable CRS.
- Watershed boundary vector.
- Optional pond/ditch layers only for cartographic context.

Forbidden substitutions:

- Do not simulate N-load values.
- Do not use interpolated TN concentration as N-load.
- Do not use soil TN concentration alone as transport/load intensity.
- Do not use MUSLE alone as nitrogen load unless a documented N source x erosion/runoff model has already produced the raster.
"""
    (FIG_E_DIR / "FigureE_required_inputs.md").write_text(required_inputs, encoding="utf-8")

    template = '''from pathlib import Path

# Template only. Do not run until a real N transport/load raster is supplied.

N_TRANSPORT_RASTER = Path("path/to/real_N_transport_or_load_raster.tif")
RUNOFF_PATHWAYS = Path("path/to/reconstructed_runoff_pathways.shp")
WATERSHED_BOUNDARY = Path("path/to/watershed_boundary.shp")
OUT_DIR = PACKAGE / "FigureE_Ntransport"


def main():
    raise RuntimeError(
        "Figure E requires a real N transport/load/migration raster. "
        "Do not substitute TN concentration, soil N concentration, MUSLE alone, or simulated values."
    )


if __name__ == "__main__":
    main()
'''
    (FIG_E_DIR / "FigureE_plot_template.py").write_text(template, encoding="utf-8")
    shutil.copy2(FIG_E_DIR / "FigureE_plot_template.py", SCRIPT_DIR / "FigureE_plot_template.py")


def write_readme(found_real_nload: bool) -> None:
    readme = f"""# Manuscript Back-Half Figures Final Package

This package contains the remaining manuscript-quality figures requested after Figure A and Figure B.

Figure A and Figure B were already completed and were not redrawn.

## Figure C

Folder: `FigureC_seasonal/`

Purpose: answer whether surface-water TN and TP differed among the four seasonal sampling campaigns spanning autumn 2024 to summer 2025.

Data: balanced repeated-site cleaned TN/TP dataset, 12 sites observed in Autumn 2024, Winter 2024, Spring 2025 and Summer 2025.

Statistics: Friedman tests for overall seasonal response; paired Wilcoxon signed-rank tests with Holm correction for pairwise comparisons.

## Figure D

Folder: `FigureD_multivariable/`

Purpose: summarize final multivariable coefficients for surface-water and soil models without merging the two model systems.

Water TN: W3 MixedLM, `Season + NetworkDist + DO + pH + (1 | Site_ID)`.

Water TP: Phase 6 recommended GEE exchangeable season-only reporting model. Because no continuous predictors are retained, TP is annotated rather than plotted as fake continuous coefficients.
TP season-only status is documented in `FigureD_caption.md`, not inside the plotting area.

Soil: D2 Network-only OLS-HC3 models for TN, TP and TOC.

Primary candidate outputs:

- `FigureC_seasonal_water_response_linear.png/pdf`
- `FigureC_seasonal_water_response_logY.png/pdf`
- `FigureD_water_soil_multivariable_sharedX.png/pdf`
- `FigureD_water_soil_multivariable_freeX.png/pdf`
- `FigureC_caption.md`
- `FigureD_caption.md`

## Figure E

Folder: `FigureE_Ntransport/`

Purpose: process consistency between reconstructed runoff pathways and modeled N transport/load intensity.

Status: {'real N-load data found; inspect inventory before plotting' if found_real_nload else 'not generated because no confirmed real N-load/transport/migration raster was found'}.

No simulated or proxy N-load figure was generated.

## Tables

Folder: `tables/`

Contains plotting data and statistical tables used for Figures C/D and the Figure E data inventory.
"""
    (PACKAGE / "README.md").write_text(readme, encoding="utf-8")
    (README_DIR / "README.md").write_text(readme, encoding="utf-8")


def copy_script() -> None:
    shutil.copy2(Path(__file__), SCRIPT_DIR / "manuscript_backhalf_figures_final.py")


def main() -> None:
    ensure_dirs()
    figure_c_data = load_figure_c_data()
    figure_c_data.to_csv(FIG_C_DIR / "FigureC_plot_data.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(FIG_C_DIR / "FigureC_plot_data.csv", TABLE_DIR / "FigureC_plot_data.csv")
    _, desc, overall, _ = figure_c_stats(figure_c_data)
    make_figure_c(figure_c_data, desc, overall)
    write_figure_c_caption(figure_c_data, overall)

    figure_d_table = build_figure_d_table()
    make_figure_d(figure_d_table)
    write_figure_d_caption()

    inventory, found_real_nload = scan_nload_inventory()
    write_figure_e_docs(inventory, found_real_nload)
    write_readme(found_real_nload)
    copy_script()

    manifest = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file():
            manifest.append({"file": str(path.relative_to(PACKAGE)), "bytes": path.stat().st_size})
    pd.DataFrame(manifest).to_csv(PACKAGE / "package_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote manuscript figure package to: {PACKAGE}")
    print(f"Confirmed real N-load data found: {found_real_nload}")


if __name__ == "__main__":
    main()
