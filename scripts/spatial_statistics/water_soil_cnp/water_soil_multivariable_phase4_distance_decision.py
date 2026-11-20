from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


ROOT = Path(__file__).resolve().parents[3]
PHASE4_DIR = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_phase4_soil_final"
OUT_DIR = PHASE4_DIR / "decision"
QA_DIR = PHASE4_DIR / "figures"

SENS_PATH = PHASE4_DIR / "summary" / "soil_distance_definition_sensitivity.csv"
COMPARE_PATH = PHASE4_DIR / "summary" / "soil_model_comparison.csv"
S6_PATH = PHASE4_DIR / "sensitivity" / "soil_S6_sensitivity.csv"
MORAN_PATH = PHASE4_DIR / "spatial" / "soil_fullmodel_residual_moransI.csv"
COV_PATH = PHASE4_DIR / "cleaned" / "soil_covariates_complete.csv"
DIST_PATH = PHASE4_DIR / "distance" / "soil_distance_diagnostics.csv"
SOIL_POINTS_PATH = ROOT / "data" / "soil_water_measurements" / "土壤采样点" / "土壤采样点.shp"
RIVER_PATH = ROOT / "data" / "soil_water_measurements" / "矫正后河流线" / "river.shp"
OUTLET_PATH = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "figure3_network_distance_stage1" / "outlet_snapped.shp"

MODEL_LABELS = {
    "D1_Euclidean": "D1 = Euclidean, N=47",
    "D2_NetworkOnly": "D2 = Network-only, N=40",
    "D3_Hybrid": "D3 = Hybrid, N=47",
}
DIST_TERMS = {
    "D1_Euclidean": "z_EuclideanDist_m",
    "D2_NetworkOnly": "z_NetworkDist_m",
    "D3_Hybrid": "z_HybridDist_m",
}
TERM_MAP = {
    "HighAg": "z_HighAg_ratio",
    "NaturalVeg": "z_NaturalVeg_ratio",
    "logMFD": "z_log_MFD",
}
POLLUTANTS = ["TN", "TP", "TOC"]
MODELS = ["D1_Euclidean", "D2_NetworkOnly", "D3_Hybrid"]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)


def read_vif_max(model: str) -> float:
    path = PHASE4_DIR / "diagnostics" / f"soil_predictor_VIF_{model}.csv"
    vif = pd.read_csv(path)
    return float(pd.to_numeric(vif["VIF"], errors="coerce").max())


def term_row(df: pd.DataFrame, pollutant: str, model: str, term: str) -> pd.Series:
    rows = df[(df["Pollutant"] == pollutant) & (df["Model"] == model) & (df["Term"] == term)]
    if rows.empty:
        raise ValueError(f"Missing term {term} for {pollutant} {model}")
    return rows.iloc[0]


def build_decision_table() -> pd.DataFrame:
    sens = pd.read_csv(SENS_PATH)
    rows = []
    for pollutant in POLLUTANTS:
        for model in MODELS:
            distance = term_row(sens, pollutant, model, DIST_TERMS[model])
            highag = term_row(sens, pollutant, model, TERM_MAP["HighAg"])
            natveg = term_row(sens, pollutant, model, TERM_MAP["NaturalVeg"])
            logmfd = term_row(sens, pollutant, model, TERM_MAP["logMFD"])
            rows.append(
                {
                    "Pollutant": pollutant,
                    "DistanceDefinition": MODEL_LABELS[model],
                    "Model": model,
                    "N": int(distance["N"]),
                    "Distance_Beta": distance["Std_Beta"],
                    "Distance_CI_low": distance["CI_low"],
                    "Distance_CI_high": distance["CI_high"],
                    "Distance_p": distance["P_value"],
                    "HighAg_Beta": highag["Std_Beta"],
                    "HighAg_p": highag["P_value"],
                    "NaturalVeg_Beta": natveg["Std_Beta"],
                    "NaturalVeg_p": natveg["P_value"],
                    "logMFD_Beta": logmfd["Std_Beta"],
                    "logMFD_p": logmfd["P_value"],
                    "R2": distance["R2"],
                    "Adjusted_R2": distance["Adjusted_R2"],
                    "AIC": distance["AIC"],
                    "Max_VIF": read_vif_max(model),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "soil_distance_model_decision_table.csv", index=False, encoding="utf-8-sig")
    return out


def sign_of(x: float) -> str:
    if pd.isna(x):
        return "NA"
    if x > 0:
        return "positive"
    if x < 0:
        return "negative"
    return "zero"


def yes_no_consistent(values: List[float]) -> Tuple[bool, str]:
    signs = [sign_of(v) for v in values if sign_of(v) != "NA" and sign_of(v) != "zero"]
    consistent = len(set(signs)) <= 1
    return consistent, ", ".join(signs)


def stability_by_predictor(table: pd.DataFrame, pollutant: str) -> Dict[str, Dict[str, object]]:
    subset = table[table["Pollutant"] == pollutant].copy()
    result = {}
    predictors = {
        "Distance": ("Distance_Beta", "Distance_p"),
        "HighAg": ("HighAg_Beta", "HighAg_p"),
        "NaturalVeg": ("NaturalVeg_Beta", "NaturalVeg_p"),
        "log_MFD": ("logMFD_Beta", "logMFD_p"),
    }
    for name, (beta_col, p_col) in predictors.items():
        betas = pd.to_numeric(subset[beta_col], errors="coerce").tolist()
        pvals = pd.to_numeric(subset[p_col], errors="coerce").tolist()
        sign_consistent, signs = yes_no_consistent(betas)
        sig_flags = [p < 0.05 for p in pvals if pd.notna(p)]
        sig_consistent = len(set(sig_flags)) <= 1
        beta_range = max(betas) - min(betas)
        result[name] = {
            "sign_consistent": sign_consistent,
            "signs": signs,
            "sig_consistent": sig_consistent,
            "significance_pattern": ", ".join(["p<0.05" if p < 0.05 else "ns" for p in pvals]),
            "beta_range": beta_range,
            "betas": betas,
            "pvals": pvals,
        }
    return result


def d2_change_text(table: pd.DataFrame, pollutant: str) -> str:
    subset = table[table["Pollutant"] == pollutant].set_index("Model")
    pieces = []
    for pred, col in [
        ("Distance", "Distance_Beta"),
        ("HighAg", "HighAg_Beta"),
        ("NaturalVeg", "NaturalVeg_Beta"),
        ("log_MFD", "logMFD_Beta"),
    ]:
        d1 = float(subset.loc["D1_Euclidean", col])
        d2 = float(subset.loc["D2_NetworkOnly", col])
        d3 = float(subset.loc["D3_Hybrid", col])
        same_direction = sign_of(d2) == sign_of(d1) == sign_of(d3)
        diff = abs(d2 - d3)
        if not same_direction:
            pieces.append(f"{pred} direction changed in D2")
        elif diff > 0.15:
            pieces.append(f"{pred} magnitude shifted in D2")
    return "; ".join(pieces) if pieces else "D2 retained the same coefficient directions with no large qualitative shift."


def write_stability_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    md_parts = [
        "# Distance Definition Stability Summary",
        "",
        "Model choice is summarized by definition consistency, hydrological meaning, sample retention, coefficient stability, diagnostics, and only lastly fit. No model is recommended only because it has the smallest p-value.",
    ]
    for pollutant in POLLUTANTS:
        stab = stability_by_predictor(table, pollutant)
        stable = [name for name, info in stab.items() if info["sign_consistent"] and info["sig_consistent"]]
        dependent = [name for name, info in stab.items() if not info["sign_consistent"] or not info["sig_consistent"]]
        md_parts.extend(
            [
                "",
                f"## {pollutant}",
                "",
                f"A. Distance coefficient direction consistent: {'Yes' if stab['Distance']['sign_consistent'] else 'No'} ({stab['Distance']['signs']}).",
                f"B. HighAg coefficient direction consistent: {'Yes' if stab['HighAg']['sign_consistent'] else 'No'} ({stab['HighAg']['signs']}).",
                f"C. NaturalVeg coefficient direction consistent: {'Yes' if stab['NaturalVeg']['sign_consistent'] else 'No'} ({stab['NaturalVeg']['signs']}).",
                f"D. log_MFD coefficient direction consistent: {'Yes' if stab['log_MFD']['sign_consistent'] else 'No'} ({stab['log_MFD']['signs']}).",
                f"E. Stable statistical conclusions (same direction and same significant/non-significant status): {', '.join(stable) if stable else 'none across all three definitions'}.",
                f"F. Conclusions dependent on distance definition: {', '.join(dependent) if dependent else 'none obvious'}.",
                f"G. D2 N=40 effect: {d2_change_text(table, pollutant)}",
            ]
        )
        rows.append(
            {
                "Pollutant": pollutant,
                "Distance_direction_consistent": stab["Distance"]["sign_consistent"],
                "HighAg_direction_consistent": stab["HighAg"]["sign_consistent"],
                "NaturalVeg_direction_consistent": stab["NaturalVeg"]["sign_consistent"],
                "logMFD_direction_consistent": stab["log_MFD"]["sign_consistent"],
                "Stable_predictors": "; ".join(stable),
                "Distance_definition_dependent_predictors": "; ".join(dependent),
                "D2_change_summary": d2_change_text(table, pollutant),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "soil_distance_stability_summary.csv", index=False, encoding="utf-8-sig")

    md_parts.extend(
        [
            "",
            "## Recommendation Principle",
            "",
            "D3 Hybrid retains all 47 sites but mixes network and Euclidean distance definitions. If the manuscript needs one unified primary distance definition and D1/D2/D3 conclusions are broadly consistent, a definitionally unified option should be preferred for the main text, with Hybrid retained as sensitivity. If D1 and D2 differ materially, the distance sensitivity itself should be reported rather than selecting the model with the smallest p-value.",
        ]
    )
    (OUT_DIR / "soil_distance_stability_summary.md").write_text("\n".join(md_parts), encoding="utf-8")
    return summary


def geometry_lines(geometry) -> List[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if hasattr(geometry, "geoms"):
        lines: List[LineString] = []
        for geom in geometry.geoms:
            lines.extend(geometry_lines(geom))
        return lines
    return []


def create_site10_qa() -> pd.DataFrame:
    cov = pd.read_csv(COV_PATH, dtype={"Site_ID": str})
    dist = pd.read_csv(DIST_PATH, dtype={"Site_ID": str})
    site = cov[cov["Site_ID"] == "10"].iloc[0]
    dist_row = dist[dist["Site_ID"] == "10"].iloc[0]
    out = pd.DataFrame(
        [
            {
                "Site_ID": "10",
                "Original_X": dist_row["Orig_X"],
                "Original_Y": dist_row["Orig_Y"],
                "Snapped_X": dist_row["Snap_X"],
                "Snapped_Y": dist_row["Snap_Y"],
                "nearest_network_segment": dist_row["nearest_segment_id"],
                "NetworkDist_m": dist_row["NetworkDist_m"],
                "EuclideanDist_m": dist_row["EuclideanDist_m"],
                "HybridDist_m": dist_row["HybridDist_m"],
                "TN": site["TN"],
                "TP": site["TP"],
                "TOC": site["TOC"],
                "HighAg_ratio": site["HighAg_ratio"],
                "NaturalVeg_ratio": site["NaturalVeg_ratio"],
                "log_MFD": site["log_MFD"],
            }
        ]
    )
    out.to_csv(OUT_DIR / "Site10_distance_QA.csv", index=False, encoding="utf-8-sig")

    river = gpd.read_file(RIVER_PATH)
    outlet = gpd.read_file(OUTLET_PATH).to_crs(river.crs)
    point = Point(float(dist_row["Orig_X"]), float(dist_row["Orig_Y"]))
    snapped = Point(float(dist_row["Snap_X"]), float(dist_row["Snap_Y"]))
    outlet_point = outlet.geometry.iloc[0]
    lines = []
    for geom in river.geometry:
        lines.extend(geometry_lines(geom))
    nearest_id = int(float(dist_row["nearest_segment_id"]))
    nearest_line = lines[nearest_id]
    buffer = point.buffer(260)
    nearby_lines = [line for line in lines if line.intersects(buffer)]

    fig, ax = plt.subplots(figsize=(7, 6))
    for line in nearby_lines:
        xs, ys = line.xy
        ax.plot(xs, ys, color="#8aa6b8", linewidth=1.1, alpha=0.8)
    xs, ys = nearest_line.xy
    ax.plot(xs, ys, color="#145a7a", linewidth=2.4, label="nearest drainage line")
    ax.plot([point.x, snapped.x], [point.y, snapped.y], color="#d55e00", linewidth=2, linestyle="--", label="snap connection")
    ax.scatter([point.x], [point.y], color="#d55e00", s=70, marker="o", label="original point", zorder=5)
    ax.scatter([snapped.x], [snapped.y], color="#0072b2", s=70, marker="x", label="snapped point", zorder=5)
    outlet_vec = np.array([outlet_point.x - snapped.x, outlet_point.y - snapped.y], dtype=float)
    outlet_len = np.linalg.norm(outlet_vec)
    if outlet_len > 0:
        outlet_unit = outlet_vec / outlet_len
        start = np.array([snapped.x, snapped.y]) + outlet_unit * 18
        end = np.array([snapped.x, snapped.y]) + outlet_unit * 105
        ax.annotate(
            "toward outlet",
            xy=(end[0], end[1]),
            xytext=(start[0] - 45, start[1] - 25),
            arrowprops=dict(arrowstyle="->", color="0.25", linewidth=1.5),
            color="0.25",
            fontsize=9,
        )
    ax.set_title("Site 10 distance QA")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(point.x - 240, point.x + 240)
    ax.set_ylim(point.y - 240, point.y + 240)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(QA_DIR / "Site10_distance_QA.png", dpi=300)
    plt.close(fig)
    return out


def coefficient_sentence(pollutant: str, rows: pd.DataFrame, term: str, label: str) -> str:
    row = rows[(rows["Pollutant"] == pollutant) & (rows["Term"] == term)].iloc[0]
    beta = float(row["Std_Beta"])
    p = float(row["P_value"])
    sig = "showed" if p < 0.05 else "did not show"
    direction = "positive" if beta > 0 else "negative"
    return f"{label} {sig} a {direction} association (beta = {beta:.3f}, 95% CI [{float(row['CI_low']):.3f}, {float(row['CI_high']):.3f}], p = {p:.3g})."


def write_interpretation() -> None:
    coeffs = pd.read_csv(PHASE4_DIR / "summary" / "soil_fullmodel_coefficients.csv")
    terms = [
        ("z_HybridDist_m", "Hydrological position"),
        ("z_HighAg_ratio", "High-intensity agriculture"),
        ("z_NaturalVeg_ratio", "Natural vegetation"),
        ("z_log_MFD", "Flow accumulation"),
    ]
    parts = ["# Phase 4 Soil Result Interpretation", ""]
    for pollutant in POLLUTANTS:
        parts.append(f"## {pollutant}")
        sentences = [coefficient_sentence(pollutant, coeffs, term, label) for term, label in terms]
        if pollutant == "TN":
            sentences.append("Overall, the available full-site soil model suggests a clearer association with hydrological position than with the land-use and MFD terms in this specification.")
        elif pollutant == "TP":
            sentences.append("Overall, soil TP showed its clearest positive association with high-intensity agricultural land-use context in this specification.")
        else:
            sentences.append("Overall, TOC associations were weaker and less statistically supported than those observed for TN distance or TP high-intensity agriculture.")
        parts.append(" ".join(sentences))
        parts.append("")
    (OUT_DIR / "phase4_soil_result_interpretation.md").write_text("\n".join(parts), encoding="utf-8")


def summarize_s6() -> pd.DataFrame:
    s6 = pd.read_csv(S6_PATH)
    terms = ["z_HybridDist_m", "z_HighAg_ratio", "z_NaturalVeg_ratio", "z_log_MFD"]
    rows = []
    md = ["# S6 Sensitivity Summary", "", "Main results continue to use all 47 soil samples. S6 is a predefined hotspot check only."]
    for pollutant in POLLUTANTS:
        full = s6[(s6["Pollutant"] == pollutant) & (s6["Model"] == "Full_N47") & (s6["Term"].isin(terms))].set_index("Term")
        excl = s6[(s6["Pollutant"] == pollutant) & (s6["Model"] == "Exclude_S6") & (s6["Term"].isin(terms))].set_index("Term")
        changes = []
        direction_stable = True
        for term in terms:
            b_full = float(full.loc[term, "Std_Beta"])
            b_excl = float(excl.loc[term, "Std_Beta"])
            p_full = float(full.loc[term, "P_value"])
            p_excl = float(excl.loc[term, "P_value"])
            if sign_of(b_full) != sign_of(b_excl):
                direction_stable = False
            delta = b_excl - b_full
            if abs(delta) >= 0.10 or ((p_full < 0.05) != (p_excl < 0.05)):
                changes.append(f"{term}: beta {b_full:.3f} -> {b_excl:.3f}, p {p_full:.3g} -> {p_excl:.3g}")
        rows.append(
            {
                "Pollutant": pollutant,
                "S6_affects_result": bool(changes),
                "Major_beta_or_significance_changes": "; ".join(changes) if changes else "none obvious",
                "Main_coefficient_directions_stable": direction_stable,
            }
        )
        md.extend(
            [
                "",
                f"## {pollutant}",
                f"- S6 affects result: {'yes' if changes else 'no major qualitative change'}",
                f"- Changed beta/significance: {'; '.join(changes) if changes else 'none obvious'}",
                f"- Main coefficient directions stable: {'yes' if direction_stable else 'no'}",
            ]
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "soil_S6_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "soil_S6_sensitivity_summary.md").write_text("\n".join(md), encoding="utf-8")
    return out


def summarize_moran() -> pd.DataFrame:
    moran = pd.read_csv(MORAN_PATH)
    rows = []
    parts = ["# Full-Model Residual Moran's I", ""]
    for _, row in moran.iterrows():
        p = float(row["Permutation_p"])
        sentence = "no significant residual global spatial autocorrelation was detected under the specified KNN spatial weights" if p > 0.05 else "significant residual global spatial autocorrelation remained under the specified KNN spatial weights"
        rows.append(
            {
                "Pollutant": row["Pollutant"],
                "Moran_I": row["Moran_I"],
                "Permutation_p": row["Permutation_p"],
                "Interpretation": sentence,
            }
        )
        parts.append(f"- {row['Pollutant']}: Moran's I = {float(row['Moran_I']):.4f}, p = {p:.3g}; {sentence}.")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "soil_fullmodel_moransI_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "soil_fullmodel_moransI_summary.md").write_text("\n".join(parts), encoding="utf-8")
    return out


def update_record() -> None:
    record = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_experiment_record.md"
    if not record.exists():
        return
    marker = "\n\n## 15. Phase 4 Distance Definition Decision Outputs\n\n"
    text = (
        marker
        + "Output root: `results/water_soil_multivariable_phase4_soil_final/decision`\n\n"
        + "- Created `soil_distance_model_decision_table.csv` summarizing D1/D2/D3 for TN, TP and TOC.\n"
        + "- Created pollutant-specific stability summaries without selecting a model by p-value.\n"
        + "- Created `Site10_distance_QA.csv` and `figures/Site10_distance_QA.png` for the large snap-distance site.\n"
        + "- Summarized S6 sensitivity and full-model residual Moran's I.\n"
        + "- Water Phase 5 was not started.\n"
    )
    current = record.read_text(encoding="utf-8")
    if marker in current:
        current = current.split(marker)[0].rstrip() + text
    else:
        current = current.rstrip() + text
    record.write_text(current, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    decision = build_decision_table()
    write_stability_summary(decision)
    create_site10_qa()
    write_interpretation()
    summarize_s6()
    summarize_moran()
    update_record()
    manifest = []
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            manifest.append({"File": str(path.relative_to(OUT_DIR)), "Bytes": path.stat().st_size})
    pd.DataFrame(manifest).to_csv(OUT_DIR / "decision_output_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote distance decision outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
