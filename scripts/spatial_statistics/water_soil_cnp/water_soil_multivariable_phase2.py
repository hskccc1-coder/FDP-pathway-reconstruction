from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent))
from water_soil_multivariable_phase1 import (  # noqa: E402
    ROOT,
    SOIL_POINTS_PATH,
    read_soil_data,
    read_water_data,
)


OUT_DIR = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_phase2"
DIAG_DIR = OUT_DIR / "diagnostics"
MODELS_DIR = OUT_DIR / "models"
SENS_DIR = OUT_DIR / "sensitivity"
SPATIAL_DIR = OUT_DIR / "spatial"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_DIR = OUT_DIR / "summary"
LOG_DIR = OUT_DIR / "experiment_log"

SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
SEASON_REF = "Spring"
ALPHA = 0.05


def ensure_dirs() -> None:
    for path in [DIAG_DIR, MODELS_DIR, SENS_DIR, SPATIAL_DIR, FIG_DIR, SUMMARY_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd


def build_design(
    df: pd.DataFrame,
    response: str,
    continuous_predictors: List[str],
    include_season: bool,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], pd.Series]:
    cols = [response] + continuous_predictors
    if include_season:
        cols.append("Season")
    model_df = df[cols + [c for c in ["Site_ID"] if c in df.columns]].copy()
    for col in [response] + continuous_predictors:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df.dropna(subset=cols).copy()
    model_df = model_df[model_df[response] >= 0].copy()
    model_df["log_response"] = np.log1p(model_df[response])

    terms = ["Intercept"]
    parts = [np.ones(len(model_df))]
    if include_season:
        season_text = model_df["Season"].astype(str)
        for season in SEASON_ORDER:
            if season == SEASON_REF:
                continue
            terms.append(f"Season_{season}_vs_{SEASON_REF}")
            parts.append((season_text == season).astype(float).to_numpy())
    for pred in continuous_predictors:
        term = f"z_{pred}"
        terms.append(term)
        model_df[term] = zscore(model_df[pred])
        parts.append(model_df[term].to_numpy())

    design_df = model_df.dropna(subset=[t for t in terms if t.startswith("z_")]).copy()
    if len(design_df) != len(model_df):
        return build_design(design_df, response, continuous_predictors, include_season)

    X = np.column_stack(parts) if parts else np.empty((len(model_df), 0))
    y = model_df["log_response"].to_numpy(dtype=float)
    groups = model_df["Site_ID"].astype(str) if "Site_ID" in model_df.columns else pd.Series(range(len(model_df)))
    return model_df, X, y, terms, groups


def normal_loglike_from_rss(y: np.ndarray, residuals: np.ndarray, k: int) -> Tuple[float, float, float, float]:
    n = len(y)
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k and pd.notna(r2) else np.nan
    sigma2 = rss / n if n > 0 else np.nan
    if n == 0 or sigma2 <= 0:
        return np.nan, np.nan, r2, adj_r2
    loglike = -0.5 * n * (math.log(2 * math.pi) + math.log(sigma2) + 1)
    aic = 2 * k - 2 * loglike
    bic = math.log(n) * k - 2 * loglike
    return aic, bic, r2, adj_r2


def ols_fit(
    df: pd.DataFrame,
    response: str,
    continuous_predictors: List[str],
    include_season: bool,
    robust: str,
    pollutant: str,
    medium: str,
    model_name: str,
    requested_model: str,
) -> Tuple[pd.DataFrame, Dict[str, object], Optional[pd.DataFrame]]:
    model_df, X, y, terms, groups = build_design(df, response, continuous_predictors, include_season)
    n, k = X.shape
    n_sites = int(pd.Series(groups).nunique())
    status = "ok"
    if n <= k + 1 or k == 0:
        status = "not_fitted_too_few_complete_cases"
        out = pd.DataFrame(
            [
                {
                    "Medium": medium,
                    "Pollutant": pollutant,
                    "Model": model_name,
                    "requested_model": requested_model,
                    "method": robust,
                    "term": "",
                    "estimate": np.nan,
                    "SE": np.nan,
                    "CI_low": np.nan,
                    "CI_high": np.nan,
                    "t_value": np.nan,
                    "P_value": np.nan,
                    "N": n,
                    "N_sites": n_sites,
                    "status": status,
                }
            ]
        )
        comp = {
            "Medium": medium,
            "Pollutant": pollutant,
            "Model": model_name,
            "N": n,
            "N_sites": n_sites,
            "AIC": np.nan,
            "BIC": np.nan,
            "R2": np.nan,
            "Adj_R2": np.nan,
            "method": robust,
            "status": status,
        }
        return out, comp, None

    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    fitted = X @ beta
    residuals = y - fitted
    xtx_inv = np.linalg.pinv(X.T @ X)

    if robust == "OLS with Site_ID cluster-robust SE":
        meat = np.zeros((k, k), dtype=float)
        group_values = pd.Series(groups).astype(str).unique()
        for group in group_values:
            idx = pd.Series(groups).astype(str).to_numpy() == group
            xg = X[idx, :]
            ug = residuals[idx][:, None]
            meat += xg.T @ ug @ ug.T @ xg
        g = len(group_values)
        correction = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else 1.0
        cov = correction * xtx_inv @ meat @ xtx_inv
        df_resid = max(g - 1, 1)
    elif robust == "OLS-HC3":
        hat = np.sum(X * (X @ xtx_inv), axis=1)
        scale = residuals**2 / np.maximum((1.0 - hat) ** 2, 1e-12)
        cov = xtx_inv @ (X.T @ (X * scale[:, None])) @ xtx_inv
        df_resid = max(n - k, 1)
    else:
        sigma2 = float(np.sum(residuals**2) / max(n - k, 1))
        cov = sigma2 * xtx_inv
        df_resid = max(n - k, 1)

    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t_values = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_values), df=df_resid)
    tcrit = stats.t.ppf(1 - ALPHA / 2, df=df_resid)
    aic, bic, r2, adj_r2 = normal_loglike_from_rss(y, residuals, k)

    rows = []
    for term, estimate, term_se, t_value, p_value in zip(terms, beta, se, t_values, p_values):
        rows.append(
            {
                "Medium": medium,
                "Pollutant": pollutant,
                "Model": model_name,
                "requested_model": requested_model,
                "method": robust,
                "term": term,
                "estimate": estimate,
                "SE": term_se,
                "CI_low": estimate - tcrit * term_se,
                "CI_high": estimate + tcrit * term_se,
                "t_value": t_value,
                "P_value": p_value,
                "N": n,
                "N_sites": n_sites,
                "status": status,
            }
        )
    comp = {
        "Medium": medium,
        "Pollutant": pollutant,
        "Model": model_name,
        "N": n,
        "N_sites": n_sites,
        "AIC": aic,
        "BIC": bic,
        "R2": r2,
        "Adj_R2": adj_r2,
        "method": robust,
        "status": status,
    }
    model_out = model_df.copy()
    model_out["fitted"] = fitted
    model_out["residual"] = residuals
    return pd.DataFrame(rows), comp, model_out


def not_run_model(medium: str, pollutant: str, model_name: str, requested_model: str, reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Medium": medium,
                "Pollutant": pollutant,
                "Model": model_name,
                "requested_model": requested_model,
                "method": "not run",
                "term": "",
                "estimate": np.nan,
                "SE": np.nan,
                "CI_low": np.nan,
                "CI_high": np.nan,
                "t_value": np.nan,
                "P_value": np.nan,
                "N": 0,
                "N_sites": 0,
                "status": reason,
            }
        ]
    )


def readable_predictor(term: str) -> str:
    mapping = {
        "z_NetworkDist_km": "Hydrological distance",
        "z_DO_mg_L": "DO",
        "z_pH": "pH",
        "z_log_MFD": "Flow accumulation",
        "Season_Summer_vs_Spring": "Season: summer vs spring",
        "Season_Autumn_vs_Spring": "Season: autumn vs spring",
        "Season_Winter_vs_Spring": "Season: winter vs spring",
    }
    return mapping.get(term, term.replace("z_", ""))


def direction(beta: float) -> str:
    if pd.isna(beta):
        return ""
    if beta > 0:
        return "Positive"
    if beta < 0:
        return "Negative"
    return "Zero"


def make_summary_rows(models: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for table in models:
        if table.empty:
            continue
        use = table[(table["term"] != "") & (table["term"] != "Intercept")].copy()
        for row in use.itertuples(index=False):
            rows.append(
                {
                    "Medium": row.Medium,
                    "Pollutant": row.Pollutant,
                    "Model": row.Model,
                    "Predictor": readable_predictor(row.term),
                    "Term": row.term,
                    "Std_Beta": row.estimate,
                    "CI_low": row.CI_low,
                    "CI_high": row.CI_high,
                    "P_value": row.P_value,
                    "Direction": direction(row.estimate),
                    "Significant": bool(pd.notna(row.P_value) and row.P_value < ALPHA),
                    "N": row.N,
                    "N_sites": row.N_sites,
                    "Method": row.method,
                    "Status": row.status,
                }
            )
    return pd.DataFrame(rows)


def morans_i_knn(coords: np.ndarray, residuals: np.ndarray, k: int = 8, permutations: int = 999, seed: int = 42) -> Dict[str, object]:
    n = len(residuals)
    if n <= k + 1:
        return {"Moran_I": np.nan, "P_value": np.nan, "N": n, "weight_definition": f"KNN k={k}", "status": "too few observations"}
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, :k]
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        W[i, neighbors[i]] = 1.0
    w_sum = W.sum()
    z = residuals - residuals.mean()
    denom = float(np.sum(z**2))
    if denom <= 0 or w_sum <= 0:
        return {"Moran_I": np.nan, "P_value": np.nan, "N": n, "weight_definition": f"KNN k={k}", "status": "constant residuals"}
    observed = (n / w_sum) * float(z @ W @ z) / denom

    rng = np.random.default_rng(seed)
    permuted = []
    for _ in range(permutations):
        zp = rng.permutation(z)
        permuted.append((n / w_sum) * float(zp @ W @ zp) / denom)
    permuted = np.asarray(permuted)
    p_value = (1 + np.sum(np.abs(permuted) >= abs(observed))) / (permutations + 1)
    return {
        "Moran_I": observed,
        "P_value": p_value,
        "N": n,
        "weight_definition": f"KNN k={k}, binary directed weights, {permutations} permutations",
        "status": "ok",
    }


def run_models() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    water, _ = read_water_data()
    soil = read_soil_data()
    water.to_csv(DIAG_DIR / "water_model_input.csv", index=False, encoding="utf-8-sig")
    soil.to_csv(DIAG_DIR / "soil_model_input.csv", index=False, encoding="utf-8-sig")

    model_tables: Dict[str, pd.DataFrame] = {}
    residual_tables: Dict[str, pd.DataFrame] = {}
    comparison_rows = []
    summary_model_tables = []

    water_specs = [
        ("W0", [], True, "log1p(Y) ~ Season + (1 | Site_ID)"),
        ("W1", ["NetworkDist_km"], True, "log1p(Y) ~ Season + z(NetworkDist_km) + (1 | Site_ID)"),
        ("W2", None, True, "log1p(Y) ~ Season + z(NetworkDist_km) + z(HighAg_ratio) + z(NaturalVeg_ratio) + (1 | Site_ID)"),
        ("W3", ["NetworkDist_km", "DO_mg_L", "pH"], True, "log1p(Y) ~ Season + z(NetworkDist_km) + z(DO) + z(pH) + (1 | Site_ID)"),
    ]
    for pollutant, response in [("TN", "TN_mg_L"), ("TP", "TP_mg_L")]:
        for model_name, predictors, include_season, requested in water_specs:
            file_path = MODELS_DIR / f"water_{pollutant}_{model_name}.csv"
            if predictors is None:
                table = not_run_model(
                    "Water",
                    pollutant,
                    model_name,
                    requested,
                    "not_run_missing_water_upstream_landuse_ratios",
                )
                table.to_csv(file_path, index=False, encoding="utf-8-sig")
                model_tables[f"water_{pollutant}_{model_name}"] = table
                comparison_rows.append(
                    {
                        "Medium": "Water",
                        "Pollutant": pollutant,
                        "Model": model_name,
                        "N": 0,
                        "N_sites": 0,
                        "AIC": np.nan,
                        "BIC": np.nan,
                        "R2": np.nan,
                        "Adj_R2": np.nan,
                        "method": "not run",
                        "status": "not_run_missing_water_upstream_landuse_ratios",
                    }
                )
                continue
            table, comp, residual_df = ols_fit(
                water,
                response,
                predictors,
                include_season,
                "OLS with Site_ID cluster-robust SE",
                pollutant,
                "Water",
                model_name,
                requested,
            )
            table.to_csv(file_path, index=False, encoding="utf-8-sig")
            model_tables[f"water_{pollutant}_{model_name}"] = table
            comparison_rows.append(comp)
            if residual_df is not None:
                residual_tables[f"water_{pollutant}_{model_name}"] = residual_df
            if model_name == "W3":
                summary_model_tables.append(table)

    soil_specs = [
        ("TN", "TN"),
        ("TP", "TP"),
        ("TOC", "TOC"),
    ]
    for pollutant, response in soil_specs:
        requested = "log1p(Y) ~ z(Dist_outlet) + z(HighAg_ratio) + z(NaturalVeg_ratio) + z(log_MFD)"
        available = "available-data fallback: log1p(Y) ~ z(log_MFD)"
        table, comp, residual_df = ols_fit(
            soil,
            response,
            ["log_MFD"],
            False,
            "OLS-HC3",
            pollutant,
            "Soil",
            "available_log_MFD",
            f"{requested}; {available}",
        )
        table.to_csv(MODELS_DIR / f"soil_{pollutant}_model.csv", index=False, encoding="utf-8-sig")
        model_tables[f"soil_{pollutant}_model"] = table
        comp["Model"] = "available_log_MFD"
        comp["status"] = str(comp["status"]) + "; requested_dist_landuse_fields_missing"
        comparison_rows.append(comp)
        if residual_df is not None:
            residual_tables[f"soil_{pollutant}_model"] = residual_df
        summary_model_tables.append(table)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(SUMMARY_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    summary = make_summary_rows(summary_model_tables)
    summary.to_csv(SUMMARY_DIR / "multivariable_summary.csv", index=False, encoding="utf-8-sig")
    return summary, comparison, model_tables, residual_tables


def run_sensitivity() -> None:
    pd.DataFrame(
        [
            {
                "Analysis": "soil_hotspot_sensitivity",
                "Status": "not_run",
                "Reason": "No predefined hotspot was supplied. Following the analysis instruction, no observation was automatically removed to chase significance.",
            }
        ]
    ).to_csv(SENS_DIR / "soil_hotspot_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "Analysis": "water_model_sensitivity",
                "Status": "not_run",
                "Reason": "No predefined water-site exclusion rule was supplied. Main results keep all complete cases for each specified model.",
            }
        ]
    ).to_csv(SENS_DIR / "water_model_sensitivity.csv", index=False, encoding="utf-8-sig")


def run_spatial(residual_tables: Dict[str, pd.DataFrame]) -> None:
    soil_gdf = gpd.read_file(SOIL_POINTS_PATH)
    try:
        projected = soil_gdf.to_crs(epsg=32650)
        coords_df = pd.DataFrame(
            {
                "Site_ID": soil_gdf["样点"].astype(str),
                "x": projected.geometry.x,
                "y": projected.geometry.y,
            }
        )
        coord_status = "EPSG:32650 projected coordinates"
    except Exception:
        coords_df = pd.DataFrame(
            {
                "Site_ID": soil_gdf["样点"].astype(str),
                "x": soil_gdf.geometry.x,
                "y": soil_gdf.geometry.y,
            }
        )
        coord_status = "fallback geographic coordinates"

    rows = []
    for key, table in residual_tables.items():
        if not key.startswith("soil_") or "residual" not in table.columns:
            continue
        residual = table[["Site_ID", "residual"]].copy()
        merged = residual.merge(coords_df, on="Site_ID", how="inner")
        result = morans_i_knn(merged[["x", "y"]].to_numpy(), merged["residual"].to_numpy(), k=8)
        result.update({"Model": key, "coordinate_basis": coord_status})
        rows.append(result)
    pd.DataFrame(rows).to_csv(SPATIAL_DIR / "soil_residual_moransI.csv", index=False, encoding="utf-8-sig")


def make_figure(summary: pd.DataFrame) -> None:
    plot_df = summary[
        ((summary["Medium"] == "Water") & (summary["Model"] == "W3"))
        | ((summary["Medium"] == "Soil") & (summary["Model"] == "available_log_MFD"))
    ].copy()
    plot_df = plot_df[plot_df["Term"].str.startswith("z_", na=False)].copy()
    if plot_df.empty:
        return

    medium_order = ["Water", "Soil"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=False)
    for ax, medium in zip(axes, medium_order):
        sub = plot_df[plot_df["Medium"] == medium].copy()
        sub["label"] = sub["Pollutant"] + " | " + sub["Predictor"]
        sub = sub.sort_values(["Pollutant", "Predictor"])
        ypos = np.arange(len(sub))
        ax.axvline(0, color="0.25", linewidth=1)
        ax.errorbar(
            sub["Std_Beta"],
            ypos,
            xerr=[sub["Std_Beta"] - sub["CI_low"], sub["CI_high"] - sub["Std_Beta"]],
            fmt="o",
            color="#1f77b4" if medium == "Water" else "#2ca02c",
            ecolor="0.35",
            capsize=3,
        )
        ax.set_yticks(ypos)
        ax.set_yticklabels(sub["label"])
        ax.set_title("(a) Surface water" if medium == "Water" else "(b) Soil")
        ax.set_xlabel("Coefficient estimate (log-response scale)")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.suptitle("Water and soil multivariable models: available predictors only", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_water_soil_multivariable.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "Figure_water_soil_multivariable.pdf", bbox_inches="tight")
    plt.close(fig)


def write_experiment_log(summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    water_rows = int(comparison[(comparison["Medium"] == "Water") & (comparison["Model"] == "W3")]["N"].max())
    soil_rows = int(comparison[comparison["Medium"] == "Soil"]["N"].max())
    significant = summary[summary["Significant"] == True].copy()  # noqa: E712
    sig_text = "None"
    if not significant.empty:
        sig_text = "\n".join(
            f"- {r.Medium} {r.Pollutant} {r.Predictor}: beta={r.Std_Beta:.4g}, "
            f"95% CI [{r.CI_low:.4g}, {r.CI_high:.4g}], p={r.P_value:.4g}"
            for r in significant.itertuples(index=False)
        )
    md = f"""---
exp_id: HY-B-260826-002
title: Water-soil CNP multivariable modelling phase 2
date: 2026-08-26
created_at: "{now}"
system: HY
device: B
type: statistical_analysis
status: completed_with_limitations
skill_basis:
  - nature-statistics
  - nature-experiment-log
inputs:
  - data/four_season_TN_TP/water_TN_TP_four_seasons_all_valid.csv
  - results/figure3_network_distance_stage1/water_sites_network_distance.csv
  - data/小流域数据 (1) (1)_csv/202410.csv
  - data/小流域数据 (1) (1)_csv/202412.csv
  - data/小流域数据 (1) (1)_csv/202504.csv
  - data/小流域数据 (1) (1)_csv/202506.csv
  - data/土壤采样点/土壤采样点.shp
outputs_root: results/water_soil_multivariable_phase2
---

# Water-Soil CNP Multivariable Experiment Log

## Scope

Continued after phase 1 diagnostics. This run fits only models defensible from available fields and records limitations explicitly.

## Skill Rules Applied

- Report independent unit and model sample size.
- Keep associations separate from causal/mechanistic claims.
- Report effect estimates, uncertainty intervals and exact p-values where available.
- Do not invent missing data fields, exclusion rules, land-use ratios or mixed-model results.
- Record outputs and anomalies in Markdown with structured frontmatter.

## Software / Method

- Python 3.7 runtime.
- `statsmodels` unavailable, so true mixed-effects models were not fitted.
- Water models used OLS on `log1p(response)` with Site_ID cluster-robust standard errors as a fallback for `(1 | Site_ID)`.
- Soil models used OLS-HC3 on `log1p(response)`.
- Soil residual spatial autocorrelation used Global Moran's I with KNN k=8 and 999 permutations, implemented locally.

## Models Run

Water:

- W0: `log1p(Y) ~ Season`
- W1: `log1p(Y) ~ Season + z(NetworkDist_km)`
- W2: not run because upstream land-use ratios were unavailable.
- W3: `log1p(Y) ~ Season + z(NetworkDist_km) + z(DO_mg_L) + z(pH)`

Soil:

- Requested full model could not be run because `Dist_outlet`, `HighAg_ratio`, `NaturalVeg_ratio`, and soil pH were missing.
- Available-data fallback: `log1p(Y) ~ z(log_MFD)` for TN, TP and TOC.

## Sample Size

- Water W3 complete cases: {water_rows} rows.
- Soil available-data models: {soil_rows} observations.

## Significant Terms In Fitted Available Models

{sig_text}

## Critical Limitations

- Water W2 land-use model remains blocked by missing upstream contributing-area land-use proportions.
- Soil requested main model remains blocked by missing `Dist_outlet`, `HighAg_ratio`, `NaturalVeg_ratio`, and soil pH.
- `statsmodels` is not installed, so water model outputs are fallback OLS cluster-robust estimates, not true random-intercept mixed models.
- No hotspot sensitivity exclusion was run because no predefined hotspot rule was supplied.

## Output Files

- `diagnostics/water_model_input.csv`
- `diagnostics/soil_model_input.csv`
- `models/water_TN_W0.csv` to `models/water_TP_W3.csv`
- `models/soil_TN_model.csv`, `models/soil_TP_model.csv`, `models/soil_TOC_model.csv`
- `summary/multivariable_summary.csv`
- `summary/model_comparison.csv`
- `spatial/soil_residual_moransI.csv`
- `sensitivity/soil_hotspot_sensitivity.csv`
- `sensitivity/water_model_sensitivity.csv`
- `figures/Figure_water_soil_multivariable.png`
- `figures/Figure_water_soil_multivariable.pdf`
"""
    (LOG_DIR / "HY-B-260826-002_water_soil_multivariable_phase2.md").write_text(md, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    summary, comparison, _, residual_tables = run_models()
    run_sensitivity()
    run_spatial(residual_tables)
    make_figure(summary)
    write_experiment_log(summary, comparison)
    manifest = []
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            manifest.append({"File": str(path.relative_to(OUT_DIR)), "Bytes": path.stat().st_size})
    pd.DataFrame(manifest).to_csv(OUT_DIR / "phase2_output_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote phase-2 outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
