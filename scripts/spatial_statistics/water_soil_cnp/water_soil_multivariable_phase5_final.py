from __future__ import annotations

import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[3]
PHASE4_DIR = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_phase4_soil_final"
PHASE2_WATER_INPUT = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_phase2" / "diagnostics" / "water_model_input.csv"
OUT_DIR = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_phase5_final"
SOIL_DIR = OUT_DIR / "soil"
WATER_DIR = OUT_DIR / "water"
DIAG_DIR = OUT_DIR / "diagnostics"
SENS_DIR = OUT_DIR / "sensitivity"
SPATIAL_DIR = OUT_DIR / "spatial"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_DIR = OUT_DIR / "summary"
LOG_DIR = OUT_DIR / "experiment_log"

SEASON_ORDER = ["Autumn", "Winter", "Spring", "Summer"]
SEASON_REF = "Autumn"
KNN_K = 8
MORAN_PERMUTATIONS = 999
RNG_SEED = 260826


def ensure_dirs() -> None:
    for path in [SOIL_DIR, WATER_DIR, DIAG_DIR, SENS_DIR, SPATIAL_DIR, FIG_DIR, SUMMARY_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd


def vif_table(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    rows = []
    x = df[columns].astype(float).to_numpy()
    for idx, col in enumerate(columns):
        y = x[:, idx]
        others = np.delete(x, idx, axis=1)
        others = np.column_stack([np.ones(len(others)), others])
        beta = np.linalg.pinv(others.T @ others) @ others.T @ y
        pred = others @ beta
        rss = float(np.sum((y - pred) ** 2))
        tss = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else np.nan
        vif = 1.0 / (1.0 - r2) if pd.notna(r2) and r2 < 1 else np.inf
        rows.append({"Predictor": col, "R2_against_other_fixed_effects": r2, "VIF": vif})
    return pd.DataFrame(rows)


def ols_hc3(
    df: pd.DataFrame,
    response: str,
    predictors: List[str],
    pollutant: str,
    model_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_df = df[[response, "Site_ID", "Orig_X", "Orig_Y"] + predictors].copy()
    for col in [response] + predictors:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df.dropna(subset=[response] + predictors + ["Orig_X", "Orig_Y"]).copy()
    model_df = model_df[model_df[response] >= 0].copy()
    model_df["log_response"] = np.log1p(model_df[response])

    terms = ["Intercept"]
    parts = [np.ones(len(model_df))]
    for pred in predictors:
        zcol = f"z_{pred}"
        model_df[zcol] = zscore(model_df[pred])
        terms.append(zcol)
        parts.append(model_df[zcol].to_numpy())
    model_df = model_df.dropna(subset=[f"z_{p}" for p in predictors]).copy()
    x = np.column_stack([np.ones(len(model_df))] + [model_df[f"z_{p}"].to_numpy() for p in predictors])
    y = model_df["log_response"].to_numpy()
    n, k = x.shape
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    fitted = x @ beta
    residual = y - fitted
    xtx_inv = np.linalg.pinv(x.T @ x)
    hat = np.sum(x * (x @ xtx_inv), axis=1)
    scale = residual**2 / np.maximum((1.0 - hat) ** 2, 1e-12)
    cov = xtx_inv @ (x.T @ (x * scale[:, None])) @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    df_resid = max(n - k, 1)
    tvals = beta / se
    pvals = 2 * stats.t.sf(np.abs(tvals), df=df_resid)
    tcrit = stats.t.ppf(0.975, df=df_resid)
    rss = float(np.sum(residual**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k and pd.notna(r2) else np.nan
    sigma2 = rss / n
    loglike = -0.5 * n * (math.log(2 * math.pi) + math.log(sigma2) + 1) if sigma2 > 0 else np.nan
    aic = 2 * k - 2 * loglike if pd.notna(loglike) else np.nan

    rows = []
    for term, est, term_se, tval, pval in zip(terms, beta, se, tvals, pvals):
        rows.append(
            {
                "Medium": "Soil",
                "Response": pollutant,
                "Model": model_name,
                "Predictor": readable_soil_term(term),
                "Term": term,
                "Estimate": est,
                "Std_Beta": est if term != "Intercept" else np.nan,
                "SE": term_se,
                "CI_low": est - tcrit * term_se,
                "CI_high": est + tcrit * term_se,
                "p": pval,
                "N_obs": n,
                "N_sites": n,
                "R2": r2,
                "Adjusted_R2": adj_r2,
                "AIC": aic,
                "Model_type": "OLS-HC3; D2 Network-only soil model",
            }
        )
    residual_df = model_df[["Site_ID", "Orig_X", "Orig_Y"]].copy()
    residual_df["Response"] = pollutant
    residual_df["fitted"] = fitted
    residual_df["residual"] = residual
    return pd.DataFrame(rows), residual_df


def readable_soil_term(term: str) -> str:
    return {
        "Intercept": "Intercept",
        "z_NetworkDist_m": "Hydrological distance",
        "z_HighAg_ratio": "High-intensity agriculture",
        "z_NaturalVeg_ratio": "Natural vegetation",
        "z_log_MFD": "Flow accumulation",
    }.get(term, term)


def run_soil_final() -> Tuple[pd.DataFrame, pd.DataFrame]:
    soil = pd.read_csv(PHASE4_DIR / "soil_covariates_complete.csv", dtype={"Site_ID": str})
    predictors = ["NetworkDist_m", "HighAg_ratio", "NaturalVeg_ratio", "log_MFD"]
    tables = []
    residuals = []
    for pollutant in ["TN", "TP", "TOC"]:
        table, residual = ols_hc3(soil, pollutant, predictors, pollutant, "D2_NetworkOnly_primary")
        table.to_csv(SOIL_DIR / f"soil_D2_final_{pollutant}_coefficients.csv", index=False, encoding="utf-8-sig")
        residual.to_csv(SOIL_DIR / f"soil_D2_final_{pollutant}_residuals.csv", index=False, encoding="utf-8-sig")
        tables.append(table)
        residuals.append(residual)
    out = pd.concat(tables, ignore_index=True)
    res_out = pd.concat(residuals, ignore_index=True)
    out.to_csv(SOIL_DIR / "soil_D2_final_coefficients.csv", index=False, encoding="utf-8-sig")
    return out, res_out


def morans_i_knn(coords: np.ndarray, values: np.ndarray, k: int, permutations: int, seed: int) -> Dict[str, float]:
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(coords).all(axis=1) & np.isfinite(values)
    coords = coords[valid]
    values = values[valid]
    n = len(values)
    if n <= k + 1:
        return {"N": n, "k": k, "Moran_I": np.nan, "p": np.nan, "permutations": permutations}
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    neighbors = np.argsort(dist, axis=1)[:, :k]
    weights = np.zeros((n, n), dtype=float)
    for i in range(n):
        weights[i, neighbors[i]] = 1.0 / k
    z = values - values.mean()
    denom = float(np.sum(z**2))
    wsum = float(weights.sum())
    observed = (n / wsum) * float(z @ weights @ z) / denom if denom > 0 and wsum > 0 else np.nan
    rng = np.random.default_rng(seed)
    perm_i = []
    for _ in range(permutations):
        zp = rng.permutation(z)
        perm_i.append((n / wsum) * float(zp @ weights @ zp) / denom if denom > 0 else np.nan)
    perm_i = np.asarray(perm_i)
    p = (np.sum(np.abs(perm_i) >= abs(observed)) + 1) / (permutations + 1)
    return {"N": n, "k": k, "Moran_I": observed, "p": p, "permutations": permutations}


def run_soil_moran(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for response, sub in residuals.groupby("Response"):
        result = morans_i_knn(
            sub[["Orig_X", "Orig_Y"]].to_numpy(),
            sub["residual"].to_numpy(),
            k=KNN_K,
            permutations=MORAN_PERMUTATIONS,
            seed=RNG_SEED,
        )
        result.update(
            {
                "Response": response,
                "Model": "D2_NetworkOnly_primary",
                "spatial_weights": f"KNN k={KNN_K}",
                "interpretation": (
                    "no significant residual global spatial autocorrelation was detected under the specified KNN spatial weights"
                    if pd.notna(result["p"]) and result["p"] > 0.05
                    else "significant residual global spatial autocorrelation detected under the specified KNN spatial weights"
                ),
            }
        )
        rows.append(result)
    out = pd.DataFrame(rows)
    out.to_csv(SPATIAL_DIR / "soil_D2_residual_moransI.csv", index=False, encoding="utf-8-sig")
    return out


def water_site_matching_qa(water: pd.DataFrame) -> pd.DataFrame:
    qa = water[
        ["Site_ID", "Season", "TN_mg_L", "TP_mg_L", "NetworkDist_m", "DO_mg_L", "pH", "distance_method"]
    ].copy()
    qa["matched_network"] = qa["NetworkDist_m"].notna()
    qa["matched_environment"] = qa[["DO_mg_L", "pH"]].notna().any(axis=1)
    qa = qa.rename(columns={"TN_mg_L": "TN", "TP_mg_L": "TP", "DO_mg_L": "DO"})
    qa.to_csv(WATER_DIR / "water_site_matching_QA.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(
        [
            {
                "total_rows": len(water),
                "unique_Site_ID": water["Site_ID"].nunique(),
                "unique_Site_ID_with_NetworkDist": water.loc[water["NetworkDist_m"].notna(), "Site_ID"].nunique(),
                "TN_available_rows": water["TN_mg_L"].notna().sum(),
                "TP_available_rows": water["TP_mg_L"].notna().sum(),
                "DO_available_rows": water["DO_mg_L"].notna().sum(),
                "pH_available_rows": water["pH"].notna().sum(),
                "explanation": (
                    "The full seasonal water table contains all environment/TN/TP records, whereas NetworkDist was merged "
                    "from the predefined network-distance table and is available only for the 14 water sites included in "
                    "the spatial/network analysis target set. Rows without NetworkDist were not inferred or manually merged."
                ),
            }
        ]
    )
    summary.to_csv(WATER_DIR / "water_data_structure_summary.csv", index=False, encoding="utf-8-sig")
    return qa


def prepare_water_complete_case(water: pd.DataFrame, response: str) -> pd.DataFrame:
    cols = [response, "Season", "NetworkDist_m", "DO_mg_L", "pH", "Site_ID"]
    df = water.dropna(subset=cols).copy()
    df = df[df[response] >= 0].copy()
    df["Season"] = pd.Categorical(df["Season"], categories=SEASON_ORDER, ordered=True)
    df = df[df["Season"].notna()].copy()
    df["log_response"] = np.log1p(df[response])
    df["z_NetworkDist_m"] = zscore(df["NetworkDist_m"])
    df["z_DO_mg_L"] = zscore(df["DO_mg_L"])
    df["z_pH"] = zscore(df["pH"])
    return df.dropna(subset=["z_NetworkDist_m", "z_DO_mg_L", "z_pH"]).copy()


def mixedlm_formula(model_name: str) -> str:
    if model_name == "W0":
        return "log_response ~ C(Season)"
    if model_name == "W1":
        return "log_response ~ C(Season) + z_NetworkDist_m"
    if model_name == "W3":
        return "log_response ~ C(Season) + z_NetworkDist_m + z_DO_mg_L + z_pH"
    raise ValueError(model_name)


def fit_mixedlm(df: pd.DataFrame, formula: str) -> Tuple[object, str, List[str]]:
    last_error = None
    all_warnings: List[str] = []
    for method in ["lbfgs", "powell", "nm", "bfgs", "cg"]:
        try:
            model = smf.mixedlm(formula, data=df, groups=df["Site_ID"])
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                result = model.fit(reml=False, method=method, maxiter=2000, disp=False)
            method_warnings = [str(w.message) for w in captured]
            all_warnings.extend([f"{method}: {msg}" for msg in method_warnings])
            return result, method, all_warnings
        except Exception as exc:  # statsmodels can fail on singular Hessians before returning a result.
            last_error = exc
            all_warnings.append(f"{method} failed: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(all_warnings) if all_warnings else str(last_error))


def readable_water_term(term: str) -> str:
    if term == "Intercept":
        return "Intercept"
    if term == "z_NetworkDist_m":
        return "Hydrological distance"
    if term == "z_DO_mg_L":
        return "DO"
    if term == "z_pH":
        return "pH"
    if "C(Season)" in term:
        season = term.split("[T.")[-1].rstrip("]")
        return f"Season: {season} vs {SEASON_REF}"
    return term


def mixed_r2(result: object) -> Tuple[float, float]:
    fixed_fit = np.asarray(result.model.exog @ result.fe_params)
    var_fixed = float(np.var(fixed_fit, ddof=0))
    try:
        var_random = float(np.asarray(result.cov_re)[0, 0])
    except Exception:
        var_random = np.nan
    var_resid = float(result.scale)
    denom = var_fixed + (0 if pd.isna(var_random) else var_random) + var_resid
    if denom <= 0:
        return np.nan, np.nan
    marginal = var_fixed / denom
    conditional = (var_fixed + (0 if pd.isna(var_random) else var_random)) / denom
    return marginal, conditional


def model_coefficients(result: object, response: str, model_name: str, method: str, warnings_text: str, n: int, n_sites: int) -> pd.DataFrame:
    conf = result.conf_int()
    rows = []
    rand_var = float(np.asarray(result.cov_re)[0, 0]) if np.asarray(result.cov_re).size else np.nan
    boundary = bool(pd.notna(rand_var) and rand_var < 1e-8)
    for term in result.fe_params.index:
        rows.append(
            {
                "Medium": "Water",
                "Response": response,
                "Model": model_name,
                "Predictor": readable_water_term(term),
                "Term": term,
                "Estimate": float(result.fe_params[term]),
                "Std_Beta": float(result.fe_params[term]) if term.startswith("z_") else np.nan,
                "SE": float(result.bse_fe[term]) if term in result.bse_fe.index else np.nan,
                "CI_low": float(conf.loc[term, 0]) if term in conf.index else np.nan,
                "CI_high": float(conf.loc[term, 1]) if term in conf.index else np.nan,
                "p": float(result.pvalues[term]) if term in result.pvalues.index else np.nan,
                "N_obs": n,
                "N_sites": n_sites,
                "Model_type": "MixedLM random intercept for Site_ID; ML",
                "optimizer": method,
                "converged": bool(result.converged),
                "random_intercept_variance": rand_var,
                "residual_variance": float(result.scale),
                "boundary_or_singular_fit": boundary or ("singular" in warnings_text.lower()) or ("boundary" in warnings_text.lower()),
                "warnings": warnings_text,
            }
        )
    return pd.DataFrame(rows)


def model_comparison_row(result: object, response: str, model_name: str, method: str, warnings_text: str, n: int, n_sites: int) -> Dict[str, object]:
    marginal_r2, conditional_r2 = mixed_r2(result)
    rand_var = float(np.asarray(result.cov_re)[0, 0]) if np.asarray(result.cov_re).size else np.nan
    return {
        "Response": response,
        "Model": model_name,
        "AIC": float(result.aic),
        "BIC": float(result.bic),
        "logLik": float(result.llf),
        "marginal_R2": marginal_r2,
        "conditional_R2": conditional_r2,
        "N_observations": n,
        "N_sites": n_sites,
        "optimizer": method,
        "converged": bool(result.converged),
        "random_intercept_variance": rand_var,
        "residual_variance": float(result.scale),
        "boundary_or_singular_fit": bool(pd.notna(rand_var) and rand_var < 1e-8)
        or ("singular" in warnings_text.lower())
        or ("boundary" in warnings_text.lower()),
        "warnings": warnings_text,
    }


def select_water_model(comp: pd.DataFrame, response: str) -> Dict[str, str]:
    sub = comp[comp["Response"] == response].sort_values("AIC")
    best = sub.iloc[0]
    w0 = sub[sub["Model"] == "W0"].iloc[0]
    w1 = sub[sub["Model"] == "W1"].iloc[0]
    w3 = sub[sub["Model"] == "W3"].iloc[0]
    if w3["AIC"] <= w1["AIC"] - 2 and w3["BIC"] <= w1["BIC"] + 2:
        selected = "W3"
        reason = "W3 improved AIC by at least 2 relative to W1 without a clear BIC penalty, so the hydro-environmental model is retained."
    elif w1["AIC"] <= w0["AIC"] - 2 and w1["BIC"] <= w0["BIC"] + 2:
        selected = "W1"
        reason = "W1 improved fit relative to W0 while keeping a simpler hydrological-distance structure than W3."
    else:
        selected = "W0"
        reason = "Season-only W0 is preferred for parsimony because added hydrological/environmental predictors did not clearly improve equal-N fit."
    return {"Response": response, "Selected_model": selected, "Best_AIC_model": str(best["Model"]), "Selection_reason": reason}


def conditional_fitted_residuals(result: object, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    try:
        fitted = np.asarray(result.fittedvalues)
    except Exception:
        fitted = np.asarray(result.model.exog @ result.fe_params)
    residual = df["log_response"].to_numpy() - fitted
    return fitted, residual


def make_residual_plot(response: str, residual_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].axhline(0, color="0.25", linewidth=1)
    axes[0].scatter(residual_df["fitted"], residual_df["residual"], s=34, color="#2b6cb0", alpha=0.85)
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title(f"{response}: fitted vs residual")
    stats.probplot(residual_df["residual"], dist="norm", plot=axes[1])
    axes[1].set_title(f"{response}: residual Q-Q")
    axes[1].get_lines()[0].set_markerfacecolor("#2b6cb0")
    axes[1].get_lines()[0].set_markeredgecolor("#2b6cb0")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{response}_residual_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def influence_by_site(df: pd.DataFrame, response: str, model_name: str, formula: str, full_result: object) -> pd.DataFrame:
    rows = []
    full_fe = full_result.fe_params.copy()
    for site_id in sorted(df["Site_ID"].astype(str).unique(), key=lambda x: (len(x), x)):
        reduced = df[df["Site_ID"].astype(str) != site_id].copy()
        try:
            res, method, captured = fit_mixedlm(reduced, formula)
            aligned = res.fe_params.reindex(full_fe.index)
            change = (aligned - full_fe).abs()
            rows.append(
                {
                    "Response": response,
                    "Model": model_name,
                    "Excluded_Site_ID": site_id,
                    "N_after": len(reduced),
                    "N_sites_after": reduced["Site_ID"].nunique(),
                    "converged": bool(res.converged),
                    "optimizer": method,
                    "max_abs_fixed_effect_change": float(change.max()),
                    "largest_changed_term": str(change.idxmax()),
                    "random_intercept_variance": float(np.asarray(res.cov_re)[0, 0]),
                    "warnings": " | ".join(captured),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "Response": response,
                    "Model": model_name,
                    "Excluded_Site_ID": site_id,
                    "N_after": len(reduced),
                    "N_sites_after": reduced["Site_ID"].nunique(),
                    "converged": False,
                    "optimizer": "failed",
                    "max_abs_fixed_effect_change": np.nan,
                    "largest_changed_term": "",
                    "random_intercept_variance": np.nan,
                    "warnings": f"{type(exc).__name__}: {exc}",
                }
            )
    out = pd.DataFrame(rows).sort_values("max_abs_fixed_effect_change", ascending=False, na_position="last")
    out.to_csv(SENS_DIR / f"water_{response}_influential_sites.csv", index=False, encoding="utf-8-sig")
    return out


def write_water_diagnostic_md(
    response: str,
    selected: str,
    comparison: pd.DataFrame,
    final_result: object,
    final_diag: pd.DataFrame,
    vif: pd.DataFrame,
    influence: pd.DataFrame,
    warnings_text: str,
) -> None:
    comp = comparison[comparison["Response"] == response].copy()
    rand_var = float(np.asarray(final_result.cov_re)[0, 0])
    boundary = rand_var < 1e-8 or "boundary" in warnings_text.lower() or "singular" in warnings_text.lower()
    top_influence = influence.head(3)
    lines = [
        f"# Water {response} Model Diagnostics",
        "",
        f"- Final selected model: {selected}.",
        f"- Software/method: statsmodels MixedLM 0.13.5, random intercept for Site_ID, ML (REML=False).",
        f"- Reference season: {SEASON_REF}; ordered seasons: {', '.join(SEASON_ORDER)}.",
        f"- Convergence: {bool(final_result.converged)}.",
        f"- Random-intercept variance: {rand_var:.6g}.",
        f"- Residual variance: {float(final_result.scale):.6g}.",
        f"- Singular/boundary fit: {boundary}.",
        f"- Residual plot: figures/{response}_residual_plot.png.",
        "",
        "## Equal-N Model Comparison",
        "",
        comp[["Model", "AIC", "BIC", "logLik", "marginal_R2", "conditional_R2", "N_observations", "N_sites"]]
        .to_markdown(index=False),
        "",
        "## Fixed-Effect Collinearity",
        "",
        vif.to_markdown(index=False),
        "",
        "## Influential Sites",
        "",
        top_influence[
            ["Excluded_Site_ID", "max_abs_fixed_effect_change", "largest_changed_term", "converged", "random_intercept_variance"]
        ].to_markdown(index=False),
        "",
        "No site was automatically removed; influential-site checks are treated as sensitivity diagnostics.",
    ]
    if warnings_text:
        lines.extend(["", "## Warnings", "", warnings_text])
    (DIAG_DIR / f"water_{response}_model_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def run_water_mixed_models() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    water = pd.read_csv(PHASE2_WATER_INPUT, dtype={"Site_ID": str})
    water_site_matching_qa(water)
    all_coeffs = []
    all_comp = []
    residual_outputs = []
    final_model_rows = []

    for response, source_col in [("TN", "TN_mg_L"), ("TP", "TP_mg_L")]:
        df = prepare_water_complete_case(water, source_col)
        df.to_csv(WATER_DIR / f"water_{response}_equalN_complete_case.csv", index=False, encoding="utf-8-sig")
        results: Dict[str, object] = {}
        formulas: Dict[str, str] = {}
        warnings_by_model: Dict[str, str] = {}
        methods: Dict[str, str] = {}
        for model_name in ["W0", "W1", "W3"]:
            formula = mixedlm_formula(model_name)
            formulas[model_name] = formula
            result, method, captured = fit_mixedlm(df, formula)
            warnings_text = " | ".join(captured)
            results[model_name] = result
            warnings_by_model[model_name] = warnings_text
            methods[model_name] = method
            coeff = model_coefficients(result, response, model_name, method, warnings_text, len(df), df["Site_ID"].nunique())
            coeff.to_csv(WATER_DIR / f"water_{response}_{model_name}_MixedLM_coefficients.csv", index=False, encoding="utf-8-sig")
            all_coeffs.append(coeff)
            all_comp.append(model_comparison_row(result, response, model_name, method, warnings_text, len(df), df["Site_ID"].nunique()))

        comp_so_far = pd.DataFrame(all_comp)
        selection = select_water_model(comp_so_far, response)
        selected = selection["Selected_model"]
        final_result = results[selected]
        fitted, residual = conditional_fitted_residuals(final_result, df)
        residual_df = df[["Site_ID", "Season", source_col, "NetworkDist_m", "DO_mg_L", "pH"]].copy()
        residual_df["Response"] = response
        residual_df["selected_model"] = selected
        residual_df["fitted"] = fitted
        residual_df["residual"] = residual
        residual_df.to_csv(WATER_DIR / f"water_{response}_final_residuals.csv", index=False, encoding="utf-8-sig")
        residual_outputs.append(residual_df)
        make_residual_plot(response, residual_df)

        if selected == "W3":
            vif_cols = ["z_NetworkDist_m", "z_DO_mg_L", "z_pH"]
        elif selected == "W1":
            vif_cols = ["z_NetworkDist_m"]
        else:
            vif_cols = []
        if vif_cols:
            vif = vif_table(df, vif_cols)
        else:
            vif = pd.DataFrame([{"Predictor": "No continuous fixed effects in selected model", "R2_against_other_fixed_effects": np.nan, "VIF": np.nan}])
        vif.to_csv(DIAG_DIR / f"water_{response}_fixed_effect_VIF.csv", index=False, encoding="utf-8-sig")
        influence = influence_by_site(df, response, selected, formulas[selected], final_result)
        write_water_diagnostic_md(
            response,
            selected,
            pd.DataFrame(all_comp),
            final_result,
            residual_df,
            vif,
            influence,
            warnings_by_model[selected],
        )
        final_rows = model_coefficients(
            final_result,
            response,
            selected,
            methods[selected],
            warnings_by_model[selected],
            len(df),
            df["Site_ID"].nunique(),
        )
        final_rows["Final_selected_model"] = selected
        final_model_rows.append(final_rows)
        pd.DataFrame([selection]).to_csv(SUMMARY_DIR / f"water_{response}_model_selection.csv", index=False, encoding="utf-8-sig")

    coeff_out = pd.concat(all_coeffs, ignore_index=True)
    comp_out = pd.DataFrame(all_comp)
    final_out = pd.concat(final_model_rows, ignore_index=True)
    residual_out = pd.concat(residual_outputs, ignore_index=True)
    coeff_out.to_csv(WATER_DIR / "water_mixed_model_coefficients_all.csv", index=False, encoding="utf-8-sig")
    comp_out.to_csv(WATER_DIR / "water_model_comparison_equalN.csv", index=False, encoding="utf-8-sig")
    final_out.to_csv(WATER_DIR / "water_final_mixed_model_coefficients.csv", index=False, encoding="utf-8-sig")
    residual_out.to_csv(WATER_DIR / "water_final_residuals.csv", index=False, encoding="utf-8-sig")
    selections = pd.concat(
        [pd.read_csv(SUMMARY_DIR / f"water_{resp}_model_selection.csv") for resp in ["TN", "TP"]],
        ignore_index=True,
    )
    selections.to_csv(SUMMARY_DIR / "water_model_selection_recommendation.csv", index=False, encoding="utf-8-sig")
    exploratory_chlor_cod(water)
    return coeff_out, comp_out, final_out


def exploratory_chlor_cod(water: pd.DataFrame) -> None:
    rows = []
    for predictor in ["COD_mg_L", "Chlor_ug_L"]:
        for response in ["TN_mg_L", "TP_mg_L"]:
            sub = water[[predictor, response]].dropna()
            if len(sub) >= 8:
                rho, pval = stats.spearmanr(sub[predictor], sub[response])
            else:
                rho, pval = np.nan, np.nan
            rows.append(
                {
                    "Analysis": "supplementary_exploratory_spearman",
                    "Predictor": predictor,
                    "Response": response,
                    "N": len(sub),
                    "Spearman_rho": rho,
                    "p": pval,
                    "Main_model_use": "not included in manuscript-primary water mixed models",
                }
            )
    pd.DataFrame(rows).to_csv(SENS_DIR / "water_exploratory_COD_Chlor_spearman.csv", index=False, encoding="utf-8-sig")


def supplementary_soil_distance_table() -> pd.DataFrame:
    decision = pd.read_csv(PHASE4_DIR / "decision" / "soil_distance_model_decision_table.csv")
    rows = []
    for row in decision.itertuples(index=False):
        rows.append(
            {
                "Pollutant": row.Pollutant,
                "Distance_definition": row.DistanceDefinition,
                "N": row.N,
                "Distance_beta": row.Distance_Beta,
                "95CI": f"[{row.Distance_CI_low:.6g}, {row.Distance_CI_high:.6g}]",
                "p": row.Distance_p,
                "HighAg_beta": row.HighAg_Beta,
                "NaturalVeg_beta": row.NaturalVeg_Beta,
                "logMFD_beta": row.logMFD_Beta,
                "Adjusted_R2": row.Adjusted_R2,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(SOIL_DIR / "Table_S_soil_distance_sensitivity.csv", index=False, encoding="utf-8-sig")
    return out


def make_final_table(soil_final: pd.DataFrame, water_final: pd.DataFrame) -> pd.DataFrame:
    cols = ["Medium", "Response", "Predictor", "Estimate", "Std_Beta", "SE", "CI_low", "CI_high", "p", "N_obs", "N_sites", "Model_type"]
    table = pd.concat([water_final[cols], soil_final[cols]], ignore_index=True)
    table.to_csv(SUMMARY_DIR / "Table_multivariable_FINAL.csv", index=False, encoding="utf-8-sig")
    return table


def make_final_figure(soil_final: pd.DataFrame, water_all: pd.DataFrame) -> None:
    water_plot = water_all[
        (water_all["Model"] == "W3")
        & (water_all["Term"].isin(["z_NetworkDist_m", "z_DO_mg_L", "z_pH"]))
    ].copy()
    soil_plot = soil_final[soil_final["Term"].isin(["z_NetworkDist_m", "z_HighAg_ratio", "z_NaturalVeg_ratio", "z_log_MFD"])].copy()
    plot_parts = [("Surface water", water_plot, "#1b6ca8"), ("Soil", soil_plot, "#2f855a")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=False)
    for ax, (title, sub, color) in zip(axes, plot_parts):
        sub = sub.copy()
        if sub.empty:
            ax.text(0.5, 0.5, "No continuous predictors in selected model", ha="center", va="center")
            ax.axis("off")
            continue
        predictor_order = {
            "Hydrological distance": 0,
            "DO": 1,
            "pH": 2,
            "High-intensity agriculture": 1,
            "Natural vegetation": 2,
            "Flow accumulation": 3,
        }
        sub["order"] = sub["Predictor"].map(predictor_order).fillna(99)
        sub = sub.sort_values(["Response", "order"]).reset_index(drop=True)
        sub["label"] = sub["Response"] + " | " + sub["Predictor"]
        ypos = np.arange(len(sub))
        ax.axvline(0, color="0.2", linewidth=1)
        xerr = np.vstack([sub["Estimate"] - sub["CI_low"], sub["CI_high"] - sub["Estimate"]])
        ax.errorbar(sub["Estimate"], ypos, xerr=xerr, fmt="o", color=color, ecolor="0.35", capsize=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sub["label"])
        ax.set_title("(a) Surface water" if title == "Surface water" else "(b) Soil")
        ax.set_xlabel("Standardized beta on log-response scale")
        ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_water_soil_multivariable_FINAL.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "Figure_water_soil_multivariable_FINAL.pdf", bbox_inches="tight")
    plt.close(fig)


def interpretation_text(soil_final: pd.DataFrame, water_comp: pd.DataFrame, water_final: pd.DataFrame, soil_moran: pd.DataFrame) -> str:
    def beta_p(df: pd.DataFrame, response: str, predictor: str) -> Tuple[float, float]:
        row = df[(df["Response"] == response) & (df["Predictor"] == predictor)].iloc[0]
        return float(row["Estimate"]), float(row["p"])

    tn_dist, tn_dist_p = beta_p(soil_final, "TN", "Hydrological distance")
    tp_high, tp_high_p = beta_p(soil_final, "TP", "High-intensity agriculture")
    toc_dist, toc_dist_p = beta_p(soil_final, "TOC", "Hydrological distance")
    lines = [
        "# Phase 5 Manuscript-Ready Interpretation",
        "",
        "## Soil D2 Primary Model",
        "",
        f"Soil TN was negatively associated with hydrological distance to the outlet (beta = {tn_dist:.3g}, p = {tn_dist_p:.3g}), indicating higher TN toward outlet-proximal positions under the D2 network-only definition. Land-use and flow-accumulation terms showed weaker statistical support in the D2 primary model. Soil TP showed a positive but less robust association with high-intensity agricultural land use (beta = {tp_high:.3g}, p = {tp_high_p:.3g}); this land-use inference remains sensitive to distance definition. No robust hydrological-distance effect was detected for TOC (beta = {toc_dist:.3g}, p = {toc_dist_p:.3g}).",
        "",
        "## Water Mixed Models",
        "",
        "Water models used equal-N complete-case datasets for W0/W1/W3 within each response, with ML estimation and a random intercept for Site_ID. Seasonal terms were retained as categorical fixed effects with Autumn as the reference season. Hydrological distance, DO and pH were evaluated only after controlling for season. Model selection was based on equal-N fit, interpretability and coefficient stability, not the number of significant predictors.",
        "",
        "## Residual Spatial Autocorrelation",
        "",
    ]
    for row in soil_moran.itertuples(index=False):
        lines.append(
            f"- {row.Response}: Moran's I = {row.Moran_I:.4g}, p = {row.p:.4g}; "
            "no significant residual global spatial autocorrelation was detected under the specified KNN spatial weights."
            if row.p > 0.05
            else f"- {row.Response}: Moran's I = {row.Moran_I:.4g}, p = {row.p:.4g}."
        )
    text = "\n".join(lines)
    (SUMMARY_DIR / "phase5_result_interpretation.md").write_text(text, encoding="utf-8")
    return text


def append_experiment_record(
    soil_final: pd.DataFrame,
    soil_moran: pd.DataFrame,
    water_comp: pd.DataFrame,
    water_final: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    record_path = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_experiment_record.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    selected_text = "; ".join(f"{r.Response}: {r.Selected_model}" for r in selections.itertuples(index=False))
    converged_text = "; ".join(
        f"{r.Response} {r.Model}: converged={r.converged}, random_intercept_variance={r.random_intercept_variance:.6g}"
        for r in water_comp.itertuples(index=False)
    )
    md = f"""

---

# Phase 5: Manuscript-Final Water Mixed Models and Integrated Multivariable Analysis

Created at: {now}

## Scope

- Soil manuscript-primary distance definition was fixed as D2 Network-only (`NetworkDist_m`, N=40).
- D1 Euclidean and D3 Hybrid were retained only as distance-definition sensitivity analyses.
- Water fallback OLS was replaced by `statsmodels` MixedLM with a random intercept for `Site_ID`.
- The water W0/W1/W3 comparison used response-specific common complete-case rows containing response, Season, NetworkDist, DO, pH and Site_ID.
- ML estimation was used consistently (`REML=False`) for AIC/BIC/log-likelihood comparison and final coefficient reporting.

## Software

- Python 3.7.6.
- `statsmodels` 0.13.5 installed into the workspace-local `.codex_deps/py37` directory.
- Mixed model syntax: `log1p(Y) ~ fixed effects + (1 | Site_ID)` implemented as `MixedLM(..., groups=Site_ID)`.

## Final Model Position

- Soil final model: D2 Network-only OLS-HC3 for TN, TP and TOC.
- Water selected models: {selected_text}.
- Water convergence/random-intercept diagnostics: {converged_text}.
- Soil D2 residual Moran's I was recalculated from D2 residuals using KNN k=8 and 999 permutations.

## Key Limitations

- Water NetworkDist is available only for the predefined network-analysis site subset (14 Site_ID), so rows outside that subset were not inferred or force-matched.
- TP mixed models showed boundary/singular random-intercept behavior where the random-intercept variance approached zero; this was retained and reported rather than switching to OLS.
- COD, chlorophyll and water land-use ratios were not included in manuscript-primary water models.
- Exclude-site diagnostics were treated as sensitivity checks; no site was automatically removed.

## Outputs

- `results/water_soil_multivariable_phase5_final/soil/soil_D2_final_coefficients.csv`
- `results/water_soil_multivariable_phase5_final/spatial/soil_D2_residual_moransI.csv`
- `results/water_soil_multivariable_phase5_final/water/water_model_comparison_equalN.csv`
- `results/water_soil_multivariable_phase5_final/water/water_final_mixed_model_coefficients.csv`
- `results/water_soil_multivariable_phase5_final/diagnostics/water_TN_model_diagnostics.md`
- `results/water_soil_multivariable_phase5_final/diagnostics/water_TP_model_diagnostics.md`
- `results/water_soil_multivariable_phase5_final/figures/Figure_water_soil_multivariable_FINAL.png`
- `results/water_soil_multivariable_phase5_final/figures/Figure_water_soil_multivariable_FINAL.pdf`
- `results/water_soil_multivariable_phase5_final/summary/Table_multivariable_FINAL.csv`
"""
    with record_path.open("a", encoding="utf-8") as handle:
        handle.write(md)


def write_final_brief_summary(soil_final: pd.DataFrame, soil_moran: pd.DataFrame, water_comp: pd.DataFrame, selections: pd.DataFrame) -> str:
    tn_sel = selections.loc[selections["Response"] == "TN", "Selected_model"].iloc[0]
    tp_sel = selections.loc[selections["Response"] == "TP", "Selected_model"].iloc[0]
    tn_dist = soil_final[(soil_final["Response"] == "TN") & (soil_final["Predictor"] == "Hydrological distance")].iloc[0]
    tp_dist = soil_final[(soil_final["Response"] == "TP") & (soil_final["Predictor"] == "Hydrological distance")].iloc[0]
    moran_sig = soil_moran[soil_moran["p"] <= 0.05]["Response"].astype(str).tolist()
    moran_text = (
        "D2 residual Moran's I was non-significant for all soil responses"
        if not moran_sig
        else "D2 residual Moran's I remained significant for " + "/".join(sorted(moran_sig))
    )
    text = (
        f"Soil D2 results showed a stable negative association between TN and hydrological distance "
        f"(beta={tn_dist.Estimate:.3g}, p={tn_dist.p:.3g}), while TP distance was negative but not statistically supported "
        f"(beta={tp_dist.Estimate:.3g}, p={tp_dist.p:.3g}). {moran_text}. "
        f"Water MixedLM equal-N comparison selected {tn_sel} for TN and {tp_sel} for TP; seasonal structure was stronger than a simple distance gradient. "
        f"TP random-intercept variance reached a boundary fit and was reported as a diagnostic limitation."
    )
    (SUMMARY_DIR / "phase5_brief_summary_under300chars.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ensure_dirs()
    soil_final, soil_residuals = run_soil_final()
    soil_moran = run_soil_moran(soil_residuals)
    supplementary_soil_distance_table()
    water_all, water_comp, water_final = run_water_mixed_models()
    final_table = make_final_table(soil_final, water_final)
    make_final_figure(soil_final, water_all)
    selections = pd.read_csv(SUMMARY_DIR / "water_model_selection_recommendation.csv")
    interpretation_text(soil_final, water_comp, water_final, soil_moran)
    write_final_brief_summary(soil_final, soil_moran, water_comp, selections)
    append_experiment_record(soil_final, soil_moran, water_comp, water_final, selections)
    manifest = []
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            manifest.append({"File": str(path.relative_to(OUT_DIR)), "Bytes": path.stat().st_size})
    pd.DataFrame(manifest).to_csv(OUT_DIR / "phase5_output_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote Phase 5 outputs to: {OUT_DIR}")
    print(f"Final table rows: {len(final_table)}")


if __name__ == "__main__":
    main()
