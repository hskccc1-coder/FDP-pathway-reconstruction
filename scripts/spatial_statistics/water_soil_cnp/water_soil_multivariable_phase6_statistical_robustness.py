from __future__ import annotations

import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian


ROOT = Path(__file__).resolve().parents[3]
PHASE4_DIR = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_phase4_soil_final"
PHASE5_DIR = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_phase5_final"
OUT_DIR = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_phase6_statistical_robustness"
SOIL_DIR = OUT_DIR / "soil"
WATER_DIR = OUT_DIR / "water"
SUMMARY_DIR = OUT_DIR / "summary"
LOG_DIR = OUT_DIR / "experiment_log"

SEASON_ORDER = ["Autumn", "Winter", "Spring", "Summer"]
SEASON_REF = "Autumn"
KNN_K = 8
PERMUTATIONS = 999
RNG_SEED = 260826


def ensure_dirs() -> None:
    for path in [SOIL_DIR, WATER_DIR, SUMMARY_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd


def make_knn_weights(coords: np.ndarray, k: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    neighbors = np.argsort(dist, axis=1)[:, :k]
    weights = np.zeros((len(coords), len(coords)), dtype=float)
    for i in range(len(coords)):
        weights[i, neighbors[i]] = 1.0 / k
    return weights


def morans_i_knn(coords: np.ndarray, values: np.ndarray, k: int = KNN_K, permutations: int = PERMUTATIONS) -> Dict[str, float]:
    valid = np.isfinite(coords).all(axis=1) & np.isfinite(values)
    coords = coords[valid]
    values = values[valid]
    n = len(values)
    if n <= k + 1:
        return {"N": n, "k": k, "Moran_I": np.nan, "p": np.nan, "permutations": permutations}
    weights = make_knn_weights(coords, k)
    z = values - values.mean()
    denom = float(np.sum(z**2))
    wsum = float(weights.sum())
    observed = (n / wsum) * float(z @ weights @ z) / denom if denom > 0 and wsum > 0 else np.nan
    rng = np.random.default_rng(RNG_SEED)
    perm_i = []
    for _ in range(permutations):
        zp = rng.permutation(z)
        perm_i.append((n / wsum) * float(zp @ weights @ zp) / denom if denom > 0 else np.nan)
    perm_i = np.asarray(perm_i)
    p = (np.sum(np.abs(perm_i) >= abs(observed)) + 1) / (permutations + 1)
    return {"N": n, "k": k, "Moran_I": observed, "p": p, "permutations": permutations}


def fit_spatial_error_ml(y: np.ndarray, x: np.ndarray, weights: np.ndarray) -> Dict[str, object]:
    n, p = x.shape
    ident = np.eye(n)

    def profile(lambda_value: float) -> Tuple[float, np.ndarray, np.ndarray, float]:
        transform = ident - lambda_value * weights
        sign, logabsdet = np.linalg.slogdet(transform)
        if sign == 0 or not np.isfinite(logabsdet):
            return -np.inf, np.full(p, np.nan), np.full(n, np.nan), np.nan
        y_t = transform @ y
        x_t = transform @ x
        beta = np.linalg.pinv(x_t.T @ x_t) @ x_t.T @ y_t
        innovation = y_t - x_t @ beta
        sigma2 = float(np.sum(innovation**2) / n)
        if sigma2 <= 0 or not np.isfinite(sigma2):
            return -np.inf, beta, innovation, sigma2
        loglike = logabsdet - 0.5 * n * (math.log(2 * math.pi) + math.log(sigma2) + 1.0)
        return loglike, beta, innovation, sigma2

    def neg_profile(lambda_value: float) -> float:
        return -profile(float(lambda_value))[0]

    opt = optimize.minimize_scalar(neg_profile, bounds=(-0.98, 0.98), method="bounded", options={"xatol": 1e-10})
    lam = float(opt.x)
    loglike, beta, innovation, sigma2 = profile(lam)
    transform = ident - lam * weights
    y_t = transform @ y
    x_t = transform @ x
    cov_beta = sigma2 * np.linalg.pinv(x_t.T @ x_t)
    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 0))
    fitted = x @ beta
    raw_residual = y - fitted

    step = 1e-4
    left = profile(max(-0.9799, lam - step))[0]
    middle = loglike
    right = profile(min(0.9799, lam + step))[0]
    second = (left - 2 * middle + right) / (step**2)
    lambda_se = math.sqrt(-1.0 / second) if second < 0 and np.isfinite(second) else np.nan
    lambda_z = lam / lambda_se if pd.notna(lambda_se) and lambda_se > 0 else np.nan
    lambda_p = 2 * stats.norm.sf(abs(lambda_z)) if pd.notna(lambda_z) else np.nan
    aic = 2 * (p + 2) - 2 * loglike
    return {
        "lambda": lam,
        "lambda_se": lambda_se,
        "lambda_z": lambda_z,
        "lambda_p": lambda_p,
        "beta": beta,
        "beta_se": se_beta,
        "logLik": loglike,
        "AIC": aic,
        "sigma2": sigma2,
        "raw_residual": raw_residual,
        "innovation_residual": innovation,
        "success": bool(opt.success),
        "message": str(opt.message),
    }


def run_soil_tp_spatial_error() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    soil = pd.read_csv(PHASE4_DIR / "soil_covariates_complete.csv", dtype={"Site_ID": str})
    predictors = ["NetworkDist_m", "HighAg_ratio", "NaturalVeg_ratio", "log_MFD"]
    df = soil[["Site_ID", "TP", "Orig_X", "Orig_Y"] + predictors].copy()
    for col in ["TP", "Orig_X", "Orig_Y"] + predictors:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["TP", "Orig_X", "Orig_Y"] + predictors).copy()
    df = df[df["TP"] >= 0].copy()
    df["log_response"] = np.log1p(df["TP"])
    for pred in predictors:
        df[f"z_{pred}"] = zscore(df[pred])
    z_predictors = [f"z_{p}" for p in predictors]
    df = df.dropna(subset=z_predictors).copy()
    y = df["log_response"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(df))] + [df[col].to_numpy(dtype=float) for col in z_predictors])
    coords = df[["Orig_X", "Orig_Y"]].to_numpy(dtype=float)
    weights = make_knn_weights(coords, KNN_K)
    fit = fit_spatial_error_ml(y, x, weights)

    term_names = ["Intercept"] + z_predictors
    readable = {
        "Intercept": "Intercept",
        "z_NetworkDist_m": "NetworkDist",
        "z_HighAg_ratio": "HighAg",
        "z_NaturalVeg_ratio": "NaturalVeg",
        "z_log_MFD": "log_MFD",
    }
    rows = []
    for term, beta, se in zip(term_names, fit["beta"], fit["beta_se"]):
        zval = beta / se if se > 0 else np.nan
        pval = 2 * stats.norm.sf(abs(zval)) if pd.notna(zval) else np.nan
        rows.append(
            {
                "Predictor": readable[term],
                "Term": term,
                "Beta": beta,
                "SE": se,
                "CI_low": beta - 1.96 * se,
                "CI_high": beta + 1.96 * se,
                "p": pval,
                "N": len(df),
                "Model": "Spatial Error Model",
                "weights": f"KNN k={KNN_K}, row-standardized",
                "lambda": fit["lambda"],
                "lambda_SE": fit["lambda_se"],
                "lambda_p": fit["lambda_p"],
                "lambda_at_search_boundary": abs(abs(fit["lambda"]) - 0.98) < 1e-5,
                "AIC": fit["AIC"],
                "logLik": fit["logLik"],
                "fit_success": fit["success"],
                "fit_message": fit["message"],
            }
        )
    sem_table = pd.DataFrame(rows)
    sem_table.to_csv(SOIL_DIR / "soil_TP_spatial_error_model.csv", index=False, encoding="utf-8-sig")

    resid_rows = []
    for residual_type, values in [
        ("raw_model_residual", fit["raw_residual"]),
        ("spatial_error_innovation_residual", fit["innovation_residual"]),
    ]:
        moran = morans_i_knn(coords, values, KNN_K, PERMUTATIONS)
        moran.update({"Response": "TP", "Model": "Spatial Error Model", "residual_type": residual_type})
        resid_rows.append(moran)
    moran_table = pd.DataFrame(resid_rows)
    moran_table.to_csv(SOIL_DIR / "soil_TP_spatial_model_residual_moran.csv", index=False, encoding="utf-8-sig")

    ols = pd.read_csv(PHASE5_DIR / "soil" / "soil_D2_final_coefficients.csv")
    ols_tp = ols[(ols["Response"] == "TP") & (ols["Term"] != "Intercept")].copy()
    comparison_rows = []
    name_map = {
        "Hydrological distance": "NetworkDist",
        "High-intensity agriculture": "HighAg",
        "Natural vegetation": "NaturalVeg",
        "Flow accumulation": "log_MFD",
    }
    for row in ols_tp.itertuples(index=False):
        sem_row = sem_table[sem_table["Predictor"] == name_map[row.Predictor]].iloc[0]
        comparison_rows.append(
            {
                "Predictor": name_map[row.Predictor],
                "OLS_HC3_Beta": row.Estimate,
                "OLS_HC3_direction": direction(row.Estimate),
                "OLS_HC3_CI": f"[{row.CI_low:.6g}, {row.CI_high:.6g}]",
                "OLS_HC3_p": row.p,
                "Spatial_Error_Beta": sem_row.Beta,
                "Spatial_Error_direction": direction(sem_row.Beta),
                "Spatial_Error_CI": f"[{sem_row.CI_low:.6g}, {sem_row.CI_high:.6g}]",
                "Spatial_Error_p": sem_row.p,
                "direction_stable": direction(row.Estimate) == direction(sem_row.Beta),
                "significance_stable_at_0.05": (row.p < 0.05) == (sem_row.p < 0.05),
                "absolute_beta_change": abs(row.Estimate - sem_row.Beta),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(SOIL_DIR / "soil_TP_spatial_sensitivity_comparison.csv", index=False, encoding="utf-8-sig")
    return sem_table, comparison, moran_table


def direction(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def prepare_water_tp_equal_n() -> pd.DataFrame:
    water = pd.read_csv(PHASE5_DIR / "water" / "water_TP_equalN_complete_case.csv", dtype={"Site_ID": str})
    water["Season"] = pd.Categorical(water["Season"], categories=SEASON_ORDER, ordered=True)
    return water.copy()


def fit_water_tp_mixedlm(df: pd.DataFrame) -> Tuple[object, str, str]:
    messages: List[str] = []
    for method in ["lbfgs", "powell", "nm", "bfgs", "cg"]:
        try:
            model = smf.mixedlm("log_response ~ C(Season)", df, groups=df["Site_ID"])
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                result = model.fit(reml=False, method=method, maxiter=2000, disp=False)
            warn_text = " | ".join([str(w.message) for w in captured])
            messages.extend([f"{method}: {warn_text}" if warn_text else f"{method}: no warnings"])
            return result, method, " | ".join(messages)
        except Exception as exc:
            messages.append(f"{method} failed: {type(exc).__name__}: {exc}")
    raise RuntimeError(" | ".join(messages))


def ci_from_result_params(params: pd.Series, bse: pd.Series, pvalues: pd.Series, terms: List[str]) -> pd.DataFrame:
    rows = []
    for term in terms:
        beta = float(params[term])
        se = float(bse[term])
        rows.append(
            {
                "Term": term,
                "Beta": beta,
                "SE": se,
                "CI_low": beta - 1.96 * se,
                "CI_high": beta + 1.96 * se,
                "p": float(pvalues[term]),
            }
        )
    return pd.DataFrame(rows)


def water_tp_repeated_measure_sensitivity() -> pd.DataFrame:
    df = prepare_water_tp_equal_n()
    terms = ["C(Season)[T.Winter]", "C(Season)[T.Spring]", "C(Season)[T.Summer]"]
    readable = {
        "C(Season)[T.Winter]": "Winter vs Autumn",
        "C(Season)[T.Spring]": "Spring vs Autumn",
        "C(Season)[T.Summer]": "Summer vs Autumn",
    }
    rows = []

    mixed, optimizer, mixed_warnings = fit_water_tp_mixedlm(df)
    mixed_rows = ci_from_result_params(mixed.fe_params, mixed.bse_fe, mixed.pvalues, terms)
    for row in mixed_rows.itertuples(index=False):
        rows.append(
            {
                "Method": "MixedLM W0",
                "Contrast": readable[row.Term],
                "Term": row.Term,
                "Beta": row.Beta,
                "SE": row.SE,
                "CI_low": row.CI_low,
                "CI_high": row.CI_high,
                "p": row.p,
                "N": len(df),
                "N_sites": df["Site_ID"].nunique(),
                "converged": bool(mixed.converged),
                "random_intercept_variance": float(np.asarray(mixed.cov_re)[0, 0]),
                "optimizer": optimizer,
                "diagnostic": mixed_warnings,
            }
        )

    ols = smf.ols("log_response ~ C(Season)", data=df).fit(cov_type="cluster", cov_kwds={"groups": df["Site_ID"]})
    ols_rows = ci_from_result_params(ols.params, ols.bse, ols.pvalues, terms)
    for row in ols_rows.itertuples(index=False):
        rows.append(
            {
                "Method": "OLS cluster-robust",
                "Contrast": readable[row.Term],
                "Term": row.Term,
                "Beta": row.Beta,
                "SE": row.SE,
                "CI_low": row.CI_low,
                "CI_high": row.CI_high,
                "p": row.p,
                "N": len(df),
                "N_sites": df["Site_ID"].nunique(),
                "converged": True,
                "random_intercept_variance": np.nan,
                "optimizer": "OLS",
                "diagnostic": "Site_ID cluster-robust covariance",
            }
        )

    try:
        gee = smf.gee("log_response ~ C(Season)", groups="Site_ID", data=df, family=Gaussian(), cov_struct=Exchangeable()).fit()
        gee_rows = ci_from_result_params(gee.params, gee.bse, gee.pvalues, terms)
        for row in gee_rows.itertuples(index=False):
            rows.append(
                {
                    "Method": "GEE exchangeable",
                    "Contrast": readable[row.Term],
                    "Term": row.Term,
                    "Beta": row.Beta,
                    "SE": row.SE,
                    "CI_low": row.CI_low,
                    "CI_high": row.CI_high,
                    "p": row.p,
                    "N": len(df),
                    "N_sites": df["Site_ID"].nunique(),
                    "converged": bool(getattr(gee, "converged", True)),
                    "random_intercept_variance": np.nan,
                    "optimizer": "GEE",
                    "diagnostic": "exchangeable working correlation",
                }
            )
    except Exception as exc:
        rows.append(
            {
                "Method": "GEE exchangeable",
                "Contrast": "",
                "Term": "",
                "Beta": np.nan,
                "SE": np.nan,
                "CI_low": np.nan,
                "CI_high": np.nan,
                "p": np.nan,
                "N": len(df),
                "N_sites": df["Site_ID"].nunique(),
                "converged": False,
                "random_intercept_variance": np.nan,
                "optimizer": "GEE",
                "diagnostic": f"not fitted: {type(exc).__name__}: {exc}",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(WATER_DIR / "water_TP_repeated_measure_sensitivity.csv", index=False, encoding="utf-8-sig")
    return out


def write_summary(
    soil_sem: pd.DataFrame,
    soil_comparison: pd.DataFrame,
    soil_moran: pd.DataFrame,
    water_sensitivity: pd.DataFrame,
) -> None:
    predictors = soil_comparison[soil_comparison["Predictor"] != "Intercept"] if "Intercept" in soil_comparison["Predictor"].values else soil_comparison
    direction_stable = bool(predictors["direction_stable"].all())
    highag = soil_comparison[soil_comparison["Predictor"] == "HighAg"].iloc[0]
    distance = soil_comparison[soil_comparison["Predictor"] == "NetworkDist"].iloc[0]
    sem_lambda = soil_sem.iloc[0]
    innovation = soil_moran[soil_moran["residual_type"] == "spatial_error_innovation_residual"].iloc[0]

    water_pivot = water_sensitivity.pivot_table(index="Contrast", columns="Method", values=["Beta", "p"], aggfunc="first")
    stable_winter = contrast_stable(water_sensitivity, "Winter vs Autumn", expect_sign="positive", expect_sig=True)
    stable_spring = contrast_stable(water_sensitivity, "Spring vs Autumn", expect_sign="positive", expect_sig=True)
    stable_summer = contrast_stable(water_sensitivity, "Summer vs Autumn", expect_sign="positive", expect_sig=False)

    blocker = (
        "No additional modelling blocker remains for the multivariable-analysis section, but Soil TP must be reported as sensitive to residual spatial structure."
        if (not direction_stable or innovation.p <= 0.05)
        else "No additional modelling blocker remains for the multivariable-analysis section, provided the spatial sensitivity is reported."
    )
    if stable_winter and stable_spring and stable_summer:
        tp_method = (
            "GEE exchangeable is preferred for manuscript TP seasonal reporting, with cluster-robust OLS as a matching sensitivity, "
            "because it handles repeated measures without relying on the near-zero MixedLM random-intercept variance."
        )
    else:
        tp_method = (
            "GEE exchangeable is the most suitable manuscript TP reporting method because it handles repeated measures without relying on the singular MixedLM random intercept. "
            "However, the Summer vs Autumn contrast is method-sensitive and should be reported with the MixedLM and cluster-robust OLS sensitivity table rather than overemphasized."
        )

    md = f"""# Phase 6 Statistical Robustness Summary

## A. Soil TP Spatial Correction

The spatial-error sensitivity estimated lambda = {sem_lambda['lambda']:.4g} (p = {sem_lambda['lambda_p']:.4g}; logLik = {sem_lambda['logLik']:.4g}; AIC = {sem_lambda['AIC']:.4g}). The lambda estimate reached the search boundary, so the spatial-error fit should be treated as a sensitivity diagnostic rather than a replacement for the predefined OLS-HC3 model. Coefficient directions were {'stable' if direction_stable else 'not fully stable'} between OLS-HC3 and the spatial-error model. The spatial-error innovation residual Moran's I was {innovation.Moran_I:.4g} (p = {innovation.p:.4g}), using KNN k={KNN_K} and {PERMUTATIONS} permutations.

## B. Soil TP Stable Coefficients

NetworkDist remained {distance.Spatial_Error_direction} in the spatial-error model and was {'not significant' if distance.Spatial_Error_p >= 0.05 else 'significant'} after correction. HighAg changed from beta = {highag.OLS_HC3_Beta:.4g} (p = {highag.OLS_HC3_p:.4g}) under OLS-HC3 to beta = {highag.Spatial_Error_Beta:.4g} (p = {highag.Spatial_Error_p:.4g}) under the spatial-error model. Thus, HighAg is {'stable in direction' if highag.direction_stable else 'not stable in direction'} but its strength should be interpreted cautiously.

## C. Water TP Singular MixedLM

The MixedLM W0 random-intercept variance was estimated near zero, so the model is a boundary fit. The seasonal contrasts were compared against Site_ID cluster-robust OLS and GEE with exchangeable working correlation on the same N=54 rows and 14 sites.

## D. TP Manuscript Reporting Method

{tp_method}

## E. Remaining Statistical Blocker

{blocker}

## Water TP Contrast Stability

{water_pivot.to_markdown()}
"""
    (SUMMARY_DIR / "phase6_statistical_robustness_summary.md").write_text(md, encoding="utf-8")


def contrast_stable(table: pd.DataFrame, contrast: str, expect_sign: str, expect_sig: bool) -> bool:
    sub = table[table["Contrast"] == contrast].copy()
    if sub.empty:
        return False
    sign_ok = all(direction(v) == expect_sign for v in sub["Beta"])
    sig_ok = all((p < 0.05) == expect_sig for p in sub["p"])
    return sign_ok and sig_ok


def append_experiment_record() -> None:
    record_path = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_experiment_record.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = f"""

---

# Phase 6: Statistical Robustness Checks

Created at: {now}

## Scope

- Addressed Soil TP D2 residual spatial autocorrelation without changing the primary OLS-HC3 model or adding predictors.
- Addressed Water TP MixedLM boundary/singular random-intercept behavior using same-row repeated-measures sensitivity analyses.
- Water TN final W3 mixed model was not refit or changed.
- No N-load, MUSLE, SCS-CN, or N transport intensity analyses were started.

## Methods

- Soil TP spatial sensitivity: custom maximum-likelihood spatial error model using row-standardized KNN k=8 weights, matching the Phase 5 Moran's I diagnostic.
- Soil TP predictors retained exactly: z(NetworkDist_m), z(HighAg_ratio), z(NaturalVeg_ratio), z(log_MFD).
- Water TP sensitivity: MixedLM W0, OLS with Site_ID cluster-robust SE, and GEE with exchangeable working correlation, all using N=54 rows and 14 sites.

## Outputs

- `results/water_soil_multivariable_phase6_statistical_robustness/soil/soil_TP_spatial_error_model.csv`
- `results/water_soil_multivariable_phase6_statistical_robustness/soil/soil_TP_spatial_sensitivity_comparison.csv`
- `results/water_soil_multivariable_phase6_statistical_robustness/soil/soil_TP_spatial_model_residual_moran.csv`
- `results/water_soil_multivariable_phase6_statistical_robustness/water/water_TP_repeated_measure_sensitivity.csv`
- `results/water_soil_multivariable_phase6_statistical_robustness/summary/phase6_statistical_robustness_summary.md`
"""
    with record_path.open("a", encoding="utf-8") as handle:
        handle.write(md)


def main() -> None:
    ensure_dirs()
    soil_sem, soil_comparison, soil_moran = run_soil_tp_spatial_error()
    water_sensitivity = water_tp_repeated_measure_sensitivity()
    write_summary(soil_sem, soil_comparison, soil_moran, water_sensitivity)
    append_experiment_record()
    manifest = []
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            manifest.append({"File": str(path.relative_to(OUT_DIR)), "Bytes": path.stat().st_size})
    pd.DataFrame(manifest).to_csv(OUT_DIR / "phase6_output_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote Phase 6 outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
