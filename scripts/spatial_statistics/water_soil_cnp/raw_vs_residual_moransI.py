from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PHASE5_DIR = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_phase5_final"
SOIL_COVARIATES = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_phase4_soil_final" / "cleaned" / "soil_covariates_complete.csv"
WATER_COORDS = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "figure3_network_distance_stage1" / "water_sites_network_distance.csv"
OUT_PATH = PHASE5_DIR / "spatial" / "raw_vs_residual_moransI.csv"

KNN_K = 8
PERMUTATIONS = 999
RNG_SEED = 260826
WEIGHT_DEFINITION = "KNN k=8, row-standardized, 999 permutations"


def morans_i_knn(coords: np.ndarray, values: np.ndarray, k: int = KNN_K, permutations: int = PERMUTATIONS) -> Dict[str, float]:
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(coords).all(axis=1) & np.isfinite(values)
    coords = coords[valid]
    values = values[valid]
    n = len(values)
    if n <= k + 1:
        return {"N": n, "Moran_I": np.nan, "p": np.nan}

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

    rng = np.random.default_rng(RNG_SEED)
    perm_i: List[float] = []
    for _ in range(permutations):
        zp = rng.permutation(z)
        perm_i.append((n / wsum) * float(zp @ weights @ zp) / denom if denom > 0 and wsum > 0 else np.nan)
    perm_i = np.asarray(perm_i)
    p = (np.sum(np.abs(perm_i) >= abs(observed)) + 1) / (permutations + 1) if np.isfinite(observed) else np.nan
    return {"N": n, "Moran_I": observed, "p": p}


def soil_rows() -> List[Dict[str, object]]:
    soil = pd.read_csv(SOIL_COVARIATES, dtype={"Site_ID": str})
    residual_moran = pd.read_csv(PHASE5_DIR / "spatial" / "soil_D2_residual_moransI.csv")
    rows: List[Dict[str, object]] = []
    for response in ["TN", "TP", "TOC"]:
        residuals = pd.read_csv(PHASE5_DIR / "soil" / f"soil_D2_final_{response}_residuals.csv", dtype={"Site_ID": str})
        model_sites = residuals[["Site_ID"]].copy()
        raw = model_sites.merge(soil[["Site_ID", "Orig_X", "Orig_Y", response]], on="Site_ID", how="left")
        raw_result = morans_i_knn(raw[["Orig_X", "Orig_Y"]].to_numpy(), raw[response].to_numpy())
        res_row = residual_moran[residual_moran["Response"] == response].iloc[0]
        rows.append(
            {
                "Variable": f"Soil {response}",
                "Raw_I": raw_result["Moran_I"],
                "Raw_p": raw_result["p"],
                "Residual_I": float(res_row["Moran_I"]),
                "Residual_p": float(res_row["p"]),
                "N": int(raw_result["N"]),
                "Weight_definition": WEIGHT_DEFINITION,
            }
        )
    return rows


def water_rows() -> List[Dict[str, object]]:
    residuals = pd.read_csv(PHASE5_DIR / "water" / "water_final_residuals.csv", dtype={"Site_ID": str})
    coords = pd.read_csv(WATER_COORDS, dtype={"Site_ID": str})[["Site_ID", "Orig_X", "Orig_Y"]]
    rows: List[Dict[str, object]] = []
    for response, raw_col in [("TN", "TN_mg_L"), ("TP", "TP_mg_L")]:
        sub = residuals[residuals["Response"] == response].copy()
        site = (
            sub.groupby("Site_ID", as_index=False)
            .agg(raw_value=(raw_col, "mean"), residual=("residual", "mean"), n_records=("residual", "size"))
            .merge(coords, on="Site_ID", how="inner")
        )
        raw_result = morans_i_knn(site[["Orig_X", "Orig_Y"]].to_numpy(), site["raw_value"].to_numpy())
        residual_result = morans_i_knn(site[["Orig_X", "Orig_Y"]].to_numpy(), site["residual"].to_numpy())
        rows.append(
            {
                "Variable": f"Water {response} site-level mean (exploratory)",
                "Raw_I": raw_result["Moran_I"],
                "Raw_p": raw_result["p"],
                "Residual_I": residual_result["Moran_I"],
                "Residual_p": residual_result["p"],
                "N": int(site["Site_ID"].nunique()),
                "Weight_definition": WEIGHT_DEFINITION
                + "; site-level means from final-model rows; exploratory diagnostic only because N≈14 and k=8 leaves limited spatial power",
            }
        )
    return rows


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(soil_rows() + water_rows())
    out = out[["Variable", "Raw_I", "Raw_p", "Residual_I", "Residual_p", "N", "Weight_definition"]]
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
