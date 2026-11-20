from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "soil_water_measurements"
RESULTS_DIR = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime"
OUT_DIR = RESULTS_DIR / "water_soil_multivariable_phase1"
DIAG_DIR = OUT_DIR / "diagnostics"
SUMMARY_DIR = OUT_DIR / "summary"

WATER_SHEETS = {
    "202410": ("Autumn", "2024 autumn"),
    "202412": ("Winter", "2024 winter"),
    "202504": ("Spring", "2025 spring"),
    "202506": ("Summer", "2025 summer"),
}

WATER_ENV_DIR = DATA_DIR / "小流域数据 (1) (1)_csv"
WATER_TNTP_PATH = DATA_DIR / "four_season_TN_TP" / "water_TN_TP_four_seasons_all_valid.csv"
WATER_DISTANCE_PATH = RESULTS_DIR / "figure3_network_distance_stage1" / "water_sites_network_distance.csv"
SOIL_POINTS_PATH = DATA_DIR / "土壤采样点" / "土壤采样点.shp"
LANDUSE_PATH = DATA_DIR / "土地利用类型" / "小流域土地利用类型.shp"

PLACEHOLDER_NA = {"", "nan", "NaN", "NA", "N/A", "未测", "None", "none", None}


def ensure_dirs() -> None:
    for path in [DIAG_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clean_cell(value):
    if value is None:
        return np.nan
    if isinstance(value, float) and math.isnan(value):
        return np.nan
    text = str(value).strip()
    if text in PLACEHOLDER_NA:
        return np.nan
    return value


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.map(clean_cell), errors="coerce")


def find_header_row(raw: pd.DataFrame) -> int:
    for idx, row in raw.iterrows():
        values = [str(v).strip() for v in row.tolist()]
        if "编号" in values and ("经度" in values or "纬度" in values):
            return int(idx)
    raise ValueError("Could not find a header row containing 编号/经度/纬度.")


def normalize_water_column(col: str) -> Optional[str]:
    name = str(col).strip()
    if not name or name.lower() == "nan" or name.startswith("Unnamed"):
        return None
    compact = name.replace(" ", "").replace("（", "(").replace("）", ")")
    lower = compact.lower()
    mapping = {
        "fid": "FID",
        "类型": "Site_type",
        "编号": "Site_ID",
        "经度": "Longitude",
        "纬度": "Latitude",
        "tn(mg/l)": "TN_mg_L",
        "tp(mg/l)": "TP_mg_L",
        "nh4+(mg/l)": "NH4_mg_L",
        "nh4+-n(mg/l)": "NH4_mg_L",
        "nh4": "NH4_mg_L",
        "no3-(mg/l)": "NO3_mg_L",
        "no3(mg/l)": "NO3_mg_L",
        "no3": "NO3_mg_L",
        "do(mg/l)": "DO_mg_L",
        "溶解氧": "DO_mg_L",
        "ph": "pH",
        "cod(mg/l)": "COD_mg_L",
        "chlor(ug/l)": "Chlor_ug_L",
        "chlorophyll-a": "Chlor_ug_L",
        "chlorophyll": "Chlor_ug_L",
        "tempc": "Temp_C",
        "temperature": "Temp_C",
        "tds(mg/l)": "TDS_mg_L",
        "ec": "EC",
        "cl-(mg/l)": "Cl_mg_L",
        "cl-": "Cl_mg_L",
        "toc": "TOC_mg_L",
    }
    return mapping.get(lower, name)


def dedupe_columns(columns: Iterable[Optional[str]]) -> List[Optional[str]]:
    seen: Dict[str, int] = {}
    out: List[Optional[str]] = []
    for col in columns:
        if col is None:
            out.append(None)
            continue
        count = seen.get(col, 0)
        seen[col] = count + 1
        out.append(col if count == 0 else f"{col}_{count + 1}")
    return out


def read_water_env_csv(path: Path, season: str, season_label: str, sheet: str) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=object, encoding="utf-8-sig")
    header_idx = find_header_row(raw)
    headers = dedupe_columns(normalize_water_column(v) for v in raw.iloc[header_idx].tolist())
    body = raw.iloc[header_idx + 1 :].copy()
    body.columns = headers
    body = body.loc[:, [c is not None for c in body.columns]]
    body = body.dropna(how="all")
    if "Site_ID" not in body.columns:
        raise ValueError(f"{path.name} missing Site_ID after header normalization.")

    keep = [
        "FID",
        "Site_type",
        "Site_ID",
        "Longitude",
        "Latitude",
        "TN_mg_L",
        "TP_mg_L",
        "DO_mg_L",
        "pH",
        "COD_mg_L",
        "Chlor_ug_L",
        "Temp_C",
        "TDS_mg_L",
        "EC",
        "NH4_mg_L",
        "NO3_mg_L",
        "Cl_mg_L",
        "TOC_mg_L",
    ]
    for col in keep:
        if col not in body.columns:
            body[col] = np.nan
    out = body[keep].copy()
    out["Sheet"] = sheet
    out["Season"] = season
    out["Season_label"] = season_label
    site_id = pd.to_numeric(out["Site_ID"].map(clean_cell), errors="coerce")
    out["Site_ID"] = site_id.map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    for col in [c for c in keep if c not in {"FID", "Site_type", "Site_ID"}]:
        out[col] = to_number(out[col])
    out = out[out["Site_ID"].notna()].copy()
    return out


def read_water_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    env_frames = []
    for sheet, (season, season_label) in WATER_SHEETS.items():
        env_frames.append(read_water_env_csv(WATER_ENV_DIR / f"{sheet}.csv", season, season_label, sheet))
    env = pd.concat(env_frames, ignore_index=True)

    tntp = pd.read_csv(WATER_TNTP_PATH, dtype={"Site_ID": str})
    tntp["Sheet"] = tntp["Sheet"].astype(str)
    tntp = tntp[tntp["Sheet"].isin(WATER_SHEETS)].copy()
    tntp["Site_ID"] = tntp["Site_ID"].astype(str)
    tntp = tntp.rename(columns={"TN_mg_L": "TN_mg_L_clean", "TP_mg_L": "TP_mg_L_clean"})

    distance = pd.read_csv(WATER_DISTANCE_PATH, dtype={"Site_ID": str})
    distance_cols = [
        c
        for c in [
            "Site_ID",
            "NetworkDist_m",
            "NetworkDist_km",
            "SnapDist_m",
            "connected_to_outlet",
            "distance_method",
        ]
        if c in distance.columns
    ]

    water = env.merge(
        tntp[
            [
                "Sheet",
                "Season",
                "Season_label",
                "Year",
                "Month",
                "Site_ID",
                "TN_mg_L_clean",
                "TP_mg_L_clean",
            ]
        ],
        on=["Sheet", "Season", "Season_label", "Site_ID"],
        how="outer",
    )
    water["TN_mg_L"] = water["TN_mg_L_clean"].combine_first(water["TN_mg_L"])
    water["TP_mg_L"] = water["TP_mg_L_clean"].combine_first(water["TP_mg_L"])
    water = water.drop(columns=["TN_mg_L_clean", "TP_mg_L_clean"])
    water = water.merge(distance[distance_cols], on="Site_ID", how="left")
    return water, distance


def read_soil_data() -> pd.DataFrame:
    soil = gpd.read_file(SOIL_POINTS_PATH)
    soil = pd.DataFrame(soil.drop(columns="geometry", errors="ignore"))
    soil = soil.rename(
        columns={
            "样点": "Site_ID",
            "底泥": "Site_type",
            "经度": "Longitude",
            "纬度": "Latitude",
            "No3N": "NO3N",
            "BurnDEMMFD": "MFD_accumulation",
            "DEM_1": "Elevation",
            "坡度_1": "Slope",
        }
    )
    for col in ["TN", "TP", "TOC", "NH4N", "NO3N", "MFD_accumulation", "Elevation", "Slope", "Longitude", "Latitude"]:
        if col in soil.columns:
            soil[col] = to_number(soil[col])
    if "MFD_accumulation" in soil.columns:
        soil["log_MFD"] = np.log1p(soil["MFD_accumulation"].clip(lower=0))
    return soil


def variable_stats(df: pd.DataFrame, variables: List[Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    total_n = len(df)
    for var, unit in variables:
        if var not in df.columns:
            rows.append(
                {
                    "Variable": var,
                    "Unit": unit,
                    "Present": False,
                    "N": 0,
                    "Missing": total_n,
                    "Min": np.nan,
                    "Median": np.nan,
                    "Mean": np.nan,
                    "Max": np.nan,
                    "SD": np.nan,
                }
            )
            continue
        x = to_number(df[var])
        rows.append(
            {
                "Variable": var,
                "Unit": unit,
                "Present": True,
                "N": int(x.notna().sum()),
                "Missing": int(x.isna().sum()),
                "Min": x.min(skipna=True),
                "Median": x.median(skipna=True),
                "Mean": x.mean(skipna=True),
                "Max": x.max(skipna=True),
                "SD": x.std(skipna=True),
            }
        )
    return pd.DataFrame(rows)


def field_presence(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    rows = []
    for col in expected:
        if col in df.columns:
            rows.append(
                {
                    "Field": col,
                    "Present": True,
                    "N_nonmissing": int(df[col].notna().sum()),
                    "N_missing": int(df[col].isna().sum()),
                    "Unique_nonmissing": int(df[col].dropna().nunique()),
                }
            )
        else:
            rows.append(
                {
                    "Field": col,
                    "Present": False,
                    "N_nonmissing": 0,
                    "N_missing": len(df),
                    "Unique_nonmissing": 0,
                }
            )
    return pd.DataFrame(rows)


def spearman_matrix(df: pd.DataFrame, variables: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    vars_present = [v for v in variables if v in df.columns and to_number(df[v]).notna().sum() >= 3]
    matrix = pd.DataFrame(index=vars_present, columns=vars_present, dtype=float)
    rows = []
    for a in vars_present:
        for b in vars_present:
            xa = to_number(df[a])
            xb = to_number(df[b])
            valid = xa.notna() & xb.notna()
            n = int(valid.sum())
            if n >= 3 and xa[valid].nunique() > 1 and xb[valid].nunique() > 1:
                rho, p_value = stats.spearmanr(xa[valid], xb[valid])
            else:
                rho, p_value = np.nan, np.nan
            matrix.loc[a, b] = rho
            if a < b:
                rows.append(
                    {
                        "Variable_1": a,
                        "Variable_2": b,
                        "N_pairwise": n,
                        "Spearman_rho": rho,
                        "P_value": p_value,
                        "High_correlation_abs_rho_gt_0_70": bool(pd.notna(rho) and abs(rho) > 0.70),
                    }
                )
    matrix = matrix.reset_index()
    matrix = matrix.rename(columns={"index": "Variable"})
    return matrix, pd.DataFrame(rows)


def vif_for_set(df: pd.DataFrame, variables: List[str], set_name: str) -> pd.DataFrame:
    present = []
    skip_rows = []
    for var in variables:
        if var not in df.columns:
            skip_rows.append({"Set": set_name, "Variable": var, "VIF": np.nan, "N_complete": 0, "Status": "missing"})
            continue
        x = to_number(df[var])
        if x.notna().sum() < 3:
            skip_rows.append({"Set": set_name, "Variable": var, "VIF": np.nan, "N_complete": 0, "Status": "too few non-missing values"})
        elif x.dropna().nunique() <= 1:
            skip_rows.append({"Set": set_name, "Variable": var, "VIF": np.nan, "N_complete": int(x.notna().sum()), "Status": "constant"})
        else:
            present.append(var)
    if len(present) < 2:
        for var in present:
            n = int(to_number(df[var]).notna().sum())
            skip_rows.append(
                {
                    "Set": set_name,
                    "Variable": var,
                    "VIF": np.nan,
                    "N_complete": n,
                    "Status": "not computed: fewer than 2 usable variables",
                }
            )
        return pd.DataFrame(skip_rows)
    X = pd.DataFrame({v: to_number(df[v]) for v in present}).dropna()
    n = len(X)
    rows = []
    if n <= len(present) + 1:
        for var in present:
            rows.append({"Set": set_name, "Variable": var, "VIF": np.nan, "N_complete": n, "Status": "too few complete cases"})
        return pd.DataFrame(rows + skip_rows)
    X = (X - X.mean()) / X.std(ddof=0)
    for var in present:
        y = X[var].to_numpy()
        other = X[[c for c in present if c != var]].to_numpy()
        design = np.column_stack([np.ones(n), other])
        beta = np.linalg.pinv(design.T @ design) @ design.T @ y
        pred = design @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        if pd.isna(r2):
            vif = np.nan
        elif r2 >= 0.999999:
            vif = np.inf
        else:
            vif = 1.0 / (1.0 - r2)
        rows.append(
            {
                "Set": set_name,
                "Variable": var,
                "VIF": vif,
                "N_complete": n,
                "Status": "warning: VIF > 5" if pd.notna(vif) and vif > 5 else "ok",
            }
        )
    return pd.DataFrame(rows + skip_rows)


def write_recommendations(
    water: pd.DataFrame,
    soil: pd.DataFrame,
    water_pairs: pd.DataFrame,
    soil_pairs: pd.DataFrame,
    water_vif: pd.DataFrame,
    soil_vif: pd.DataFrame,
) -> None:
    high_water = water_pairs[water_pairs["High_correlation_abs_rho_gt_0_70"]].copy()
    high_soil = soil_pairs[soil_pairs["High_correlation_abs_rho_gt_0_70"]].copy()
    cod_n = int(water["COD_mg_L"].notna().sum()) if "COD_mg_L" in water.columns else 0
    do_n = int(water["DO_mg_L"].notna().sum()) if "DO_mg_L" in water.columns else 0
    ph_n = int(water["pH"].notna().sum()) if "pH" in water.columns else 0
    chlor_n = int(water["Chlor_ug_L"].notna().sum()) if "Chlor_ug_L" in water.columns else 0
    water_n = len(water)
    water_sites = int(water["Site_ID"].nunique())
    water_distance_sites = int(water.loc[water["NetworkDist_km"].notna(), "Site_ID"].nunique())
    water_distance_rows = int(water["NetworkDist_km"].notna().sum())
    soil_n = len(soil)

    def format_high(df: pd.DataFrame) -> str:
        if df.empty:
            return "未发现 |rho| > 0.70 的高相关连续变量对。"
        pieces = []
        for row in df.sort_values("Spearman_rho", key=lambda s: s.abs(), ascending=False).itertuples():
            pieces.append(
                f"- {row.Variable_1} vs {row.Variable_2}: rho={row.Spearman_rho:.3f}, "
                f"N={row.N_pairwise}, p={row.P_value:.4g}"
            )
        return "\n".join(pieces)

    cod_status = "不建议进入第一版 W3"
    cod_reason = (
        f"COD 只有 {cod_n}/{water_n} 条非缺失记录，且主要集中在秋冬两季；春夏 COD 基本为空。"
    )
    cod_rows = water_vif[(water_vif["Variable"] == "COD_mg_L") & (water_vif["Set"].str.contains("with_COD"))]
    if not cod_rows.empty and cod_rows["Status"].astype(str).str.contains("warning").any():
        cod_reason += " 含 COD 的 VIF 诊断还提示共线性风险。"

    do_ph_status = "建议优先保留 DO 和 pH 作为候选水环境变量"
    do_ph_reason = (
        f"DO 非缺失 {do_n}/{water_n}，pH 非缺失 {ph_n}/{water_n}；覆盖明显好于 COD，"
        "且作为局地水环境条件具有明确解释意义。"
    )

    chlor_status = "建议单独做探索/补充分析"
    chlor_reason = (
        f"Chlorophyll 非缺失 {chlor_n}/{water_n}；它更可能是 TN/TP 变化后的生物响应，"
        "存在反向因果/内生性风险，不应默认放入 TN/TP 主模型。"
    )

    soil_predictors = ["log_MFD"]
    soil_notes = [
        "土壤点含 TN、TP、TOC、NH4N、NO3N、MFD_accumulation、log_MFD、Elevation、Slope。",
        "未发现 Dist_outlet、HighAg_ratio、NaturalVeg_ratio 或 soil pH 字段。",
        "Elevation 和 Slope 当前均为常数 0，不能用于回归或 VIF。",
    ]
    if "MFD_accumulation" in soil.columns and soil["MFD_accumulation"].notna().sum() > 0:
        soil_notes.append("在现有字段中，只有 log_MFD 可作为可用的水文汇流候选变量。")

    rec_rows = [
        {
            "Question": "A_Water_model_variables",
            "Recommendation": "W1: Season + z(NetworkDist_km). W3 exploratory: Season + z(NetworkDist_km) + z(DO_mg_L) + z(pH). Land-use W2 暂缓。",
            "Reason": f"NetworkDist_km 已由 corrected drainage network 计算；当前可用于 {water_distance_sites} 个水体点、{water_distance_rows} 条记录。DO/pH 覆盖较完整；缺少 water-site upstream contributing area/land-use ratios。",
        },
        {
            "Question": "B_High_collinearity",
            "Recommendation": "查看 diagnostics/*_spearman_pairs.csv 与 *_VIF.csv 中的标记。",
            "Reason": "阈值采用 |rho| > 0.70 与 VIF > 5；高相关对已逐项标记。",
        },
        {"Question": "C_COD", "Recommendation": cod_status, "Reason": cod_reason},
        {"Question": "D_DO_pH", "Recommendation": do_ph_status, "Reason": do_ph_reason},
        {"Question": "E_Chlorophyll", "Recommendation": chlor_status, "Reason": chlor_reason},
        {
            "Question": "F_Soil_model_variables",
            "Recommendation": "现有数据只能支持以 log_MFD 为水文候选变量的诊断；最终土壤主模型需先补齐 Dist_outlet 与土地利用比例。",
            "Reason": "Dist_outlet、HighAg_ratio、NaturalVeg_ratio、soil pH 缺失；DEM/Slope 为常数。",
        },
        {
            "Question": "G_Overfitting_risk",
            "Recommendation": "存在过拟合风险，尤其是 water 环境变量和 soil 多变量模型。",
            "Reason": f"Water 原始四季表有 {water_sites} 个独立点、{water_n} 条季节观测；其中有 NetworkDist_km 的为 {water_distance_sites} 个点、{water_distance_rows} 条记录。soil N={soil_n}。在缺失较多和候选变量相关时，应限制最终变量数。",
        },
    ]
    pd.DataFrame(rec_rows).to_csv(SUMMARY_DIR / "phase1_recommendations.csv", index=False, encoding="utf-8-sig")

    md = f"""# Water-Soil Multivariable Phase 1 Diagnostics

This phase intentionally stops before final mixed-effects / OLS-HC3 modelling.

## A. Water model recommendation

- Recommended baseline: `Season + z(NetworkDist_km)`.
- Recommended first water-environment candidate model: `Season + z(NetworkDist_km) + z(DO_mg_L) + z(pH)`.
- Do not run W2 yet: no water-site upstream contributing area or upstream land-use composition table was found.
- `NetworkDist_km` comes from the corrected runoff/drainage-network distance table, not Euclidean distance.

## B. High collinearity

Water high-correlation pairs:

{format_high(high_water)}

Soil high-correlation pairs:

{format_high(high_soil)}

VIF warnings are written in `water_VIF.csv` and `soil_VIF.csv`.

## C. COD

{cod_status}: {cod_reason}

## D. DO / pH

{do_ph_status}: {do_ph_reason}

## E. Chlorophyll

{chlor_status}: {chlor_reason}

## F. Soil model recommendation

- Current usable hydrological predictor: `{', '.join(soil_predictors)}`.
- Final requested soil model should wait until `Dist_outlet`, `HighAg_ratio`, and `NaturalVeg_ratio` are supplied or computed from appropriate hydrological/catchment data.
- {' '.join(soil_notes)}

## G. Overfitting risk

Water has {water_sites} site IDs and {water_n} seasonal rows in the scanned four-season records, but only
{water_distance_sites} sites and {water_distance_rows} rows currently have `NetworkDist_km`.
Soil has N={soil_n}.
Given this sample size, do not fit large models such as distance + multiple land-use ratios + DO + pH + COD + EC + Temp + Chlorophyll.

## Missing critical data

- Water upstream contributing area / upstream land-use proportions: not found.
- Soil hydrological distance to outlet (`Dist_outlet`): not found.
- Soil land-use proportions (`HighAg_ratio`, `NaturalVeg_ratio`): not found.
- Soil pH: not found.
"""
    (SUMMARY_DIR / "phase1_recommendations.md").write_text(md, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    water, water_distance = read_water_data()
    soil = read_soil_data()
    landuse = gpd.read_file(LANDUSE_PATH)

    water.to_csv(DIAG_DIR / "water_phase1_merged_data.csv", index=False, encoding="utf-8-sig")
    soil.to_csv(DIAG_DIR / "soil_phase1_attributes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"Water_distance_fields": list(water_distance.columns)}).to_csv(
        DIAG_DIR / "water_distance_fields.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"Landuse_fields": list(landuse.columns), "Note": "polygon layer; no upstream catchment proportions"}).to_csv(
        DIAG_DIR / "landuse_fields.csv", index=False, encoding="utf-8-sig"
    )

    water_vars = [
        ("TN_mg_L", "mg/L"),
        ("TP_mg_L", "mg/L"),
        ("NetworkDist_km", "km"),
        ("NetworkDist_m", "m"),
        ("DO_mg_L", "mg/L"),
        ("pH", "unitless"),
        ("COD_mg_L", "mg/L"),
        ("Chlor_ug_L", "ug/L"),
        ("Temp_C", "deg C"),
        ("TDS_mg_L", "mg/L"),
        ("EC", "unknown"),
        ("NH4_mg_L", "mg/L"),
        ("NO3_mg_L", "mg/L"),
        ("Cl_mg_L", "mg/L"),
        ("HighAg_ratio", "proportion"),
        ("NaturalVeg_ratio", "proportion"),
        ("Residential_ratio", "proportion"),
    ]
    soil_vars = [
        ("TN", "mg/kg"),
        ("TP", "mg/kg"),
        ("TOC", "g/kg"),
        ("NH4N", "mg/kg"),
        ("NO3N", "mg/kg"),
        ("Dist_outlet", "unknown"),
        ("MFD_accumulation", "flow accumulation"),
        ("log_MFD", "log1p(flow accumulation)"),
        ("Elevation", "m"),
        ("Slope", "unknown"),
        ("HighAg_ratio", "proportion"),
        ("NaturalVeg_ratio", "proportion"),
        ("soil_pH", "unitless"),
    ]

    variable_stats(water, water_vars).to_csv(DIAG_DIR / "water_descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    variable_stats(soil, soil_vars).to_csv(DIAG_DIR / "soil_descriptive_statistics.csv", index=False, encoding="utf-8-sig")

    field_presence(
        water,
        [
            "Site_ID",
            "Season",
            "TN_mg_L",
            "TP_mg_L",
            "Longitude",
            "Latitude",
            "NetworkDist_km",
            "pH",
            "DO_mg_L",
            "COD_mg_L",
            "Chlor_ug_L",
            "HighAg_ratio",
            "NaturalVeg_ratio",
            "Residential_ratio",
        ],
    ).to_csv(DIAG_DIR / "water_field_confirmation.csv", index=False, encoding="utf-8-sig")
    field_presence(
        soil,
        [
            "Site_ID",
            "Site_type",
            "Longitude",
            "Latitude",
            "TN",
            "TP",
            "TOC",
            "Dist_outlet",
            "MFD_accumulation",
            "log_MFD",
            "Elevation",
            "Slope",
            "HighAg_ratio",
            "NaturalVeg_ratio",
            "soil_pH",
        ],
    ).to_csv(DIAG_DIR / "soil_field_confirmation.csv", index=False, encoding="utf-8-sig")

    water_corr_vars = [
        "TN_mg_L",
        "TP_mg_L",
        "NetworkDist_km",
        "DO_mg_L",
        "pH",
        "COD_mg_L",
        "Chlor_ug_L",
        "Temp_C",
        "TDS_mg_L",
        "EC",
        "NH4_mg_L",
        "NO3_mg_L",
        "Cl_mg_L",
    ]
    soil_corr_vars = ["TN", "TP", "TOC", "NH4N", "NO3N", "Dist_outlet", "MFD_accumulation", "log_MFD", "Elevation", "Slope"]
    water_matrix, water_pairs = spearman_matrix(water, water_corr_vars)
    soil_matrix, soil_pairs = spearman_matrix(soil, soil_corr_vars)
    water_matrix.to_csv(DIAG_DIR / "water_spearman_matrix.csv", index=False, encoding="utf-8-sig")
    water_pairs.to_csv(DIAG_DIR / "water_spearman_pairs.csv", index=False, encoding="utf-8-sig")
    soil_matrix.to_csv(DIAG_DIR / "soil_spearman_matrix.csv", index=False, encoding="utf-8-sig")
    soil_pairs.to_csv(DIAG_DIR / "soil_spearman_pairs.csv", index=False, encoding="utf-8-sig")

    water_vif = pd.concat(
        [
            vif_for_set(water, ["NetworkDist_km", "DO_mg_L", "pH"], "water_core_no_COD"),
            vif_for_set(water, ["NetworkDist_km", "DO_mg_L", "pH", "COD_mg_L"], "water_core_with_COD"),
            vif_for_set(water, ["NetworkDist_km", "DO_mg_L", "pH", "COD_mg_L", "Chlor_ug_L"], "water_exploratory_with_Chlor"),
        ],
        ignore_index=True,
    )
    soil_vif = pd.concat(
        [
            vif_for_set(soil, ["Dist_outlet", "HighAg_ratio", "NaturalVeg_ratio", "log_MFD"], "soil_requested_core"),
            vif_for_set(soil, ["log_MFD", "Elevation", "Slope"], "soil_available_hydro_topo"),
        ],
        ignore_index=True,
    )
    water_vif.to_csv(DIAG_DIR / "water_VIF.csv", index=False, encoding="utf-8-sig")
    soil_vif.to_csv(DIAG_DIR / "soil_VIF.csv", index=False, encoding="utf-8-sig")

    write_recommendations(water, soil, water_pairs, soil_pairs, water_vif, soil_vif)

    manifest = pd.DataFrame(
        [
            {"File": str(path.relative_to(OUT_DIR)), "Rows": len(pd.read_csv(path)) if path.suffix == ".csv" else np.nan}
            for path in sorted(OUT_DIR.rglob("*"))
            if path.is_file()
        ]
    )
    manifest.to_csv(OUT_DIR / "phase1_output_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote phase-1 diagnostics to: {OUT_DIR}")


if __name__ == "__main__":
    main()
