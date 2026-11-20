from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import Voronoi
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

sys.path.append(str(Path(__file__).resolve().parent))
from water_soil_multivariable_phase2 import morans_i_knn  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "soil_water_measurements"
RESULTS_DIR = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime"
PHASE3_DIR = RESULTS_DIR / "water_soil_multivariable_phase3_soil_covariates"
OUT_DIR = RESULTS_DIR / "water_soil_multivariable_phase4_soil_final"

CLEANED_DIR = OUT_DIR / "cleaned"
DIAG_DIR = OUT_DIR / "diagnostics"
LANDUSE_DIR = OUT_DIR / "landuse"
DIST_DIR = OUT_DIR / "distance"
MODELS_DIR = OUT_DIR / "models"
SENS_DIR = OUT_DIR / "sensitivity"
SPATIAL_DIR = OUT_DIR / "spatial"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_DIR = OUT_DIR / "summary"
LOG_DIR = OUT_DIR / "experiment_log"

SOIL_POINTS_PATH = DATA_DIR / "土壤采样点" / "土壤采样点.shp"
LANDUSE_PATH = DATA_DIR / "土地利用类型" / "小流域土地利用类型.shp"
PHASE3_DISTANCE_PATH = PHASE3_DIR / "diagnostics" / "soil_distance_diagnostics.csv"

ALPHA = 0.05
LANDUSE_FIELD = "NAME"
RATIO_FIELD_MAP = {
    "大棚": "Greenhouse_ratio",
    "菜地": "Vegetable_ratio",
    "旱地": "Dryland_ratio",
    "水稻田": "Rice_ratio",
    "苗木地": "Nursery_ratio",
    "林地": "Forest_ratio",
    "灌木地": "Shrub_ratio",
    "竹林": "Bamboo_ratio",
    "草地": "Grass_ratio",
    "居民地": "Residential_ratio",
    "道路": "Road_ratio",
    "河流": "River_ratio",
    "湖泊": "Lake_ratio",
    "荒地": "Wasteland_ratio",
}


def ensure_dirs() -> None:
    for path in [
        CLEANED_DIR,
        DIAG_DIR,
        LANDUSE_DIR,
        DIST_DIR,
        MODELS_DIR,
        SENS_DIR,
        SPATIAL_DIR,
        FIG_DIR,
        SUMMARY_DIR,
        LOG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def finite_voronoi_polygons(vor: Voronoi, radius: float) -> Tuple[List[List[int]], np.ndarray]:
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi input must be 2D.")

    new_regions: List[List[int]] = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)

    all_ridges: Dict[int, List[Tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if vertices and all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            tangent = vor.points[p2] - vor.points[p1]
            tangent = tangent / np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[v] for v in new_region])
        centroid = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - centroid[1], vs[:, 0] - centroid[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def clean_geometry(geom):
    if geom is None or geom.is_empty:
        return geom
    try:
        fixed = geom.buffer(0)
        if fixed.is_empty:
            return geom
        return fixed
    except Exception:
        return geom


def find_existing_thiessen() -> pd.DataFrame:
    rows = []
    for path in sorted(list(ROOT.rglob("*.shp")) + list(ROOT.rglob("*.gpkg"))):
        if OUT_DIR in path.parents:
            continue
        low = str(path).lower()
        name_hit = any(
            token in low
            for token in [
                "thiessen",
                "voronoi",
                "泰森",
                "土壤采样点泰森多边形",
                "sampling polygon",
                "soil polygon",
            ]
        )
        if not name_hit:
            continue
        try:
            gdf = gpd.read_file(path)
            geom_types = "; ".join(sorted(gdf.geom_type.astype(str).unique()))
            polygon_like = any("Polygon" in item for item in gdf.geom_type.astype(str).unique())
            rows.append(
                {
                    "Path": str(path),
                    "FeatureCount": len(gdf),
                    "CRS": str(gdf.crs),
                    "GeometryTypes": geom_types,
                    "Fields": "; ".join([c for c in gdf.columns if c != "geometry"]),
                    "PolygonLike": polygon_like,
                }
            )
        except Exception as exc:
            rows.append({"Path": str(path), "FeatureCount": np.nan, "CRS": "", "GeometryTypes": "", "Fields": f"read_error: {exc}", "PolygonLike": False})
    out = pd.DataFrame(rows)
    out.to_csv(DIAG_DIR / "existing_thiessen_search.csv", index=False, encoding="utf-8-sig")
    return out


def create_thiessen() -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    soil = gpd.read_file(SOIL_POINTS_PATH)
    landuse = gpd.read_file(LANDUSE_PATH)
    if LANDUSE_FIELD not in landuse.columns:
        raise ValueError(f"Land-use field {LANDUSE_FIELD} is missing.")
    soil_proj = soil.to_crs(landuse.crs)
    soil_proj["Site_ID"] = soil_proj["样点"].astype(str)

    watershed_boundary = clean_geometry(unary_union([clean_geometry(g) for g in landuse.geometry]))
    watershed_area = watershed_boundary.area
    points = np.column_stack([soil_proj.geometry.x.to_numpy(), soil_proj.geometry.y.to_numpy()])
    span = max(points[:, 0].max() - points[:, 0].min(), points[:, 1].max() - points[:, 1].min())
    radius = max(span * 100, math.sqrt(watershed_area) * 10)
    vor = Voronoi(points)
    regions, vertices = finite_voronoi_polygons(vor, radius)

    records = []
    for idx, region in enumerate(regions):
        poly = Polygon(vertices[region])
        clipped = clean_geometry(poly.intersection(watershed_boundary))
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda g: g.area)
        records.append(
            {
                "Site_ID": soil_proj.iloc[idx]["Site_ID"],
                "Input_FID": int(soil_proj.index[idx]),
                "ThiessArea": clipped.area if clipped and not clipped.is_empty else 0.0,
                "geometry": clipped,
            }
        )
    thiessen = gpd.GeoDataFrame(records, geometry="geometry", crs=landuse.crs)
    thiessen_path = LANDUSE_DIR / "soil_thiessen_47.shp"
    clear_vector(thiessen_path)
    thiessen.to_file(thiessen_path, encoding="utf-8")

    union_cells = clean_geometry(unary_union(thiessen.geometry))
    intersection_area = union_cells.intersection(watershed_boundary).area
    overlap_area = max(thiessen["ThiessArea"].sum() - union_cells.area, 0.0)
    gap_area = max(watershed_area - intersection_area, 0.0)
    qa = pd.DataFrame(
        [
            {
                "Source": "created_from_47_soil_points",
                "Polygon_N": len(thiessen),
                "TotalPolygonArea_m2": thiessen["ThiessArea"].sum(),
                "WatershedArea_m2": watershed_area,
                "CoverageRatio": intersection_area / watershed_area if watershed_area else np.nan,
                "OverlapArea_m2": overlap_area,
                "GapArea_m2": gap_area,
                "GapRatio": gap_area / watershed_area if watershed_area else np.nan,
                "BoundarySource": "union of land-use polygon layer, not convex hull",
            }
        ]
    )
    qa.to_csv(LANDUSE_DIR / "soil_thiessen_47_QA.csv", index=False, encoding="utf-8-sig")
    return thiessen, qa


def clear_vector(path: Path) -> None:
    for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".qix"]:
        target = path.with_suffix(suffix)
        if target.exists():
            target.unlink()


def calculate_landuse_ratios(thiessen: gpd.GeoDataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    landuse = gpd.read_file(LANDUSE_PATH).to_crs(thiessen.crs)
    landuse[LANDUSE_FIELD] = landuse[LANDUSE_FIELD].astype(str)
    categories = sorted(landuse[LANDUSE_FIELD].dropna().astype(str).unique())
    pd.DataFrame({"CategoryField": LANDUSE_FIELD, "LandUseCategory": categories}).to_csv(
        LANDUSE_DIR / "landuse_NAME_categories.csv", index=False, encoding="utf-8-sig"
    )
    mapping_rows = []
    for category in categories:
        mapping_rows.append(
            {
                "NAME": category,
                "RatioField": RATIO_FIELD_MAP.get(category, f"LU_{category}_ratio"),
            }
        )
    pd.DataFrame(mapping_rows).to_csv(LANDUSE_DIR / "landuse_ratio_field_mapping.csv", index=False, encoding="utf-8-sig")

    area_rows = []
    lu_geoms = [(row[LANDUSE_FIELD], clean_geometry(row.geometry)) for _, row in landuse.iterrows()]
    for _, cell in thiessen.iterrows():
        cell_geom = clean_geometry(cell.geometry)
        site_id = cell["Site_ID"]
        thiessen_area = cell_geom.area
        by_category = {category: 0.0 for category in categories}
        for category, lu_geom in lu_geoms:
            if not cell_geom.intersects(lu_geom):
                continue
            area = cell_geom.intersection(lu_geom).area
            if area > 0:
                by_category[str(category)] += area
        row = {"Site_ID": site_id, "Thiessen_area": thiessen_area}
        covered = 0.0
        for category in categories:
            ratio_field = RATIO_FIELD_MAP.get(category, f"LU_{category}_ratio")
            area = by_category[category]
            covered += area
            row[f"{ratio_field.replace('_ratio', '')}_area_m2"] = area
            row[ratio_field] = area / thiessen_area if thiessen_area else np.nan
        row["covered_landuse_area"] = covered
        row["coverage_fraction"] = covered / thiessen_area if thiessen_area else np.nan
        area_rows.append(row)

    raw = pd.DataFrame(area_rows)
    ratio_cols = [c for c in raw.columns if c.endswith("_ratio") or c.startswith("LU_")]
    raw["ratio_sum"] = raw[ratio_cols].sum(axis=1)
    raw.to_csv(LANDUSE_DIR / "soil_landuse_raw_ratios.csv", index=False, encoding="utf-8-sig")

    qa = raw[["Site_ID", "ratio_sum", "Thiessen_area", "covered_landuse_area", "coverage_fraction"]].copy()
    qa["warning"] = np.where((qa["ratio_sum"] < 0.98) | (qa["ratio_sum"] > 1.02), "ratio_sum_outside_0.98_1.02", "")
    qa.to_csv(LANDUSE_DIR / "landuse_ratio_QA.csv", index=False, encoding="utf-8-sig")

    grouped = pd.DataFrame({"Site_ID": raw["Site_ID"]})
    grouped["HighAg_ratio"] = raw.get("Greenhouse_ratio", 0) + raw.get("Vegetable_ratio", 0)
    grouped["NaturalVeg_ratio"] = (
        raw.get("Forest_ratio", 0) + raw.get("Shrub_ratio", 0) + raw.get("Bamboo_ratio", 0) + raw.get("Grass_ratio", 0)
    )
    grouped["ModerateAg_ratio"] = raw.get("Dryland_ratio", 0) + raw.get("Rice_ratio", 0) + raw.get("Nursery_ratio", 0)
    grouped["Built_ratio"] = raw.get("Residential_ratio", 0) + raw.get("Road_ratio", 0)
    grouped["Water_ratio"] = raw.get("River_ratio", 0) + raw.get("Lake_ratio", 0)
    grouped.to_csv(LANDUSE_DIR / "soil_landuse_grouped_ratios.csv", index=False, encoding="utf-8-sig")
    return raw, grouped, qa


def build_covariates(raw: pd.DataFrame, grouped: pd.DataFrame) -> pd.DataFrame:
    soil = gpd.read_file(SOIL_POINTS_PATH)
    soil["Site_ID"] = soil["样点"].astype(str)
    attrs = pd.DataFrame(soil.drop(columns="geometry", errors="ignore"))
    attrs = attrs.rename(columns={"BurnDEMMFD": "MFD_accumulation", "No3N": "NO3N"})
    attrs["log_MFD"] = np.log1p(pd.to_numeric(attrs["MFD_accumulation"], errors="coerce").clip(lower=0))

    dist = pd.read_csv(PHASE3_DISTANCE_PATH, dtype={"Site_ID": str})
    dist = dist.rename(columns={"FinalDist_m": "HybridDist_m"})
    dist["FinalDist_m"] = dist["HybridDist_m"]

    cov = attrs.merge(dist, on="Site_ID", how="left").merge(raw, on="Site_ID", how="left").merge(grouped, on="Site_ID", how="left")
    cov.to_csv(CLEANED_DIR / "soil_covariates_complete.csv", index=False, encoding="utf-8-sig")
    cov.to_csv(OUT_DIR / "soil_covariates_complete.csv", index=False, encoding="utf-8-sig")

    fallback = cov[cov["DistanceMethod"] == "euclidean_fallback"][
        ["Site_ID", "SnapDist_m", "EuclideanDist_m", "NetworkDist_m", "Orig_X", "Orig_Y"]
    ].copy()
    fallback["reason_for_failure"] = "not connected to outlet in corrected drainage network"
    fallback = fallback.rename(columns={"Orig_X": "X", "Orig_Y": "Y"})
    fallback.to_csv(DIST_DIR / "soil_euclidean_fallback_sites.csv", index=False, encoding="utf-8-sig")

    copy_distance_outputs(dist)
    write_covariate_shp(cov)
    return cov


def copy_distance_outputs(dist: pd.DataFrame) -> None:
    dist.to_csv(DIST_DIR / "soil_distance_diagnostics.csv", index=False, encoding="utf-8-sig")
    n_total = len(dist)
    n_network = int((dist["DistanceMethod"] == "network").sum())
    n_fallback = int((dist["DistanceMethod"] == "euclidean_fallback").sum())
    summary = pd.DataFrame(
        [
            {
                "N_total": n_total,
                "N_network": n_network,
                "N_euclidean_fallback": n_fallback,
                "Network_percentage": n_network / n_total,
                "Fallback_percentage": n_fallback / n_total,
                "SnapDist_mean_m": dist["SnapDist_m"].mean(),
                "SnapDist_median_m": dist["SnapDist_m"].median(),
                "SnapDist_max_m": dist["SnapDist_m"].max(),
                "MaxSnapDist_Site_ID": dist.loc[dist["SnapDist_m"].idxmax(), "Site_ID"],
            }
        ]
    )
    summary.to_csv(DIST_DIR / "soil_distance_summary.csv", index=False, encoding="utf-8-sig")


def write_covariate_shp(cov: pd.DataFrame) -> None:
    soil = gpd.read_file(SOIL_POINTS_PATH)
    soil["Site_ID"] = soil["样点"].astype(str)
    cols = [
        "Site_ID",
        "NetworkDist_m",
        "EuclideanDist_m",
        "HybridDist_m",
        "DistanceMethod",
        "HighAg_ratio",
        "NaturalVeg_ratio",
        "ModerateAg_ratio",
        "Built_ratio",
        "Water_ratio",
        "log_MFD",
    ]
    shp = soil[["Site_ID", "geometry"]].merge(cov[cols], on="Site_ID", how="left")
    shp = shp.rename(
        columns={
            "NetworkDist_m": "NetDist_m",
            "EuclideanDist_m": "EucDist_m",
            "HybridDist_m": "HybDist_m",
            "DistanceMethod": "DistMeth",
            "HighAg_ratio": "HighAg",
            "NaturalVeg_ratio": "NatVeg",
            "ModerateAg_ratio": "ModAg",
            "Built_ratio": "Built",
            "Water_ratio": "Water",
        }
    )
    out_path = CLEANED_DIR / "soil_sites_with_covariates.shp"
    clear_vector(out_path)
    shp.to_file(out_path, encoding="utf-8")
    pd.DataFrame(
        [
            {"ShapefileField": "NetDist_m", "FullMeaning": "NetworkDist_m"},
            {"ShapefileField": "EucDist_m", "FullMeaning": "EuclideanDist_m"},
            {"ShapefileField": "HybDist_m", "FullMeaning": "HybridDist_m / FinalDist_m"},
            {"ShapefileField": "DistMeth", "FullMeaning": "DistanceMethod"},
            {"ShapefileField": "HighAg", "FullMeaning": "HighAg_ratio"},
            {"ShapefileField": "NatVeg", "FullMeaning": "NaturalVeg_ratio"},
            {"ShapefileField": "ModAg", "FullMeaning": "ModerateAg_ratio"},
        ]
    ).to_csv(CLEANED_DIR / "soil_sites_with_covariates_field_mapping.csv", index=False, encoding="utf-8-sig")


def spearman_matrix(df: pd.DataFrame, vars_: List[str], out_path: Path) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(vars_):
        for b in vars_[i + 1 :]:
            xa = pd.to_numeric(df.get(a), errors="coerce")
            xb = pd.to_numeric(df.get(b), errors="coerce")
            valid = xa.notna() & xb.notna()
            if valid.sum() >= 3 and xa[valid].nunique() > 1 and xb[valid].nunique() > 1:
                rho, p = stats.spearmanr(xa[valid], xb[valid])
            else:
                rho, p = np.nan, np.nan
            rows.append(
                {
                    "Variable_1": a,
                    "Variable_2": b,
                    "N_pairwise": int(valid.sum()),
                    "Spearman_rho": rho,
                    "P_value": p,
                    "High_correlation_abs_rho_gt_0_70": bool(pd.notna(rho) and abs(rho) > 0.70),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def vif_table(df: pd.DataFrame, variables: List[str], label: str) -> pd.DataFrame:
    present = []
    rows = []
    for var in variables:
        x = pd.to_numeric(df.get(var), errors="coerce")
        if x.notna().sum() < 3:
            rows.append({"Set": label, "Variable": var, "VIF": np.nan, "N_complete": int(x.notna().sum()), "Status": "too few non-missing values"})
        elif x.dropna().nunique() <= 1:
            rows.append({"Set": label, "Variable": var, "VIF": np.nan, "N_complete": int(x.notna().sum()), "Status": "constant"})
        else:
            present.append(var)
    X = pd.DataFrame({v: pd.to_numeric(df[v], errors="coerce") for v in present}).dropna()
    if len(present) < 2 or len(X) <= len(present) + 1:
        for var in present:
            rows.append({"Set": label, "Variable": var, "VIF": np.nan, "N_complete": len(X), "Status": "not computed: too few usable variables/cases"})
        return pd.DataFrame(rows)
    X = (X - X.mean()) / X.std(ddof=0)
    for var in present:
        y = X[var].to_numpy()
        other = X[[c for c in present if c != var]].to_numpy()
        design = np.column_stack([np.ones(len(X)), other])
        beta = np.linalg.pinv(design.T @ design) @ design.T @ y
        pred = design @ beta
        rss = float(np.sum((y - pred) ** 2))
        tss = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else np.nan
        vif = np.inf if pd.notna(r2) and r2 >= 0.999999 else (1.0 / (1.0 - r2) if pd.notna(r2) else np.nan)
        rows.append({"Set": label, "Variable": var, "VIF": vif, "N_complete": len(X), "Status": "warning: VIF > 5" if pd.notna(vif) and vif > 5 else "ok"})
    return pd.DataFrame(rows)


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd


def fit_ols_hc3(df: pd.DataFrame, response: str, predictors: List[str], model_name: str, pollutant: str) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    model_df = df[["Site_ID", response] + predictors].copy()
    for col in [response] + predictors:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df.dropna().copy()
    model_df = model_df[model_df[response] >= 0].copy()
    model_df["log_response"] = np.log1p(model_df[response])
    terms = ["Intercept"]
    parts = [np.ones(len(model_df))]
    for pred in predictors:
        z_col = f"z_{pred}"
        model_df[z_col] = zscore(model_df[pred])
        terms.append(z_col)
        parts.append(model_df[z_col].to_numpy())
    model_df = model_df.dropna(subset=[f"z_{p}" for p in predictors]).copy()
    parts = [np.ones(len(model_df))] + [model_df[f"z_{p}"].to_numpy() for p in predictors]
    X = np.column_stack(parts)
    y = model_df["log_response"].to_numpy()
    n, k = X.shape
    if n <= k + 1:
        raise ValueError(f"Too few cases for {model_name} {pollutant}: n={n}, k={k}")
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    fitted = X @ beta
    resid = y - fitted
    xtx_inv = np.linalg.pinv(X.T @ X)
    hat = np.sum(X * (X @ xtx_inv), axis=1)
    scale = resid**2 / np.maximum((1.0 - hat) ** 2, 1e-12)
    cov = xtx_inv @ (X.T @ (X * scale[:, None])) @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    df_resid = max(n - k, 1)
    tcrit = stats.t.ppf(1 - ALPHA / 2, df=df_resid)
    t_values = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_values), df=df_resid)
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if pd.notna(r2) and n > k else np.nan
    sigma2 = rss / n
    loglike = -0.5 * n * (math.log(2 * math.pi) + math.log(sigma2) + 1) if sigma2 > 0 else np.nan
    aic = 2 * k - 2 * loglike if pd.notna(loglike) else np.nan
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if pd.notna(aic) and n > k + 1 else np.nan

    rows = []
    for term, estimate, term_se, t_value, p_value in zip(terms, beta, se, t_values, p_values):
        rows.append(
            {
                "Model": model_name,
                "Pollutant": pollutant,
                "Response": f"log1p({response})",
                "Term": term,
                "Std_Beta": estimate,
                "SE": term_se,
                "CI_low": estimate - tcrit * term_se,
                "CI_high": estimate + tcrit * term_se,
                "t_value": t_value,
                "P_value": p_value,
                "N": n,
                "R2": r2,
                "Adjusted_R2": adj_r2,
                "AIC": aic,
                "AICc": aicc,
                "Method": "OLS-HC3",
            }
        )
    model_df["fitted"] = fitted
    model_df["residual"] = resid
    return pd.DataFrame(rows), {"Model": model_name, "Pollutant": pollutant, "N": n, "R2": r2, "Adjusted_R2": adj_r2, "AIC": aic, "AICc": aicc}, model_df


def run_models(cov: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    distance_specs = {
        "D1_Euclidean": ("EuclideanDist_m", cov.copy()),
        "D2_NetworkOnly": ("NetworkDist_m", cov[cov["DistanceMethod"] == "network"].copy()),
        "D3_Hybrid": ("HybridDist_m", cov.copy()),
    }
    pollutants = [("TN", "TN"), ("TP", "TP"), ("TOC", "TOC")]
    all_tables = []
    comparison = []
    residuals: Dict[str, pd.DataFrame] = {}
    for model_name, (distance_col, data) in distance_specs.items():
        predictors = [distance_col, "HighAg_ratio", "NaturalVeg_ratio", "log_MFD"]
        vif = vif_table(data, predictors, model_name)
        vif.to_csv(DIAG_DIR / f"soil_predictor_VIF_{model_name}.csv", index=False, encoding="utf-8-sig")
        for pollutant, response in pollutants:
            table, comp, model_df = fit_ols_hc3(data, response, predictors, model_name, pollutant)
            table.to_csv(MODELS_DIR / f"soil_model_{pollutant}_{model_name}.csv", index=False, encoding="utf-8-sig")
            all_tables.append(table)
            comparison.append(comp)
            residuals[f"{pollutant}_{model_name}"] = model_df

    model_results = pd.concat(all_tables, ignore_index=True)
    model_results.to_csv(SUMMARY_DIR / "soil_distance_definition_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparison).to_csv(SUMMARY_DIR / "soil_model_comparison.csv", index=False, encoding="utf-8-sig")

    main = model_results[model_results["Model"] == "D3_Hybrid"].copy()
    for pollutant in ["TN", "TP", "TOC"]:
        main[main["Pollutant"] == pollutant].to_csv(MODELS_DIR / f"soil_model_{pollutant}_full.csv", index=False, encoding="utf-8-sig")
    main.to_csv(SUMMARY_DIR / "soil_fullmodel_coefficients.csv", index=False, encoding="utf-8-sig")
    return model_results, pd.DataFrame(comparison), residuals


def run_supplementary_model(cov: pd.DataFrame) -> None:
    predictors = ["HybridDist_m", "HighAg_ratio", "ModerateAg_ratio", "NaturalVeg_ratio", "log_MFD"]
    vif = vif_table(cov, predictors, "Supplementary_ModerateAg")
    vif.to_csv(DIAG_DIR / "soil_predictor_VIF_supplementary_moderateAg.csv", index=False, encoding="utf-8-sig")
    if vif["Status"].astype(str).str.contains("warning", na=False).any():
        pd.DataFrame(
            [
                {
                    "Model": "Supplementary_ModerateAg",
                    "Status": "not_run_due_to_VIF_warning",
                    "Reason": "The user requested this supplementary model only when VIF is acceptable. Candidate VIF diagnostics exceeded 5.",
                }
            ]
        ).to_csv(MODELS_DIR / "soil_supplementary_moderateAg_model.csv", index=False, encoding="utf-8-sig")
        return
    rows = []
    for pollutant in ["TN", "TP", "TOC"]:
        table, _, _ = fit_ols_hc3(cov, pollutant, predictors, "Supplementary_ModerateAg", pollutant)
        rows.append(table)
    out = pd.concat(rows, ignore_index=True)
    out["SupplementaryStatus"] = "run"
    out.to_csv(MODELS_DIR / "soil_supplementary_moderateAg_model.csv", index=False, encoding="utf-8-sig")


def run_s6_sensitivity(cov: pd.DataFrame) -> pd.DataFrame:
    s6 = cov[cov["Site_ID"] == "S6"].copy()
    s6.to_csv(SENS_DIR / "S6_record_check.csv", index=False, encoding="utf-8-sig")
    rows = []
    for dataset_name, data in [("Full_N47", cov.copy()), ("Exclude_S6", cov[cov["Site_ID"] != "S6"].copy())]:
        for pollutant in ["TN", "TP", "TOC"]:
            table, _, _ = fit_ols_hc3(data, pollutant, ["HybridDist_m", "HighAg_ratio", "NaturalVeg_ratio", "log_MFD"], dataset_name, pollutant)
            table["Dataset"] = dataset_name
            rows.append(table)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(SENS_DIR / "soil_S6_sensitivity.csv", index=False, encoding="utf-8-sig")
    return out


def run_morans(cov: pd.DataFrame, residuals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    soil = gpd.read_file(SOIL_POINTS_PATH).to_crs(epsg=32650)
    coords = pd.DataFrame({"Site_ID": gpd.read_file(SOIL_POINTS_PATH)["样点"].astype(str), "x": soil.geometry.x, "y": soil.geometry.y})
    rows = []
    for pollutant in ["TN", "TP", "TOC"]:
        key = f"{pollutant}_D3_Hybrid"
        model_df = residuals[key][["Site_ID", "residual"]].merge(coords, on="Site_ID", how="inner")
        result = morans_i_knn(model_df[["x", "y"]].to_numpy(), model_df["residual"].to_numpy(), k=8)
        result.update({"Pollutant": pollutant, "Model": "D3_Hybrid_full", "CoordinateBasis": "EPSG:32650"})
        rows.append(result)
    out = pd.DataFrame(rows).rename(columns={"P_value": "Permutation_p"})
    out.to_csv(SPATIAL_DIR / "soil_fullmodel_residual_moransI.csv", index=False, encoding="utf-8-sig")
    return out


def predictor_label(term: str) -> str:
    mapping = {
        "z_EuclideanDist_m": "Hydrological distance",
        "z_NetworkDist_m": "Hydrological distance",
        "z_HybridDist_m": "Hydrological distance",
        "z_HighAg_ratio": "High-intensity agriculture",
        "z_NaturalVeg_ratio": "Natural vegetation",
        "z_log_MFD": "Flow accumulation",
    }
    return mapping.get(term, term)


def make_forest_plot(main: pd.DataFrame) -> None:
    plot_df = main[main["Term"].isin(["z_HybridDist_m", "z_HighAg_ratio", "z_NaturalVeg_ratio", "z_log_MFD"])].copy()
    plot_df["Predictor"] = plot_df["Term"].map(predictor_label)
    plot_df["Label"] = plot_df["Pollutant"] + " | " + plot_df["Predictor"]
    order = []
    for pollutant in ["TN", "TP", "TOC"]:
        for term in ["z_HybridDist_m", "z_HighAg_ratio", "z_NaturalVeg_ratio", "z_log_MFD"]:
            row = plot_df[(plot_df["Pollutant"] == pollutant) & (plot_df["Term"] == term)]
            if not row.empty:
                order.append(row.index[0])
    plot_df = plot_df.loc[order].copy()
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    colors = plot_df["Pollutant"].map({"TN": "#1f77b4", "TP": "#2ca02c", "TOC": "#9467bd"})
    ax.axvline(0, color="0.25", linewidth=1)
    ax.errorbar(
        plot_df["Std_Beta"],
        y,
        xerr=[plot_df["Std_Beta"] - plot_df["CI_low"], plot_df["CI_high"] - plot_df["Std_Beta"]],
        fmt="none",
        ecolor="0.35",
        capsize=3,
        linewidth=1.4,
    )
    ax.scatter(plot_df["Std_Beta"], y, c=colors, s=42, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Label"])
    ax.invert_yaxis()
    ax.set_xlabel("Standardized coefficient estimate (log-response scale)")
    ax.set_title("Soil multivariable models")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Figure_soil_multivariable_phase4.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "Figure_soil_multivariable_phase4.pdf", bbox_inches="tight")
    plt.close(fig)


def write_phase4_summary(
    existing: pd.DataFrame,
    thiessen_qa: pd.DataFrame,
    lu_qa: pd.DataFrame,
    cov: pd.DataFrame,
    spearman: pd.DataFrame,
    vif_main: pd.DataFrame,
    model_results: pd.DataFrame,
    moran: pd.DataFrame,
) -> None:
    dist = pd.read_csv(DIST_DIR / "soil_distance_summary.csv")
    fallback = pd.read_csv(DIST_DIR / "soil_euclidean_fallback_sites.csv")
    cats = pd.read_csv(LANDUSE_DIR / "landuse_NAME_categories.csv")
    max_snap_site = dist.loc[0, "MaxSnapDist_Site_ID"]
    high_corr = spearman[spearman["High_correlation_abs_rho_gt_0_70"] == True]  # noqa: E712
    high_corr_text = "None"
    if not high_corr.empty:
        high_corr_text = "\n".join(
            f"- {r.Variable_1} vs {r.Variable_2}: rho={r.Spearman_rho:.3f}, N={r.N_pairwise}, p={r.P_value:.4g}"
            for r in high_corr.itertuples(index=False)
        )

    def distribution_line(col: str) -> str:
        x = pd.to_numeric(cov[col], errors="coerce")
        return f"{col}: min={x.min():.3f}, median={x.median():.3f}, mean={x.mean():.3f}, max={x.max():.3f}"

    md = f"""---
exp_id: HY-B-260826-004
title: Phase 4 soil covariate completion and formal multivariable models
date: 2026-08-26
created_at: "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
system: HY
device: B
type: statistical_analysis
status: completed
outputs_root: results/water_soil_multivariable_phase4_soil_final
---

# Phase 4 Diagnostic And Model Summary

## Existing Thiessen Search

- Existing Thiessen/Voronoi polygon found: {not existing.empty}
- If not found, created `soil_thiessen_47.shp` from all 47 soil points.
- Clip boundary: union of `小流域土地利用类型.shp`, not convex hull.

## Thiessen QA

- Polygon N: {int(thiessen_qa.loc[0, 'Polygon_N'])}
- Total polygon area: {thiessen_qa.loc[0, 'TotalPolygonArea_m2']:.3f} m2
- Watershed area: {thiessen_qa.loc[0, 'WatershedArea_m2']:.3f} m2
- Coverage ratio: {thiessen_qa.loc[0, 'CoverageRatio']:.6f}
- Gap ratio: {thiessen_qa.loc[0, 'GapRatio']:.6g}

## Distance QA

- Network distance successful: {int(dist.loc[0, 'N_network'])}/47.
- Euclidean fallback: {int(dist.loc[0, 'N_euclidean_fallback'])}/47.
- Maximum SnapDist: {dist.loc[0, 'SnapDist_max_m']:.3f} m, Site_ID = {max_snap_site}.

Fallback sites:

{fallback[['Site_ID', 'SnapDist_m', 'EuclideanDist_m']].to_markdown(index=False)}

## Land-Use Categories

{chr(10).join('- ' + v for v in cats['LandUseCategory'].astype(str).tolist())}

## Ratio QA

- Sites with ratio_sum outside [0.98, 1.02]: {(lu_qa['warning'] != '').sum()}
- Mean coverage fraction: {lu_qa['coverage_fraction'].mean():.6f}

## Grouped Ratio Distribution

- {distribution_line('HighAg_ratio')}
- {distribution_line('NaturalVeg_ratio')}
- {distribution_line('ModerateAg_ratio')}

## Predictor Correlation

High-correlation pairs:

{high_corr_text}

## Main Predictor VIF

{vif_main.to_markdown(index=False)}

## Model Outputs

For file naming continuity, `*_full.csv` uses D3 Hybrid distance for full 47-site models, but this is not a decision that D3 should be the manuscript-final distance definition:

- `models/soil_model_TN_full.csv`
- `models/soil_model_TP_full.csv`
- `models/soil_model_TOC_full.csv`

Distance-definition sensitivity across D1/D2/D3 is in `summary/soil_distance_definition_sensitivity.csv`.

## Moran's I

{moran.to_markdown(index=False)}

## Limitations

- Thiessen polygons were newly generated because no existing representative polygons were found.
- The watershed boundary was represented by the union of the available land-use polygon layer.
- D1/D2/D3 are provided for comparison; this script does not choose the most significant distance definition as the main scientific conclusion.
- Water models were not modified in Phase 4.
"""
    (SUMMARY_DIR / "phase4_diagnostic_summary.md").write_text(md, encoding="utf-8")

    record = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_experiment_record.md"
    marker = "\n\n## 14. Phase 4: Soil Covariate Completion and Formal Multivariable Models\n\n"
    phase_text = (
        marker
        + "Output root: `results/water_soil_multivariable_phase4_soil_final`\n\n"
        + f"- Existing Thiessen polygons found: {not existing.empty}.\n"
        + "- New Thiessen polygons were created from all 47 soil sites and clipped by the land-use layer union boundary.\n"
        + f"- Land-use categories: {', '.join(cats['LandUseCategory'].astype(str).tolist())}.\n"
        + f"- Network/fallback distance: {int(dist.loc[0, 'N_network'])}/47 network, {int(dist.loc[0, 'N_euclidean_fallback'])}/47 Euclidean fallback.\n"
        + f"- Max SnapDist: {dist.loc[0, 'SnapDist_max_m']:.3f} m at Site_ID {max_snap_site}.\n"
        + "- D1/D2/D3 distance sensitivity, D3 all-site `*_full.csv` outputs, S6 sensitivity, Moran's I and forest plot were generated; no final distance definition was selected.\n"
        + "- Water models were not modified.\n"
    )
    if record.exists():
        text = record.read_text(encoding="utf-8")
        if marker in text:
            text = text.split(marker)[0].rstrip() + phase_text
        else:
            text = text.rstrip() + phase_text
        record.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    existing = find_existing_thiessen()
    thiessen, thiessen_qa = create_thiessen()
    raw, grouped, lu_qa = calculate_landuse_ratios(thiessen)
    cov = build_covariates(raw, grouped)

    predictor_vars = [
        "EuclideanDist_m",
        "NetworkDist_m",
        "HybridDist_m",
        "HighAg_ratio",
        "NaturalVeg_ratio",
        "ModerateAg_ratio",
        "Built_ratio",
        "Water_ratio",
        "log_MFD",
    ]
    spearman = spearman_matrix(cov, predictor_vars, DIAG_DIR / "soil_predictor_spearman.csv")
    vif_main = vif_table(cov, ["HybridDist_m", "HighAg_ratio", "NaturalVeg_ratio", "log_MFD"], "D3_Hybrid_main")
    vif_exploratory = vif_table(
        cov,
        ["HybridDist_m", "HighAg_ratio", "ModerateAg_ratio", "NaturalVeg_ratio", "Built_ratio", "Water_ratio", "log_MFD"],
        "exploratory_candidates",
    )
    pd.concat([vif_main, vif_exploratory], ignore_index=True).to_csv(DIAG_DIR / "soil_predictor_VIF.csv", index=False, encoding="utf-8-sig")

    model_results, comparison, residuals = run_models(cov)
    run_supplementary_model(cov)
    run_s6_sensitivity(cov)
    moran = run_morans(cov, residuals)
    main = model_results[model_results["Model"] == "D3_Hybrid"].copy()
    make_forest_plot(main)
    write_phase4_summary(existing, thiessen_qa, lu_qa, cov, spearman, vif_main, model_results, moran)

    manifest = []
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            manifest.append({"File": str(path.relative_to(OUT_DIR)), "Bytes": path.stat().st_size})
    pd.DataFrame(manifest).to_csv(OUT_DIR / "phase4_output_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote phase-4 soil outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
