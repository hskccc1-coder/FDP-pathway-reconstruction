from __future__ import annotations

import heapq
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "soil_water_measurements"
RESULTS_DIR = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime"
OUT_DIR = RESULTS_DIR / "water_soil_multivariable_phase3_soil_covariates"
DIAG_DIR = OUT_DIR / "diagnostics"
GIS_DIR = OUT_DIR / "gis"
SENS_DIR = OUT_DIR / "sensitivity"
SUMMARY_DIR = OUT_DIR / "summary"
LOG_DIR = OUT_DIR / "experiment_log"

SOIL_POINTS_PATH = DATA_DIR / "土壤采样点" / "土壤采样点.shp"
RIVER_PATH = DATA_DIR / "矫正后河流线" / "river.shp"
OUTLET_PATH = RESULTS_DIR / "figure3_network_distance_stage1" / "outlet_snapped.shp"
LANDUSE_PATH = DATA_DIR / "土地利用类型" / "小流域土地利用类型.shp"
MFD_CANDIDATES = [
    DATA_DIR / "单位侵蚀量" / "水土总氮多流向总氮1.tif",
]

NODE_PRECISION = 3
SNAP_WARNING_THRESHOLDS_M = (5.0, 10.0, 20.0, 50.0, 100.0)
FALLBACK_WARNING_THRESHOLD = 0.20


def ensure_dirs() -> None:
    for path in [DIAG_DIR, GIS_DIR, SENS_DIR, SUMMARY_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def geometry_lines(geometry) -> List[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, (MultiLineString, GeometryCollection)):
        lines: List[LineString] = []
        for part in geometry.geoms:
            lines.extend(geometry_lines(part))
        return lines
    return []


def node_key_xy(x: float, y: float) -> Tuple[float, float]:
    return (round(float(x), NODE_PRECISION), round(float(y), NODE_PRECISION))


def node_key(point: Point) -> Tuple[float, float]:
    return node_key_xy(point.x, point.y)


def crs_unit_name(gdf: gpd.GeoDataFrame) -> str:
    try:
        return str(gdf.crs.axis_info[0].unit_name)
    except Exception:
        return "unknown"


def add_edge(adjacency: Dict[Tuple[float, float], Dict[Tuple[float, float], float]], a, b, length: float) -> None:
    if a == b or length <= 0 or pd.isna(length):
        return
    current = adjacency[a].get(b)
    if current is None or length < current:
        adjacency[a][b] = float(length)
        adjacency[b][a] = float(length)


def build_base_graph(lines: List[LineString]) -> Dict[Tuple[float, float], Dict[Tuple[float, float], float]]:
    adjacency: Dict[Tuple[float, float], Dict[Tuple[float, float], float]] = defaultdict(dict)
    for line in lines:
        coords = list(line.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            ak = node_key_xy(a[0], a[1])
            bk = node_key_xy(b[0], b[1])
            add_edge(adjacency, ak, bk, Point(a).distance(Point(b)))
    return adjacency


def nearest_line(point: Point, lines: List[LineString]) -> Tuple[int, LineString, Point, float, float]:
    best_idx = -1
    best_line = None
    best_point = None
    best_measure = math.nan
    best_dist = math.inf
    for idx, line in enumerate(lines):
        measure = line.project(point)
        snapped = line.interpolate(measure)
        dist = point.distance(snapped)
        if dist < best_dist:
            best_idx = idx
            best_line = line
            best_point = snapped
            best_measure = measure
            best_dist = dist
    if best_line is None or best_point is None:
        raise RuntimeError("Could not snap point to any river line.")
    return best_idx, best_line, best_point, best_measure, best_dist


def cumulative_distances(coords: List[Tuple[float, float]]) -> List[float]:
    distances = [0.0]
    total = 0.0
    for a, b in zip(coords[:-1], coords[1:]):
        total += Point(a).distance(Point(b))
        distances.append(total)
    return distances


def connect_snapped_point(
    adjacency: Dict[Tuple[float, float], Dict[Tuple[float, float], float]],
    line: LineString,
    snapped: Point,
    measure: float,
) -> Tuple[float, float]:
    coords = list(line.coords)
    cum = cumulative_distances(coords)
    key = node_key(snapped)

    if measure <= 1e-9:
        first = node_key_xy(coords[0][0], coords[0][1])
        add_edge(adjacency, key, first, snapped.distance(Point(coords[0])))
        return key
    if measure >= cum[-1] - 1e-9:
        last = node_key_xy(coords[-1][0], coords[-1][1])
        add_edge(adjacency, key, last, snapped.distance(Point(coords[-1])))
        return key

    for idx in range(len(cum) - 1):
        if cum[idx] - 1e-9 <= measure <= cum[idx + 1] + 1e-9:
            left = node_key_xy(coords[idx][0], coords[idx][1])
            right = node_key_xy(coords[idx + 1][0], coords[idx + 1][1])
            add_edge(adjacency, key, left, max(measure - cum[idx], 0.0))
            add_edge(adjacency, key, right, max(cum[idx + 1] - measure, 0.0))
            return key

    nearest_vertex = min(coords, key=lambda xy: Point(xy).distance(snapped))
    add_edge(adjacency, key, node_key_xy(nearest_vertex[0], nearest_vertex[1]), Point(nearest_vertex).distance(snapped))
    return key


def dijkstra(adjacency: Dict[Tuple[float, float], Dict[Tuple[float, float], float]], source) -> Dict[Tuple[float, float], float]:
    distances = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > distances.get(node, math.inf):
            continue
        for neighbor, weight in adjacency.get(node, {}).items():
            new_dist = dist + weight
            if new_dist < distances.get(neighbor, math.inf):
                distances[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    return distances


def unique_values(series: pd.Series) -> str:
    values = sorted([str(v) for v in series.dropna().unique()])
    return "; ".join(values)


def scan_gis_data() -> Tuple[pd.DataFrame, Optional[Path], str]:
    datasets = [
        ("soil_points", SOIL_POINTS_PATH),
        ("corrected_river_network", RIVER_PATH),
        ("watershed_outlet", OUTLET_PATH),
        ("landuse_polygons", LANDUSE_PATH),
    ]
    rows = []
    for name, path in datasets:
        gdf = gpd.read_file(path)
        line_part_count = sum(len(geometry_lines(geom)) for geom in gdf.geometry)
        site_id_field = "样点" if name == "soil_points" and "样点" in gdf.columns else ""
        category_field = "NAME" if name == "landuse_polygons" and "NAME" in gdf.columns else ""
        rows.append(
            {
                "Dataset": name,
                "Path": str(path),
                "FeatureCount": len(gdf),
                "LinePartCount": line_part_count if line_part_count else np.nan,
                "CRS": str(gdf.crs),
                "CoordinateUnits": crs_unit_name(gdf),
                "GeometryTypes": unique_values(gdf.geom_type),
                "Fields": "; ".join([c for c in gdf.columns if c != "geometry"]),
                "SoilSiteIDField": site_id_field,
                "LandUseCategoryField": category_field,
            }
        )

    for path in MFD_CANDIDATES:
        rows.append(
            {
                "Dataset": "MFD_or_flow_accumulation_candidate",
                "Path": str(path),
                "FeatureCount": np.nan,
                "LinePartCount": np.nan,
                "CRS": "not inspected: rasterio unavailable",
                "CoordinateUnits": "unknown",
                "GeometryTypes": "raster",
                "Fields": "file exists" if path.exists() else "missing",
                "SoilSiteIDField": "",
                "LandUseCategoryField": "",
            }
        )

    polygon_candidates = []
    for path in sorted(list(ROOT.rglob("*.shp")) + list(ROOT.rglob("*.gpkg"))):
        try:
            gdf = gpd.read_file(path)
        except Exception:
            continue
        if any("Polygon" in gt for gt in set(gdf.geom_type.astype(str))):
            if path.resolve() != LANDUSE_PATH.resolve():
                polygon_candidates.append(path)

    thiessen_path = None
    thiessen_status = "not_found"
    for path in polygon_candidates:
        low = str(path).lower()
        if any(token in low for token in ["thiessen", "voronoi", "泰森", "采样", "代表", "influence", "contributing", "catchment"]):
            thiessen_path = path
            thiessen_status = "candidate_found"
            break

    rows.append(
        {
            "Dataset": "soil_thiessen_or_representative_polygon_search",
            "Path": str(thiessen_path) if thiessen_path else "",
            "FeatureCount": np.nan,
            "LinePartCount": np.nan,
            "CRS": "",
            "CoordinateUnits": "",
            "GeometryTypes": "",
            "Fields": thiessen_status,
            "SoilSiteIDField": "",
            "LandUseCategoryField": "",
        }
    )
    scan = pd.DataFrame(rows)
    scan.to_csv(DIAG_DIR / "gis_data_scan.csv", index=False, encoding="utf-8-sig")
    return scan, thiessen_path, thiessen_status


def calculate_soil_distances() -> Tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    soil = gpd.read_file(SOIL_POINTS_PATH)
    river = gpd.read_file(RIVER_PATH)
    outlet = gpd.read_file(OUTLET_PATH)
    if river.crs is None:
        raise ValueError("River network has no CRS.")
    if crs_unit_name(river).lower() not in {"metre", "meter"}:
        raise ValueError(f"River CRS is not meter-based: {river.crs}")

    soil_proj = soil.to_crs(river.crs)
    outlet_proj = outlet.to_crs(river.crs)
    outlet_point = outlet_proj.geometry.iloc[0]
    if not isinstance(outlet_point, Point):
        outlet_point = outlet_point.centroid

    lines = []
    for geom in river.geometry:
        lines.extend(geometry_lines(geom))
    if not lines:
        raise ValueError("No line geometry found in corrected river network.")

    adjacency = build_base_graph(lines)
    out_idx, out_line, out_snap, out_measure, out_snapdist = nearest_line(outlet_point, lines)
    outlet_node = connect_snapped_point(adjacency, out_line, out_snap, out_measure)

    site_records = []
    site_nodes = {}
    for idx, row in soil_proj.iterrows():
        point = row.geometry
        site_id = str(row["样点"])
        line_idx, line, snapped, measure, snapdist = nearest_line(point, lines)
        site_node = connect_snapped_point(adjacency, line, snapped, measure)
        site_nodes[site_id] = site_node
        euclidean = point.distance(outlet_point)
        snap_warning = ";".join([f">{t:g}m" for t in SNAP_WARNING_THRESHOLDS_M if snapdist > t])
        site_records.append(
            {
                "Site_ID": site_id,
                "Orig_X": point.x,
                "Orig_Y": point.y,
                "Snap_X": snapped.x,
                "Snap_Y": snapped.y,
                "SnapDist_m": snapdist,
                "nearest_segment_id": line_idx,
                "measure_on_segment": measure,
                "EuclideanDist_m": euclidean,
                "snap_warning": snap_warning,
            }
        )

    network_distances = dijkstra(adjacency, outlet_node)
    rows = []
    for record in site_records:
        site_node = site_nodes[record["Site_ID"]]
        network_dist = network_distances.get(site_node, np.nan)
        connected = pd.notna(network_dist)
        final_dist = network_dist if connected else record["EuclideanDist_m"]
        method = "network" if connected else "euclidean_fallback"
        rows.append(
            {
                **record,
                "NetworkDist_m": network_dist,
                "FinalDist_m": final_dist,
                "Dist_outlet": final_dist,
                "DistanceMethod": method,
                "ConnectedToOutlet": bool(connected),
            }
        )

    diag = pd.DataFrame(rows)
    diag = diag[
        [
            "Site_ID",
            "Orig_X",
            "Orig_Y",
            "Snap_X",
            "Snap_Y",
            "SnapDist_m",
            "NetworkDist_m",
            "EuclideanDist_m",
            "FinalDist_m",
            "Dist_outlet",
            "DistanceMethod",
            "ConnectedToOutlet",
            "nearest_segment_id",
            "measure_on_segment",
            "snap_warning",
        ]
    ]
    diag.to_csv(DIAG_DIR / "soil_distance_diagnostics.csv", index=False, encoding="utf-8-sig")

    n_total = len(diag)
    n_network = int((diag["DistanceMethod"] == "network").sum())
    n_fallback = int((diag["DistanceMethod"] == "euclidean_fallback").sum())
    summary = pd.DataFrame(
        [
            {
                "N_total": n_total,
                "N_network": n_network,
                "N_euclidean_fallback": n_fallback,
                "Network_percentage": n_network / n_total if n_total else np.nan,
                "Fallback_percentage": n_fallback / n_total if n_total else np.nan,
                "Fallback_warning": bool(n_total and n_fallback / n_total > FALLBACK_WARNING_THRESHOLD),
                "SnapDist_mean_m": diag["SnapDist_m"].mean(),
                "SnapDist_median_m": diag["SnapDist_m"].median(),
                "SnapDist_max_m": diag["SnapDist_m"].max(),
                "NetworkDist_min_m": diag["NetworkDist_m"].min(skipna=True),
                "NetworkDist_max_m": diag["NetworkDist_m"].max(skipna=True),
                "EuclideanDist_min_m": diag["EuclideanDist_m"].min(skipna=True),
                "EuclideanDist_max_m": diag["EuclideanDist_m"].max(skipna=True),
                "OutletSnapDist_m": out_snapdist,
                "OutletNearestSegmentID": out_idx,
            }
        ]
    )
    summary.to_csv(DIAG_DIR / "soil_distance_summary.csv", index=False, encoding="utf-8-sig")

    soil_with_dist = soil.copy()
    soil_with_dist = soil_with_dist.merge(diag, left_on=soil_with_dist["样点"].astype(str), right_on="Site_ID", how="left")
    return soil_with_dist, diag, summary


def safe_ratio_name(category: str) -> str:
    cleaned = str(category).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    return f"LU_{cleaned}_ratio"


def build_landuse_tables(soil_with_dist: gpd.GeoDataFrame, thiessen_path: Optional[Path], thiessen_status: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    landuse = gpd.read_file(LANDUSE_PATH)
    candidate_fields = [field for field in ["NAME", "Class", "Type", "LandUse", "LU_Type"] if field in landuse.columns]
    category_field = candidate_fields[0] if candidate_fields else None
    category_rows = []
    if category_field:
        for value in sorted(landuse[category_field].dropna().astype(str).unique()):
            category_rows.append({"CategoryField": category_field, "LandUseCategory": value})
    categories = pd.DataFrame(category_rows)
    categories.to_csv(DIAG_DIR / "landuse_categories.csv", index=False, encoding="utf-8-sig")

    field_scan = pd.DataFrame(
        [
            {
                "CandidateField": field,
                "Present": field in landuse.columns,
                "UniqueValues": unique_values(landuse[field]) if field in landuse.columns else "",
            }
            for field in ["NAME", "Class", "Type", "LandUse", "LU_Type", "土类", "K值"]
        ]
    )
    field_scan.to_csv(DIAG_DIR / "landuse_category_field_scan.csv", index=False, encoding="utf-8-sig")

    site_ids = soil_with_dist["样点"].astype(str).tolist()
    if thiessen_path is None:
        reason = "not_computed_missing_soil_thiessen_or_representative_polygons"
        raw = pd.DataFrame({"Site_ID": site_ids, "Status": reason})
        grouped = pd.DataFrame(
            {
                "Site_ID": site_ids,
                "HighAg_ratio": np.nan,
                "NaturalVeg_ratio": np.nan,
                "ModerateAg_ratio": np.nan,
                "Built_ratio": np.nan,
                "Water_ratio": np.nan,
                "sum_ratio": np.nan,
                "ratio_warning": reason,
            }
        )
        raw.to_csv(DIAG_DIR / "soil_landuse_raw_ratios.csv", index=False, encoding="utf-8-sig")
        grouped.to_csv(DIAG_DIR / "soil_landuse_grouped_ratios.csv", index=False, encoding="utf-8-sig")
        return raw, grouped, categories

    thiessen = gpd.read_file(thiessen_path)
    if thiessen.crs != landuse.crs:
        thiessen = thiessen.to_crs(landuse.crs)
    if category_field is None:
        raise ValueError("No suitable land-use category field found.")

    join_candidates = [c for c in ["Site_ID", "样点", "Input_FID", "OBJECTID", "FID"] if c in thiessen.columns]
    if not join_candidates:
        raise ValueError(f"Thiessen candidate has no obvious site join field: {thiessen_path}")
    site_field = join_candidates[0]
    thiessen = thiessen[[site_field, "geometry"]].copy()
    thiessen["Site_ID"] = thiessen[site_field].astype(str)
    thiessen["poly_area_m2"] = thiessen.geometry.area
    intersections = gpd.overlay(thiessen, landuse[[category_field, "geometry"]], how="intersection")
    intersections["area_m2"] = intersections.geometry.area
    areas = intersections.groupby(["Site_ID", category_field])["area_m2"].sum().reset_index()
    totals = thiessen[["Site_ID", "poly_area_m2"]]
    areas = areas.merge(totals, on="Site_ID", how="left")
    areas["ratio"] = areas["area_m2"] / areas["poly_area_m2"]
    raw = areas.pivot_table(index="Site_ID", columns=category_field, values="ratio", fill_value=0).reset_index()
    raw = raw.rename(columns={c: safe_ratio_name(c) for c in raw.columns if c != "Site_ID"})
    ratio_cols = [c for c in raw.columns if c.startswith("LU_")]
    raw["sum_ratio"] = raw[ratio_cols].sum(axis=1)
    raw["ratio_warning"] = np.where((raw["sum_ratio"] < 0.98) | (raw["sum_ratio"] > 1.02), "sum_ratio_outside_0.98_1.02", "")

    grouped = pd.DataFrame({"Site_ID": raw["Site_ID"]})
    for col in ["HighAg_ratio", "NaturalVeg_ratio", "ModerateAg_ratio", "Built_ratio", "Water_ratio"]:
        grouped[col] = 0.0

    mapping = {
        "HighAg_ratio": ["大棚", "菜地"],
        "NaturalVeg_ratio": ["林地", "灌木地", "竹林", "草地"],
        "ModerateAg_ratio": ["旱地", "水稻田", "苗木地"],
        "Built_ratio": ["居民地", "道路"],
        "Water_ratio": ["河流", "湖泊"],
    }
    mapping_rows = []
    for group, names in mapping.items():
        used_cols = []
        for name in names:
            col = safe_ratio_name(name)
            if col in raw.columns:
                grouped[group] += raw[col]
                used_cols.append(col)
        mapping_rows.append({"GroupedVariable": group, "SourceCategories": "; ".join(names), "SourceColumnsFound": "; ".join(used_cols)})
    grouped["sum_grouped_ratio"] = grouped[["HighAg_ratio", "NaturalVeg_ratio", "ModerateAg_ratio", "Built_ratio", "Water_ratio"]].sum(axis=1)
    raw.to_csv(DIAG_DIR / "soil_landuse_raw_ratios.csv", index=False, encoding="utf-8-sig")
    grouped.to_csv(DIAG_DIR / "soil_landuse_grouped_ratios.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mapping_rows).to_csv(DIAG_DIR / "landuse_group_mapping.csv", index=False, encoding="utf-8-sig")
    return raw, grouped, categories


def spearman_pairs(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    usable = [v for v in variables if v in df.columns and pd.to_numeric(df[v], errors="coerce").notna().sum() >= 3]
    rows = []
    for i, a in enumerate(usable):
        for b in usable[i + 1 :]:
            xa = pd.to_numeric(df[a], errors="coerce")
            xb = pd.to_numeric(df[b], errors="coerce")
            valid = xa.notna() & xb.notna()
            n = int(valid.sum())
            if n >= 3 and xa[valid].nunique() > 1 and xb[valid].nunique() > 1:
                rho, p_value = stats.spearmanr(xa[valid], xb[valid])
            else:
                rho, p_value = np.nan, np.nan
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
    return pd.DataFrame(rows)


def vif_table(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    present = []
    rows = []
    for var in variables:
        if var not in df.columns:
            rows.append({"Variable": var, "VIF": np.nan, "N_complete": 0, "Status": "missing"})
            continue
        x = pd.to_numeric(df[var], errors="coerce")
        if x.notna().sum() < 3:
            rows.append({"Variable": var, "VIF": np.nan, "N_complete": int(x.notna().sum()), "Status": "too few non-missing values"})
        elif x.dropna().nunique() <= 1:
            rows.append({"Variable": var, "VIF": np.nan, "N_complete": int(x.notna().sum()), "Status": "constant"})
        else:
            present.append(var)
    if len(present) < 2:
        for var in present:
            rows.append({"Variable": var, "VIF": np.nan, "N_complete": int(pd.to_numeric(df[var], errors="coerce").notna().sum()), "Status": "not computed: fewer than 2 usable variables"})
        return pd.DataFrame(rows)
    X = pd.DataFrame({v: pd.to_numeric(df[v], errors="coerce") for v in present}).dropna()
    n = len(X)
    if n <= len(present) + 1:
        for var in present:
            rows.append({"Variable": var, "VIF": np.nan, "N_complete": n, "Status": "too few complete cases"})
        return pd.DataFrame(rows)
    X = (X - X.mean()) / X.std(ddof=0)
    for var in present:
        y = X[var].to_numpy()
        other = X[[c for c in present if c != var]].to_numpy()
        design = np.column_stack([np.ones(n), other])
        beta = np.linalg.pinv(design.T @ design) @ design.T @ y
        pred = design @ beta
        rss = float(np.sum((y - pred) ** 2))
        tss = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else np.nan
        vif = np.inf if pd.notna(r2) and r2 >= 0.999999 else (1.0 / (1.0 - r2) if pd.notna(r2) else np.nan)
        rows.append(
            {
                "Variable": var,
                "VIF": vif,
                "N_complete": n,
                "Status": "warning: VIF > 5" if pd.notna(vif) and vif > 5 else "ok",
            }
        )
    return pd.DataFrame(rows)


def write_covariate_outputs(soil_with_dist: gpd.GeoDataFrame, grouped: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    soil = soil_with_dist.copy()
    soil["Site_ID"] = soil["样点"].astype(str)
    soil = soil.rename(
        columns={
            "BurnDEMMFD": "BurnDEMMFDAcc",
        }
    )
    if "BurnDEMMFDAcc" not in soil.columns and "BurnDEMMFD" in soil.columns:
        soil["BurnDEMMFDAcc"] = soil["BurnDEMMFD"]
    soil["log_MFD"] = np.log1p(pd.to_numeric(soil["BurnDEMMFDAcc"], errors="coerce").clip(lower=0))
    cov = pd.DataFrame(soil.drop(columns="geometry", errors="ignore"))
    cov = cov.merge(grouped, on="Site_ID", how="left")
    for col in ["NetworkDist_m", "EuclideanDist_m", "FinalDist_m", "DistanceMethod", "BurnDEMMFDAcc", "log_MFD"]:
        if col not in cov.columns:
            cov[col] = np.nan
    base_cols = [
        "Site_ID",
        "TN",
        "TP",
        "TOC",
        "NetworkDist_m",
        "EuclideanDist_m",
        "FinalDist_m",
        "DistanceMethod",
        "BurnDEMMFDAcc",
        "log_MFD",
    ]
    grouped_cols = ["HighAg_ratio", "NaturalVeg_ratio", "ModerateAg_ratio", "Built_ratio", "Water_ratio"]
    raw_ratio_cols = [c for c in raw.columns if c.startswith("LU_")]
    out_cols = base_cols + raw_ratio_cols + grouped_cols
    for col in out_cols:
        if col not in cov.columns:
            cov[col] = np.nan
    cov[out_cols].to_csv(OUT_DIR / "soil_covariates_complete.csv", index=False, encoding="utf-8-sig")

    shp = soil.copy()
    shp = shp.merge(grouped, on="Site_ID", how="left")
    shp["Dist_out_m"] = shp["FinalDist_m"]
    shp["DistMeth"] = shp["DistanceMethod"].astype(str)
    shp["HighAg"] = shp["HighAg_ratio"]
    shp["NatVeg"] = shp["NaturalVeg_ratio"]
    shp["ModAg"] = shp["ModerateAg_ratio"]
    shp["log_MFD"] = soil["log_MFD"]
    keep = ["Site_ID", "Dist_out_m", "DistMeth", "HighAg", "NatVeg", "ModAg", "log_MFD", "geometry"]
    shp_out = shp[keep].copy()
    shp_path = GIS_DIR / "soil_sites_with_covariates.shp"
    for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        target = shp_path.with_suffix(suffix)
        if target.exists():
            target.unlink()
    shp_out.to_file(shp_path, encoding="utf-8")
    pd.DataFrame(
        [
            {"ShapefileField": "Dist_out_m", "FullMeaning": "Dist_outlet / FinalDist_m"},
            {"ShapefileField": "DistMeth", "FullMeaning": "DistanceMethod"},
            {"ShapefileField": "HighAg", "FullMeaning": "HighAg_ratio"},
            {"ShapefileField": "NatVeg", "FullMeaning": "NaturalVeg_ratio"},
            {"ShapefileField": "ModAg", "FullMeaning": "ModerateAg_ratio"},
            {"ShapefileField": "log_MFD", "FullMeaning": "log1p(BurnDEMMFDAcc)"},
        ]
    ).to_csv(GIS_DIR / "soil_sites_with_covariates_field_mapping.csv", index=False, encoding="utf-8-sig")
    return cov[out_cols]


def write_record(scan: pd.DataFrame, distance_summary: pd.DataFrame, categories: pd.DataFrame, cov: pd.DataFrame, spearman: pd.DataFrame, vif: pd.DataFrame, thiessen_status: str) -> None:
    s = distance_summary.iloc[0].to_dict()
    cat_text = "None found"
    if not categories.empty:
        cat_text = "\n".join(f"- {v}" for v in categories["LandUseCategory"].tolist())
    high_corr = spearman[spearman["High_correlation_abs_rho_gt_0_70"] == True]  # noqa: E712
    high_text = "未发现 |rho| > 0.70 的候选 predictor 高相关对。"
    if not high_corr.empty:
        high_text = "\n".join(
            f"- {r.Variable_1} vs {r.Variable_2}: rho={r.Spearman_rho:.3f}, N={r.N_pairwise}, p={r.P_value:.4g}"
            for r in high_corr.itertuples(index=False)
        )
    vif_warn = vif[vif["Status"].astype(str).str.contains("warning", na=False)]
    vif_text = "主模型 VIF 暂无可计算的明显问题；土地利用变量缺失，因此完整主模型 VIF 尚不能评价。"
    if not vif_warn.empty:
        vif_text = "\n".join(f"- {r.Variable}: VIF={r.VIF}" for r in vif_warn.itertuples(index=False))

    def dist_line(col: str) -> str:
        vals = pd.to_numeric(cov[col], errors="coerce") if col in cov.columns else pd.Series(dtype=float)
        if vals.notna().sum() == 0:
            return f"{col}: not available"
        return f"{col}: min={vals.min():.4f}, median={vals.median():.4f}, mean={vals.mean():.4f}, max={vals.max():.4f}"

    md = f"""---
exp_id: HY-B-260826-003
title: Soil distance and land-use covariate completion phase
date: 2026-08-26
created_at: "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
system: HY
device: B
type: gis_covariate_diagnostics
status: completed_with_landuse_blocked
skill_basis:
  - nature-statistics
  - nature-experiment-log
outputs_root: results/water_soil_multivariable_phase3_soil_covariates
---

# Soil Covariate Completion Experiment Log

## Scope

This stage followed the requested stop point: distance calculation, GIS scan, land-use representative-unit check, grouped-variable preparation where possible, Spearman and VIF diagnostics. No TN/TP/TOC multivariable soil models were run in this stage.

## GIS Scan

See `diagnostics/gis_data_scan.csv`.

- Soil sampling points: 47 features, Site_ID field = `样点`.
- Corrected river network: 1 feature, line count after exploding geometry = derived from `MultiLineString`, CRS EPSG:32650.
- Outlet: 1 point, read from `results/figure3_network_distance_stage1/outlet_snapped.shp`.
- Land-use polygon layer: 394 features, category field = `NAME`.
- Soil Thiessen / representative polygons: {thiessen_status}.

## Distance Diagnostics

- A. Soil points with successful NetworkDist: {int(s['N_network'])}.
- B. Soil points using Euclidean fallback: {int(s['N_euclidean_fallback'])}.
- C. Maximum SnapDist: {float(s['SnapDist_max_m']):.3f} m.
- Fallback percentage: {float(s['Fallback_percentage']) * 100:.2f}%.
- Fallback warning (>20%): {s['Fallback_warning']}.

Distance outputs:

- `diagnostics/soil_distance_diagnostics.csv`
- `diagnostics/soil_distance_summary.csv`

## Land-Use Categories

Actual categories from `NAME`:

{cat_text}

Land-use ratio status:

- Raw land-use ratios were not calculated because no existing soil Thiessen polygons, sampling influence polygons or contributing-area polygons were found.
- No arbitrary 100 m / 200 m buffer was created.
- Grouped ratios are therefore missing and explicitly marked in `soil_landuse_grouped_ratios.csv`.

## Grouped Variable Definitions

Planned definitions retained for the next stage:

- `HighAg_ratio = 大棚 + 菜地`
- `NaturalVeg_ratio = 林地 + 灌木地 + 竹林 + 草地`
- `ModerateAg_ratio = 旱地 + 水稻田 + 苗木地`
- `Built_ratio = 居民地 + 道路`
- `Water_ratio = 河流 + 湖泊`

## Ratio Distribution

- E. {dist_line('HighAg_ratio')}
- F. {dist_line('NaturalVeg_ratio')}
- G. {dist_line('ModerateAg_ratio')}

These distributions are unavailable until a valid representative polygon layer is provided.

## Spearman Diagnostics

H. High-correlation predictor pairs:

{high_text}

## VIF Diagnostics

I. {vif_text}

## Generated Files

- `soil_covariates_complete.csv`
- `gis/soil_sites_with_covariates.shp`
- `gis/soil_sites_with_covariates_field_mapping.csv`
- `diagnostics/gis_data_scan.csv`
- `diagnostics/landuse_categories.csv`
- `diagnostics/landuse_category_field_scan.csv`
- `diagnostics/soil_distance_diagnostics.csv`
- `diagnostics/soil_distance_summary.csv`
- `diagnostics/soil_landuse_raw_ratios.csv`
- `diagnostics/soil_landuse_grouped_ratios.csv`
- `diagnostics/soil_predictor_spearman.csv`
- `diagnostics/soil_predictor_VIF.csv`
- `sensitivity/soil_distance_method_sensitivity.csv`

## Stop Point

Stopped before formal TN/TP/TOC soil multivariable models. The requested full soil model remains blocked until valid land-use ratios are supplied or an existing representative polygon layer is added.
"""
    path = LOG_DIR / "HY-B-260826-003_soil_covariate_completion.md"
    path.write_text(md, encoding="utf-8")

    all_record = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "water_soil_multivariable_experiment_record.md"
    if all_record.exists():
        marker = "\n\n## 13. Phase 3: Soil Covariate Completion Stop Point\n\n"
        phase3_text = (
            marker
            + "Script: `analysis/water_soil_multivariable_phase3_soil_covariates.py`\n\n"
            + "Output root: `results/water_soil_multivariable_phase3_soil_covariates`\n\n"
            + f"- Network distance successful: {int(s['N_network'])}/47 soil sites.\n"
            + f"- Euclidean fallback: {int(s['N_euclidean_fallback'])}/47 soil sites.\n"
            + f"- Maximum soil snap distance: {float(s['SnapDist_max_m']):.3f} m.\n"
            + "- Land-use category field: `NAME`.\n"
            + f"- Soil Thiessen / representative polygons: {thiessen_status}; land-use ratios not computed when missing.\n"
            + "- Formal soil TN/TP/TOC models were not run in this phase.\n"
        )
        text = all_record.read_text(encoding="utf-8")
        if marker in text:
            text = text.split(marker)[0].rstrip() + phase3_text
        else:
            text = text.rstrip() + phase3_text
        all_record.write_text(text, encoding="utf-8")


def write_sensitivity_placeholder() -> None:
    pd.DataFrame(
        [
            {
                "Analysis": "soil_distance_method_sensitivity",
                "Status": "not_run_at_phase3_stop_point",
                "Reason": "Distance-definition sensitivity requires fitting Model A/B/C. The user requested this phase stop after distance calculation, land-use diagnostics, Spearman and VIF.",
                "Model_A": "FinalDist_m, network + Euclidean fallback",
                "Model_B": "EuclideanDist_m for all soil sites",
                "Model_C": "NetworkDist_m only for network-connected soil sites",
            }
        ]
    ).to_csv(SENS_DIR / "soil_distance_method_sensitivity.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    scan, thiessen_path, thiessen_status = scan_gis_data()
    soil_with_dist, distance_diag, distance_summary = calculate_soil_distances()
    raw, grouped, categories = build_landuse_tables(soil_with_dist, thiessen_path, thiessen_status)
    cov = write_covariate_outputs(soil_with_dist, grouped, raw)

    predictor_vars = [
        "FinalDist_m",
        "EuclideanDist_m",
        "NetworkDist_m",
        "HighAg_ratio",
        "NaturalVeg_ratio",
        "ModerateAg_ratio",
        "Built_ratio",
        "Water_ratio",
        "log_MFD",
    ]
    spearman = spearman_pairs(cov, predictor_vars)
    vif = vif_table(cov, ["FinalDist_m", "HighAg_ratio", "NaturalVeg_ratio", "ModerateAg_ratio", "Built_ratio", "Water_ratio", "log_MFD"])
    spearman.to_csv(DIAG_DIR / "soil_predictor_spearman.csv", index=False, encoding="utf-8-sig")
    vif.to_csv(DIAG_DIR / "soil_predictor_VIF.csv", index=False, encoding="utf-8-sig")
    write_sensitivity_placeholder()

    write_record(scan, distance_summary, categories, cov, spearman, vif, thiessen_status)
    manifest = []
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            manifest.append({"File": str(path.relative_to(OUT_DIR)), "Bytes": path.stat().st_size})
    pd.DataFrame(manifest).to_csv(OUT_DIR / "phase3_output_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote phase-3 soil covariate diagnostics to: {OUT_DIR}")


if __name__ == "__main__":
    main()
