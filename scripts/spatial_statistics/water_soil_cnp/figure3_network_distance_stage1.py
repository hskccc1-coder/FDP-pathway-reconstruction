from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[3]
RIVER_PATH = ROOT / "data" / "soil_water_measurements" / "矫正后河流线" / "river.shp"
# The original ArcGIS outlet was normalized to this project-relative shapefile.
OUTLET_PATH = ROOT / "data" / "soil_water_measurements" / "outlet" / "outlet_snapped.shp"
WATER_POINTS_DIR = ROOT / "data" / "soil_water_measurements" / "water_seasonal_points" / "shp"
OUTPUT_DIR = ROOT / "results" / "pathway_metrics" / "water_soil_cnp" / "runtime" / "figure3_network_distance_stage1"
NODE_PRECISION = 3
SNAP_WARNING_THRESHOLDS_M = (5.0, 10.0, 20.0)
EUCLIDEAN_FALLBACK_SITE_IDS = {"3"}


def geometry_lines(geometry) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, (MultiLineString, GeometryCollection)):
        lines: list[LineString] = []
        for part in geometry.geoms:
            lines.extend(geometry_lines(part))
        return lines
    return []


def node_key(point: Point | tuple[float, float]) -> tuple[float, float]:
    if isinstance(point, Point):
        x, y = point.x, point.y
    else:
        x, y = point
    return (round(float(x), NODE_PRECISION), round(float(y), NODE_PRECISION))


def clear_vector(path: Path) -> None:
    for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".qix"]:
        target = path.with_suffix(suffix)
        if target.exists():
            target.unlink()


def crs_unit_name(gdf: gpd.GeoDataFrame) -> str:
    try:
        axis = gdf.crs.axis_info[0]
        return str(axis.unit_name)
    except Exception:
        return "unknown"


def read_unique_water_sites(target_crs) -> tuple[gpd.GeoDataFrame, list[str]]:
    logs: list[str] = []
    files = sorted(WATER_POINTS_DIR.glob("water_*.shp"))
    if not files:
        raise FileNotFoundError(f"No water point shapefiles found in {WATER_POINTS_DIR}")

    frames = []
    for path in files:
        gdf = gpd.read_file(path)
        logs.append(
            f"WATER FILE {path.name}: rows={len(gdf)}, crs={gdf.crs}, fields={list(gdf.columns)}"
        )
        required = {"SiteID", "Season", "Variable", "Value"}
        missing = required - set(gdf.columns)
        if missing:
            raise ValueError(f"{path.name} missing required fields: {sorted(missing)}")
        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
        frames.append(gdf)

    all_points = pd.concat(frames, ignore_index=True)
    all_points["Site_ID"] = all_points["SiteID"].astype(str)
    all_points["Orig_X"] = all_points.geometry.x
    all_points["Orig_Y"] = all_points.geometry.y

    coord_check = (
        all_points.groupby("Site_ID")
        .agg(
            n_records=("Site_ID", "size"),
            n_x=("Orig_X", "nunique"),
            n_y=("Orig_Y", "nunique"),
        )
        .reset_index()
    )
    inconsistent = coord_check[(coord_check["n_x"] > 1) | (coord_check["n_y"] > 1)]
    if not inconsistent.empty:
        logs.append("WARNING: Some Site_ID values have inconsistent coordinates across seasonal files.")
        logs.append(inconsistent.to_string(index=False))

    keep_cols = [c for c in ["Site_ID", "SiteType", "Orig_X", "Orig_Y", "geometry"] if c in all_points.columns]
    sites = all_points.sort_values("Site_ID").drop_duplicates("Site_ID")[keep_cols].copy()
    sites = gpd.GeoDataFrame(sites, geometry="geometry", crs=target_crs)
    logs.append(
        f"UNIQUE WATER SITES: site_count={len(sites)}, "
        f"records_read={len(all_points)}, duplicate_site_records_collapsed={len(all_points) - len(sites)}"
    )
    return sites, logs


def nearest_line_snap(point: Point, lines: list[LineString]) -> tuple[int, float, Point, float]:
    best_idx = -1
    best_dist = math.inf
    best_measure = math.nan
    best_point = None
    for idx, line in enumerate(lines):
        dist = point.distance(line)
        if dist < best_dist:
            measure = line.project(point)
            snapped = line.interpolate(measure)
            best_idx = idx
            best_dist = dist
            best_measure = measure
            best_point = snapped
    if best_point is None:
        raise RuntimeError("Could not snap point to river network.")
    return best_idx, best_measure, best_point, best_dist


def add_edge(adjacency: dict, a: tuple[float, float], b: tuple[float, float], length: float) -> None:
    if a == b or length <= 0:
        return
    current = adjacency[a].get(b)
    if current is None or length < current:
        adjacency[a][b] = length
        adjacency[b][a] = length


def connected_components(adjacency: dict[tuple[float, float], dict]) -> list[set[tuple[float, float]]]:
    seen = set()
    components = []
    for node in adjacency:
        if node in seen:
            continue
        comp = set()
        queue = deque([node])
        seen.add(node)
        while queue:
            current = queue.popleft()
            comp.add(current)
            for nxt in adjacency[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(comp)
    return components


def dijkstra(adjacency: dict, source: tuple[float, float]) -> dict[tuple[float, float], float]:
    distances = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > distances[node]:
            continue
        for nxt, weight in adjacency[node].items():
            ndist = dist + weight
            if ndist < distances.get(nxt, math.inf):
                distances[nxt] = ndist
                heapq.heappush(heap, (ndist, nxt))
    return distances


def build_graph(lines: list[LineString], snap_records: pd.DataFrame, outlet_record: dict) -> tuple[dict, dict]:
    measures_by_line: dict[int, list[float]] = defaultdict(list)
    for idx, line in enumerate(lines):
        measures_by_line[idx].extend([0.0, line.length])

    for row in snap_records.itertuples(index=False):
        measures_by_line[int(row.nearest_segment_id)].append(float(row.measure_on_segment))
    measures_by_line[int(outlet_record["nearest_segment_id"])].append(float(outlet_record["measure_on_segment"]))

    adjacency: dict[tuple[float, float], dict[tuple[float, float], float]] = defaultdict(dict)
    measure_to_node: dict[tuple[int, float], tuple[float, float]] = {}

    for idx, line in enumerate(lines):
        raw = sorted(measures_by_line[idx])
        unique = []
        for measure in raw:
            measure = min(max(float(measure), 0.0), line.length)
            if not unique or abs(measure - unique[-1]) > 1e-7:
                unique.append(measure)
        for measure in unique:
            measure_to_node[(idx, measure)] = node_key(line.interpolate(measure))
        for m1, m2 in zip(unique[:-1], unique[1:]):
            n1 = measure_to_node[(idx, m1)]
            n2 = measure_to_node[(idx, m2)]
            add_edge(adjacency, n1, n2, abs(m2 - m1))

    # Ensure isolated inserted nodes appear even when they are exactly on endpoints.
    for node in measure_to_node.values():
        adjacency[node] = adjacency[node]

    return adjacency, measure_to_node


def site_warning(distance: float) -> str:
    warnings = [f">{int(t)}m" for t in SNAP_WARNING_THRESHOLDS_M if distance > t]
    return ";".join(warnings)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []
    logs.append("Figure 3 network-distance stage 1")
    logs.append(f"river network path: {RIVER_PATH}")
    logs.append(f"outlet path: {OUTLET_PATH}")
    logs.append(f"water points path: {WATER_POINTS_DIR}")

    river = gpd.read_file(RIVER_PATH)
    outlet = gpd.read_file(OUTLET_PATH)
    logs.append(
        f"RIVER: rows={len(river)}, crs={river.crs}, projected={river.crs.is_projected}, "
        f"unit={crs_unit_name(river)}, fields={list(river.columns)}"
    )
    logs.append(
        f"OUTLET: rows={len(outlet)}, crs={outlet.crs}, projected={outlet.crs.is_projected}, "
        f"unit={crs_unit_name(outlet)}, fields={list(outlet.columns)}"
    )
    if len(outlet) != 1:
        raise ValueError(f"Expected one outlet point, found {len(outlet)}")
    if not river.crs or not river.crs.is_projected:
        raise ValueError("River CRS is not a projected CRS; provide or define a meter-based CRS first.")

    target_crs = river.crs
    if outlet.crs != target_crs:
        outlet = outlet.to_crs(target_crs)

    water_sites, water_logs = read_unique_water_sites(target_crs)
    logs.extend(water_logs)
    logs.append(
        f"WATER UNIQUE SITES: rows={len(water_sites)}, crs={water_sites.crs}, "
        f"projected={water_sites.crs.is_projected}, unit={crs_unit_name(water_sites)}, "
        f"fields={list(water_sites.columns)}"
    )

    noded = unary_union(list(river.geometry))
    lines = [line for line in geometry_lines(noded) if line.length > 0]
    logs.append(
        f"RIVER GRAPH INPUT: original_features={len(river)}, original_total_length_m={river.length.sum():.3f}, "
        f"noded_lines={len(lines)}, noded_total_length_m={sum(line.length for line in lines):.3f}"
    )

    snap_rows = []
    connector_geoms = []
    for row in water_sites.itertuples(index=False):
        seg_idx, measure, snapped, snap_dist = nearest_line_snap(row.geometry, lines)
        snap_rows.append(
            {
                "Site_ID": row.Site_ID,
                "SiteType": getattr(row, "SiteType", None),
                "Orig_X": row.geometry.x,
                "Orig_Y": row.geometry.y,
                "Snap_X": snapped.x,
                "Snap_Y": snapped.y,
                "SnapDist_m": snap_dist,
                "nearest_segment_id": seg_idx,
                "measure_on_segment": measure,
                "snap_warning": site_warning(snap_dist),
                "geometry": snapped,
            }
        )
        connector_geoms.append(LineString([row.geometry, snapped]))

    outlet_point = outlet.geometry.iloc[0]
    out_seg_idx, out_measure, outlet_snapped, outlet_snap_dist = nearest_line_snap(outlet_point, lines)
    outlet_record = {
        "Outlet_ID": 0,
        "Orig_X": outlet_point.x,
        "Orig_Y": outlet_point.y,
        "Snap_X": outlet_snapped.x,
        "Snap_Y": outlet_snapped.y,
        "SnapDist_m": outlet_snap_dist,
        "nearest_segment_id": out_seg_idx,
        "measure_on_segment": out_measure,
        "geometry": outlet_snapped,
    }
    logs.append(
        f"OUTLET SNAP: SnapDist_m={outlet_snap_dist:.6f}, nearest_segment_id={out_seg_idx}, "
        f"measure_on_segment={out_measure:.3f}"
    )

    snap_df = pd.DataFrame(snap_rows)
    adjacency, _ = build_graph(lines, snap_df, outlet_record)
    components = connected_components(adjacency)
    edge_count = sum(len(nbrs) for nbrs in adjacency.values()) // 2
    cycle_rank = edge_count - len(adjacency) + len(components)
    outlet_node = node_key(outlet_snapped)
    outlet_component_idx = next((i for i, comp in enumerate(components) if outlet_node in comp), None)
    logs.append(
        f"GRAPH: nodes={len(adjacency)}, edges={edge_count}, components={len(components)}, "
        f"cycle_rank={cycle_rank}, outlet_component_index={outlet_component_idx}"
    )
    logs.append(
        "GRAPH COMPONENT SIZES: "
        + ", ".join(str(len(comp)) for comp in sorted(components, key=len, reverse=True))
    )
    if cycle_rank > 0:
        logs.append(
            "WARNING: Network contains one or more loops. Distances were not assigned because downstream "
            "direction is ambiguous without flow-direction or elevation fields."
        )
        distances = {}
    else:
        distances = dijkstra(adjacency, outlet_node)

    network_rows = []
    disconnected_rows = []
    for row in snap_rows:
        sample_node = node_key(Point(row["Snap_X"], row["Snap_Y"]))
        netdist = distances.get(sample_node, math.nan)
        is_connected = not pd.isna(netdist)
        distance_method = "network_distance"
        euclidean_fallback_m = math.nan
        if (not is_connected) and str(row["Site_ID"]) in EUCLIDEAN_FALLBACK_SITE_IDS:
            euclidean_fallback_m = Point(row["Orig_X"], row["Orig_Y"]).distance(outlet_point)
            netdist = euclidean_fallback_m
            distance_method = "euclidean_straight_line_fallback"
        out = dict(row)
        out.pop("geometry")
        out["NetworkDist_m"] = netdist
        out["NetworkDist_km"] = netdist / 1000.0 if not pd.isna(netdist) else math.nan
        out["connected_to_outlet"] = bool(is_connected)
        out["distance_method"] = distance_method
        out["EuclideanFallback_m"] = euclidean_fallback_m
        network_rows.append(out)
        if not is_connected:
            disconnected_rows.append(
                {
                    "Site_ID": row["Site_ID"],
                    "Snap_X": row["Snap_X"],
                    "Snap_Y": row["Snap_Y"],
                    "nearest_segment_id": row["nearest_segment_id"],
                    "reason": "not reachable from snapped outlet in river graph",
                    "fallback_assigned": str(row["Site_ID"]) in EUCLIDEAN_FALLBACK_SITE_IDS,
                    "EuclideanFallback_m": euclidean_fallback_m,
                }
            )

    network_df = pd.DataFrame(network_rows)
    snap_diag = network_df[
        ["Site_ID", "SnapDist_m", "nearest_segment_id", "snap_warning"]
    ].copy()
    logs.append(
        "SNAP DISTANCE SUMMARY: "
        f"mean={snap_diag['SnapDist_m'].mean():.3f} m, "
        f"median={snap_diag['SnapDist_m'].median():.3f} m, "
        f"min={snap_diag['SnapDist_m'].min():.3f} m, "
        f"max={snap_diag['SnapDist_m'].max():.3f} m"
    )
    logs.append(f"DISCONNECTED WATER SITES: {len(disconnected_rows)}")
    logs.append(
        "EUCLIDEAN FALLBACK: disconnected Site_ID values assigned straight-line distance "
        f"from original sampling point to original outlet: {sorted(EUCLIDEAN_FALLBACK_SITE_IDS)}"
    )
    if network_df["NetworkDist_m"].notna().any():
        logs.append(
            "NETWORK DISTANCE SUMMARY: "
            f"mean={network_df['NetworkDist_m'].mean():.3f} m, "
            f"median={network_df['NetworkDist_m'].median():.3f} m, "
            f"min={network_df['NetworkDist_m'].min():.3f} m, "
            f"max={network_df['NetworkDist_m'].max():.3f} m"
        )
    if (network_df["NetworkDist_m"].dropna() < 0).any():
        logs.append("WARNING: Negative network distance detected.")
    else:
        logs.append("LOGIC CHECK: No negative network distances detected.")

    snapped_gdf = gpd.GeoDataFrame(snap_df.drop(columns="geometry"), geometry=snap_df["geometry"], crs=target_crs)
    outlet_gdf = gpd.GeoDataFrame(
        [{k: v for k, v in outlet_record.items() if k != "geometry"}],
        geometry=[outlet_record["geometry"]],
        crs=target_crs,
    )
    network_gdf = gpd.GeoDataFrame(
        network_df,
        geometry=[Point(xy) for xy in zip(network_df["Snap_X"], network_df["Snap_Y"])],
        crs=target_crs,
    )
    connector_gdf = gpd.GeoDataFrame(
        network_df[["Site_ID"]].copy(), geometry=connector_geoms, crs=target_crs
    )

    for stem in [
        "water_sites_snapped",
        "outlet_snapped",
        "water_sites_with_network_distance",
        "water_site_snap_connectors",
    ]:
        clear_vector(OUTPUT_DIR / f"{stem}.shp")
    snapped_gdf.to_file(OUTPUT_DIR / "water_sites_snapped.shp", encoding="UTF-8")
    outlet_gdf.to_file(OUTPUT_DIR / "outlet_snapped.shp", encoding="UTF-8")
    network_gdf.to_file(OUTPUT_DIR / "water_sites_with_network_distance.shp", encoding="UTF-8")
    connector_gdf.to_file(OUTPUT_DIR / "water_site_snap_connectors.shp", encoding="UTF-8")

    network_df.to_csv(OUTPUT_DIR / "water_sites_network_distance.csv", index=False, encoding="utf-8-sig")
    snap_diag.to_csv(OUTPUT_DIR / "snap_distance_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        disconnected_rows,
        columns=[
            "Site_ID",
            "Snap_X",
            "Snap_Y",
            "nearest_segment_id",
            "reason",
            "fallback_assigned",
            "EuclideanFallback_m",
        ],
    ).to_csv(OUTPUT_DIR / "disconnected_sites.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    river.plot(ax=ax, color="#78909C", linewidth=0.8, label="Corrected river network")
    connector_gdf.plot(ax=ax, color="#BDBDBD", linewidth=0.55, alpha=0.8, label="Original-to-snapped connector")
    water_sites.plot(ax=ax, color="#4D4D4D", markersize=18, alpha=0.65, label="Original water sites")
    network_gdf.plot(ax=ax, color="#1F78B4", markersize=22, alpha=0.9, label="Snapped water sites")
    outlet.plot(ax=ax, color="#D95F02", markersize=42, marker="*", label="Original outlet")
    outlet_gdf.plot(ax=ax, color="#E31A1C", markersize=55, marker="*", label="Snapped outlet")
    for row in network_gdf.itertuples(index=False):
        label = f"{row.Site_ID}"
        if not pd.isna(row.NetworkDist_km):
            label += f"\n{row.NetworkDist_km:.2f} km"
        else:
            label += "\nNA"
        ax.annotate(label, xy=(row.geometry.x, row.geometry.y), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.grid(True, color="#E6E6E6", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=7, frameon=True)
    fig.savefig(OUTPUT_DIR / "network_distance_QA.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    logs.append("OUTPUTS WRITTEN:")
    for name in [
        "water_sites_snapped.shp",
        "outlet_snapped.shp",
        "water_sites_with_network_distance.shp",
        "water_sites_network_distance.csv",
        "snap_distance_diagnostics.csv",
        "disconnected_sites.csv",
        "network_distance_QA.png",
    ]:
        logs.append(f"- {OUTPUT_DIR / name}")
    logs.append(
        "NOTE: Shapefile field names longer than 10 characters may be truncated by the ESRI Shapefile format; "
        "CSV outputs preserve full field names."
    )
    logs.append("Watershed boundary was not plotted because no boundary layer path was provided.")
    (OUTPUT_DIR / "analysis_log.txt").write_text("\n".join(logs), encoding="utf-8")


if __name__ == "__main__":
    main()
