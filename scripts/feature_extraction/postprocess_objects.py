from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from common import load_config, resolve, save_json

try:
    from shapely.validation import make_valid
except ImportError:
    make_valid = None


def repair_geometry(geometry):
    if geometry is None or geometry.is_empty:
        return geometry
    if geometry.is_valid:
        return geometry
    repaired = make_valid(geometry) if make_valid is not None else geometry.buffer(0)
    if not repaired.is_valid:
        repaired = repaired.buffer(0)
    return repaired


def polygon_parts(geometry) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return [part for part in geometry.geoms if not part.is_empty]
    repaired = repair_geometry(geometry)
    if repaired.geom_type == "Polygon":
        return [repaired]
    if repaired.geom_type == "MultiPolygon":
        return [part for part in repaired.geoms if not part.is_empty]
    return []


def fill_small_holes_polygon(polygon: Polygon, maximum_area: float) -> tuple[Polygon, int, float]:
    kept_interiors = []
    filled_count = 0
    filled_area = 0.0
    for ring in polygon.interiors:
        hole = Polygon(ring)
        if hole.area <= maximum_area:
            filled_count += 1
            filled_area += float(hole.area)
        else:
            kept_interiors.append(ring.coords)
    result = Polygon(polygon.exterior.coords, kept_interiors)
    return result, filled_count, filled_area


def fill_small_holes(geometry, maximum_area: float):
    output = []
    count = 0
    area = 0.0
    for part in polygon_parts(geometry):
        cleaned, local_count, local_area = fill_small_holes_polygon(part, maximum_area)
        if not cleaned.is_empty:
            output.append(cleaned)
        count += local_count
        area += local_area
    if not output:
        return geometry, count, area
    result = output[0] if len(output) == 1 else MultiPolygon(output)
    return result, count, area


def intersection_over_union(first, second) -> float:
    intersection = first.intersection(second).area
    if intersection <= 0:
        return 0.0
    union = first.area + second.area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _score_values(frame: gpd.GeoDataFrame, preferred: str) -> pd.Series:
    candidates = [preferred, "pred_iou", "class_prob", "stability", "area_m2"]
    for field in candidates:
        if field and field in frame.columns:
            values = pd.to_numeric(frame[field], errors="coerce")
            if values.notna().any():
                return values.fillna(values.min() - 1.0)
    return frame.geometry.area.astype(float)


def _spatial_index(frame: gpd.GeoDataFrame):
    try:
        return frame.sindex
    except ImportError:
        return None


def _candidates(frame: gpd.GeoDataFrame, spatial_index, bounds):
    return range(len(frame)) if spatial_index is None else spatial_index.intersection(bounds)


def class_aware_nms(
    frame: gpd.GeoDataFrame,
    threshold: float,
    class_field: str | None,
    score_field: str,
) -> tuple[gpd.GeoDataFrame, int]:
    if frame.empty:
        return frame, 0
    work = frame.reset_index(drop=True).copy()
    work["__score"] = _score_values(work, score_field)
    work["__area"] = work.geometry.area
    groups = work[class_field].astype(str) if class_field and class_field in work.columns else pd.Series("all", index=work.index)
    spatial_index = _spatial_index(work)
    suppressed: set[int] = set()
    kept: list[int] = []
    order = work.sort_values(["__score", "__area"], ascending=[False, False]).index.tolist()
    for index in order:
        if index in suppressed:
            continue
        kept.append(index)
        geometry = work.geometry.iloc[index]
        for candidate in _candidates(work, spatial_index, geometry.bounds):
            candidate = int(candidate)
            if candidate == index or candidate in suppressed:
                continue
            if groups.iloc[candidate] != groups.iloc[index]:
                continue
            other = work.geometry.iloc[candidate]
            if not geometry.intersects(other):
                continue
            if intersection_over_union(geometry, other) >= threshold:
                suppressed.add(candidate)
    result = work.loc[kept].drop(columns=["__score", "__area"]).reset_index(drop=True)
    return result, len(suppressed)


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        root_first, root_second = self.find(first), self.find(second)
        if root_first != root_second:
            self.parent[root_second] = root_first

    def components(self) -> list[list[int]]:
        values: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self.parent)):
            values[self.find(index)].append(index)
        return list(values.values())


def _component_frame(frame: gpd.GeoDataFrame, components: Iterable[list[int]], closing_radius: float = 0.0):
    rows = []
    score = _score_values(frame, "pred_iou")
    for component in components:
        best = max(component, key=lambda index: float(score.iloc[index]))
        row = frame.iloc[best].drop(labels=[frame.geometry.name]).to_dict()
        geometries = [frame.geometry.iloc[index] for index in component]
        if closing_radius > 0:
            geometry = unary_union([value.buffer(closing_radius) for value in geometries]).buffer(-closing_radius)
        else:
            geometry = unary_union(geometries)
        row["source_n"] = int(sum(int(frame.iloc[index].get("source_n", 1)) for index in component))
        row["geometry"] = repair_geometry(geometry)
        rows.append(row)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=frame.crs)


def merge_overlapping_fragments(
    frame: gpd.GeoDataFrame,
    class_field: str | None,
    allowed_classes: set[str],
    minimum_overlap_fraction: float,
) -> tuple[gpd.GeoDataFrame, int]:
    if frame.empty or not class_field or class_field not in frame.columns or not allowed_classes:
        return frame, 0
    output_frames = []
    merged_count = 0
    for class_name, group in frame.groupby(class_field, dropna=False, sort=False):
        group = group.reset_index(drop=True)
        if str(class_name) not in allowed_classes or len(group) < 2:
            output_frames.append(group)
            continue
        disjoint = DisjointSet(len(group))
        spatial_index = _spatial_index(group)
        for index, geometry in enumerate(group.geometry):
            for candidate in _candidates(group, spatial_index, geometry.bounds):
                candidate = int(candidate)
                if candidate <= index:
                    continue
                other = group.geometry.iloc[candidate]
                if not geometry.intersects(other):
                    continue
                intersection = geometry.intersection(other).area
                denominator = min(geometry.area, other.area)
                if denominator > 0 and intersection / denominator >= minimum_overlap_fraction:
                    disjoint.union(index, candidate)
        components = disjoint.components()
        merged_count += sum(len(component) - 1 for component in components)
        output_frames.append(_component_frame(group, components))
    result = gpd.GeoDataFrame(pd.concat(output_frames, ignore_index=True), geometry="geometry", crs=frame.crs)
    return result, merged_count


def close_water_gaps(
    frame: gpd.GeoDataFrame,
    class_field: str | None,
    water_class: str,
    radius: float,
) -> tuple[gpd.GeoDataFrame, int]:
    if frame.empty or radius <= 0 or not class_field or class_field not in frame.columns:
        return frame, 0
    water_mask = frame[class_field].astype(str) == water_class
    water = frame.loc[water_mask].reset_index(drop=True)
    other = frame.loc[~water_mask].copy()
    if water.empty:
        return frame, 0
    disjoint = DisjointSet(len(water))
    expanded = water.geometry.buffer(radius)
    expanded_frame = gpd.GeoDataFrame({"geometry": expanded}, geometry="geometry", crs=frame.crs)
    spatial_index = _spatial_index(expanded_frame)
    for index, geometry in enumerate(expanded_frame.geometry):
        for candidate in _candidates(expanded_frame, spatial_index, geometry.bounds):
            candidate = int(candidate)
            if candidate <= index:
                continue
            if geometry.intersects(expanded_frame.geometry.iloc[candidate]):
                disjoint.union(index, candidate)
    components = disjoint.components()
    merged_count = sum(len(component) - 1 for component in components)
    closed = _component_frame(water, components, closing_radius=radius)
    result = gpd.GeoDataFrame(pd.concat([other, closed], ignore_index=True), geometry="geometry", crs=frame.crs)
    return result, merged_count


def read_inputs(paths: list[Path]) -> gpd.GeoDataFrame:
    frames = []
    target_crs = None
    for path in paths:
        frame = gpd.read_file(path)
        if frame.crs is None:
            raise ValueError(f"Input has no CRS: {path}")
        if target_crs is None:
            target_crs = frame.crs
        elif frame.crs != target_crs:
            frame = frame.to_crs(target_crs)
        frame["source_file"] = path.name
        frame["source_n"] = 1
        frames.append(frame)
    result = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=target_crs)
    if getattr(result.crs, "is_geographic", False):
        raise ValueError("Postprocessing area and distance thresholds require a projected CRS")
    return result


def process(
    config: dict,
    input_paths: list[Path],
    output: Path,
    class_field: str | None,
    score_field: str,
    overwrite: bool,
) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
        output.unlink()
    settings = config["postprocessing"]
    frame = read_inputs(input_paths)
    input_count = len(frame)
    input_area = float(frame.geometry.area.sum())
    repaired_count = 0
    records = []
    filled_holes = 0
    filled_hole_area = 0.0
    for _, row in frame.iterrows():
        geometry = row.geometry
        if not geometry.is_valid:
            repaired_count += 1
        geometry = repair_geometry(geometry)
        geometry, local_count, local_area = fill_small_holes(
            geometry, float(settings["maximum_hole_area_m2"])
        )
        filled_holes += local_count
        filled_hole_area += local_area
        for part in polygon_parts(geometry):
            values = row.drop(labels=[frame.geometry.name]).to_dict()
            values["geometry"] = part
            records.append(values)
    work = gpd.GeoDataFrame(records, geometry="geometry", crs=frame.crs)
    after_explode = len(work)
    minimum_area = float(settings["minimum_polygon_area_m2"])
    small_mask = work.geometry.area < minimum_area
    small_removed = int(small_mask.sum())
    work = work.loc[~small_mask].reset_index(drop=True)
    work, nms_removed = class_aware_nms(
        work,
        float(settings["nms_iou_threshold"]),
        class_field,
        score_field,
    )
    work, overlap_merged = merge_overlapping_fragments(
        work,
        class_field,
        set(settings.get("merge_overlap_classes", [])),
        float(settings["minimum_overlap_fraction_of_smaller"]),
    )
    work, water_merged = close_water_gaps(
        work,
        class_field,
        str(settings["water_class_name"]),
        float(settings["water_closing_radius_m"]),
    )
    simplification = float(settings.get("polygon_simplification_tolerance_m", 0.0))
    if simplification > 0:
        work.geometry = work.geometry.simplify(simplification, preserve_topology=True)
    work["area_m2"] = work.geometry.area.astype(float)
    work = work[~work.geometry.is_empty & work.geometry.notna()].reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    work.to_file(output, driver="GPKG", layer="objects")
    save_json(
        output.with_suffix(".postprocess.json"),
        {
            "inputs": [str(path) for path in input_paths],
            "output": str(output),
            "class_field": class_field,
            "score_field": score_field,
            "rules": settings,
            "input_polygons": input_count,
            "input_total_area_m2": input_area,
            "repaired_invalid_geometries": repaired_count,
            "polygon_parts_after_repair": after_explode,
            "filled_holes": filled_holes,
            "filled_hole_area_m2": filled_hole_area,
            "small_fragments_removed": small_removed,
            "nms_duplicates_removed": nms_removed,
            "overlapping_fragments_merged": overlap_merged,
            "water_gap_fragments_merged": water_merged,
            "output_polygons": int(len(work)),
            "output_total_area_m2": float(work.geometry.area.sum()),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry-safe SAM2 object postprocessing")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-field", default=None)
    parser.add_argument("--score-field", default="pred_iou")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config, base = load_config(args.config)
    process(
        config,
        [Path(value).resolve() for value in args.inputs],
        Path(args.output).resolve(),
        args.class_field,
        args.score_field,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
