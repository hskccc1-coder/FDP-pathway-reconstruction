# -*- coding: utf-8 -*-
"""Generate D8 stream lines for a FlowAcc-threshold sensitivity experiment.

Fixed burning parameters:
    alpha (field ditch) = 0.8 m
    beta  (channel)     = 1.2 m
    gamma (pond)        = 0.0 m

The burned surface and D8 continuous-flow accumulation are calculated once.
Multiple stream-line outputs are then extracted from the same accumulation
raster. Temporary rasters are deleted after all thresholds are complete.
"""

from __future__ import print_function

import gc
import glob
import os
import shutil
import time

import arcpy
from arcpy.sa import Con, DeriveContinuousFlow, IsNull, Raster, StreamToFeature


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DEM = os.environ.get(
    "FDP_RAW_DEM", os.path.join(PROJECT_ROOT, "data", "external", "raw_dem.tif")
)
FIELD_SHP = os.path.join(
    PROJECT_ROOT, "data", "fdp_constraints", "burn_features", "field_ditch_burn.shp"
)
CHANNEL_SHP = os.path.join(
    PROJECT_ROOT, "data", "fdp_constraints", "burn_features", "channel_burn.shp"
)

ALPHA_M = 0.8
BETA_M = 1.2
GAMMA_M = 0.0
THRESHOLDS = (
    list(range(400, 2401, 200))
    + [3000, 4000, 5000, 6000, 8000, 10000]
    + [15000, 20000, 30000, 50000, 80000, 120000]
)

OUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    "pathway_metrics",
    "dem_routing_runtime",
    u"汇流累积阈值敏感性_A0p8_B1p2_C0",
)
LINE_DIR = os.path.join(OUT_ROOT, "D8_lines")
SCRATCH_ROOT = os.path.join(OUT_ROOT, "_scratch_delete_ok")


def log(message):
    text = str(message)
    print(text, flush=True)
    try:
        arcpy.AddMessage(text)
    except Exception:
        pass


def delete_path(path):
    try:
        if arcpy.Exists(path):
            arcpy.management.Delete(path)
        elif os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def delete_shapefile(path):
    delete_path(path)
    stem = os.path.splitext(path)[0]
    for item in glob.glob(stem + ".*"):
        try:
            os.remove(item)
        except OSError:
            pass


def output_line(threshold):
    return os.path.join(
        LINE_DIR,
        "StreamLine_D8_A0p80_B1p20_C0p00_thr%d.shp" % int(threshold),
    )


def output_valid(path):
    try:
        return arcpy.Exists(path) and int(arcpy.management.GetCount(path)[0]) > 0
    except Exception:
        return False


def create_binary_mask(feature_path, name):
    oid_field = arcpy.Describe(feature_path).OIDFieldName
    oid_raster = "memory\\%s_oid" % name
    mask_path = "memory\\%s_mask" % name
    for path in (oid_raster, mask_path):
        if arcpy.Exists(path):
            arcpy.management.Delete(path)
    arcpy.conversion.FeatureToRaster(
        in_features=feature_path,
        field=oid_field,
        out_raster=oid_raster,
        cell_size=RAW_DEM,
    )
    oid = Raster(oid_raster)
    mask = Con(IsNull(oid), 0, 1)
    mask.save(mask_path)
    del mask, oid
    arcpy.management.Delete(oid_raster)
    return Raster(mask_path)


def clean_memory():
    for workspace in ("memory", "in_memory"):
        try:
            arcpy.management.Delete(workspace)
        except Exception:
            pass
    try:
        arcpy.management.ClearWorkspaceCache()
    except Exception:
        pass


def main():
    for path in (RAW_DEM, FIELD_SHP, CHANNEL_SHP):
        if not arcpy.Exists(path):
            raise FileNotFoundError("Input does not exist: %s" % path)

    os.makedirs(LINE_DIR, exist_ok=True)
    os.makedirs(SCRATCH_ROOT, exist_ok=True)

    pending = [value for value in THRESHOLDS if not output_valid(output_line(value))]
    log("Fixed burns: alpha=%.1f m, beta=%.1f m, gamma=%.1f m" % (ALPHA_M, BETA_M, GAMMA_M))
    log("FlowAcc thresholds: %s" % ", ".join(str(value) for value in THRESHOLDS))
    log("Pending D8 outputs: %d/%d" % (len(pending), len(THRESHOLDS)))
    if not pending:
        log("All D8 threshold outputs already exist.")
        return

    spatial_status = arcpy.CheckExtension("Spatial")
    if spatial_status not in ("Available", "CheckedOut"):
        raise RuntimeError("Spatial Analyst unavailable: %s" % spatial_status)
    checked_out_here = spatial_status == "Available"
    if checked_out_here:
        arcpy.CheckOutExtension("Spatial")

    raw_dem = field_mask = channel_mask = burned_dem = flow_acc = None
    start_time = time.time()
    try:
        arcpy.env.overwriteOutput = True
        arcpy.env.addOutputsToMap = False
        arcpy.env.snapRaster = RAW_DEM
        arcpy.env.cellSize = RAW_DEM
        arcpy.env.extent = RAW_DEM
        arcpy.env.mask = RAW_DEM
        arcpy.env.outputCoordinateSystem = arcpy.Describe(RAW_DEM).spatialReference
        arcpy.env.scratchWorkspace = SCRATCH_ROOT

        clean_memory()
        log("Creating field/channel masks...")
        field_mask = create_binary_mask(FIELD_SHP, "field_fixed")
        channel_mask = create_binary_mask(CHANNEL_SHP, "channel_fixed")
        raw_dem = Raster(RAW_DEM)
        burned_dem = raw_dem - field_mask * ALPHA_M - channel_mask * BETA_M

        flow_direction_path = os.path.join(SCRATCH_ROOT, "flow_direction_d8.tif")
        flow_acc_path = os.path.join(SCRATCH_ROOT, "flow_accumulation_d8.tif")
        delete_path(flow_direction_path)
        delete_path(flow_acc_path)

        log("Calculating D8 continuous-flow accumulation once...")
        flow_acc = DeriveContinuousFlow(
            in_surface_raster=burned_dem,
            in_depressions_data="",
            in_weight_raster="",
            out_flow_direction_raster=flow_direction_path,
            flow_direction_type="D8",
            force_flow="NORMAL",
        )
        flow_acc.save(flow_acc_path)
        del flow_acc
        flow_acc = Raster(flow_acc_path)

        for index, threshold in enumerate(pending, 1):
            output_shp = output_line(threshold)
            delete_shapefile(output_shp)
            stream_raster = None
            try:
                log("[%d/%d] Extracting D8 stream line: FlowAcc > %d" % (index, len(pending), threshold))
                stream_raster = Con(flow_acc > float(threshold), 1)
                StreamToFeature(
                    in_stream_raster=stream_raster,
                    in_flow_direction_raster=flow_direction_path,
                    out_polyline_features=output_shp,
                    simplify="NO_SIMPLIFY",
                )
                if not output_valid(output_shp):
                    raise RuntimeError("Empty stream-line output: %s" % output_shp)
                log("  features=%s" % arcpy.management.GetCount(output_shp)[0])
            finally:
                if stream_raster is not None:
                    del stream_raster
                gc.collect()
                try:
                    arcpy.management.ClearWorkspaceCache()
                except Exception:
                    pass

        elapsed = (time.time() - start_time) / 60.0
        log("D8 threshold generation completed in %.2f minutes." % elapsed)
    finally:
        flow_acc = None
        burned_dem = None
        raw_dem = None
        field_mask = None
        channel_mask = None
        gc.collect()
        clean_memory()
        # Remove only known disposable rasters. Final SHP outputs are preserved.
        delete_path(os.path.join(SCRATCH_ROOT, "flow_accumulation_d8.tif"))
        delete_path(os.path.join(SCRATCH_ROOT, "flow_direction_d8.tif"))
        if checked_out_here:
            try:
                arcpy.CheckInExtension("Spatial")
            except Exception:
                pass


if __name__ == "__main__":
    main()
