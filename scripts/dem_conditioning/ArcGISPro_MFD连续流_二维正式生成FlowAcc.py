# -*- coding: utf-8 -*-
"""
Generate MFD Derive Continuous Flow rasters for the formal 2D burn-depth test.

Parameters:
    Field ditch A:   0.0 to 2.0 m, step 0.2 m
    Channel B:       0.0 to 2.0 m, step 0.2 m
    Flow direction:  MFD

Only final FlowAcc TIF rasters are saved. Burned DEM, flow direction raster,
and task scratch files are not kept.

Run this file with the ArcGIS Pro Python environment. Set FDP_RAW_DEM to
override the default data/external/raw_dem.tif input.
"""

import gc
import glob
import os
import shutil
import time

import arcpy
from arcpy.sa import Con, DeriveContinuousFlow, IsNull, Raster


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 1. Input / output paths
# ============================================================
raw_dem_path = os.environ.get(
    "FDP_RAW_DEM", os.path.join(PROJECT_ROOT, "data", "external", "raw_dem.tif")
)
field_shp = os.path.join(
    PROJECT_ROOT, "data", "fdp_constraints", "burn_features", "field_ditch_burn.shp"
)
channel_shp = os.path.join(
    PROJECT_ROOT, "data", "fdp_constraints", "burn_features", "channel_burn.shp"
)

out_root = os.environ.get(
    "FDP_MFD_OUTPUT",
    os.path.join(PROJECT_ROOT, "results", "pathway_metrics", "dem_routing_runtime", "mfd"),
)
flowacc_dir = os.path.join(out_root, "FlowAcc_MFD")
scratch_root = os.path.join(out_root, "_scratch_delete_ok")


# ============================================================
# 2. Formal experiment parameters
# ============================================================
burn_min = 0.0
burn_max = 2.0
burn_step = 0.2

flow_direction_type = "MFD"
force_flow = "NORMAL"


# ============================================================
# 3. Helpers
# ============================================================
def make_values(start, stop, step):
    count = int(round((stop - start) / step))
    return [round(start + i * step, 2) for i in range(count + 1)]


def value_tag(value):
    return f"{float(value):.2f}".replace(".", "p")


def output_name(a, b):
    return f"FlowAcc_MFD_A{value_tag(a)}_B{value_tag(b)}.tif"


def output_path(a, b):
    return os.path.join(flowacc_dir, output_name(a, b))


def delete_path(path):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def delete_raster(path):
    if arcpy.Exists(path):
        try:
            arcpy.management.Delete(path)
            return
        except Exception:
            pass

    for item in glob.glob(os.path.splitext(path)[0] + ".*"):
        delete_path(item)


def is_completed_raster(path):
    if not arcpy.Exists(path):
        return False
    try:
        desc = arcpy.Describe(path)
        return int(desc.width) > 0 and int(desc.height) > 0
    except Exception:
        return False


def clean_scratch_root():
    for workspace in ["memory", "in_memory"]:
        try:
            arcpy.management.Delete(workspace)
        except Exception:
            pass

    try:
        arcpy.management.ClearWorkspaceCache()
    except Exception:
        pass

    os.makedirs(scratch_root, exist_ok=True)
    for name in os.listdir(scratch_root):
        delete_path(os.path.join(scratch_root, name))


def make_task_scratch(a, b):
    path = os.path.join(scratch_root, f"task_A{value_tag(a)}_B{value_tag(b)}")
    delete_path(path)
    os.makedirs(path, exist_ok=True)
    return path


def create_binary_mask(feature_path, name):
    desc = arcpy.Describe(feature_path)
    oid_field = desc.OIDFieldName

    temp_raster = f"memory\\{name}_oid"
    mask_path = f"memory\\{name}_mask"

    for item in [temp_raster, mask_path]:
        if arcpy.Exists(item):
            arcpy.management.Delete(item)

    print(f"Creating {name} mask...")
    arcpy.conversion.FeatureToRaster(
        in_features=feature_path,
        field=oid_field,
        out_raster=temp_raster,
        cell_size=raw_dem_path,
    )

    temp = Raster(temp_raster)
    mask = Con(IsNull(temp), 0, 1)
    mask.save(mask_path)

    del mask
    del temp
    arcpy.management.Delete(temp_raster)
    return Raster(mask_path)


def build_tasks():
    values = make_values(burn_min, burn_max, burn_step)
    return [(a, b) for a in values for b in values]


# ============================================================
# 4. Main
# ============================================================
def main():
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(flowacc_dir, exist_ok=True)
    os.makedirs(scratch_root, exist_ok=True)

    print("ArcGIS Pro:", arcpy.GetInstallInfo().get("Version"))
    print("Spatial Analyst:", arcpy.CheckExtension("Spatial"))
    arcpy.CheckOutExtension("Spatial")

    arcpy.env.overwriteOutput = True
    arcpy.env.addOutputsToMap = False

    dem_desc = arcpy.Describe(raw_dem_path)
    arcpy.env.snapRaster = raw_dem_path
    arcpy.env.cellSize = raw_dem_path
    arcpy.env.extent = raw_dem_path
    arcpy.env.mask = raw_dem_path
    arcpy.env.outputCoordinateSystem = dem_desc.spatialReference
    arcpy.env.scratchWorkspace = scratch_root

    tasks = build_tasks()
    print("\n========================================")
    print("MFD Derive Continuous Flow formal 2D test")
    print("========================================")
    print("A field ditch values:", make_values(burn_min, burn_max, burn_step))
    print("B channel values:", make_values(burn_min, burn_max, burn_step))
    print("flow_direction_type:", flow_direction_type)
    print("force_flow:", force_flow)
    print("task_count:", len(tasks))
    print("output:", flowacc_dir)
    print("Only final FlowAcc TIFs will be saved.")
    print("Existing completed TIFs will be skipped.")
    print("========================================\n")

    clean_scratch_root()
    mask_field = create_binary_mask(field_shp, "field")
    mask_channel = create_binary_mask(channel_shp, "channel")
    raw_dem = Raster(raw_dem_path)

    success_count = 0
    skip_count = 0
    failed_count = 0
    start_time = time.time()

    for index, (a, b) in enumerate(tasks, start=1):
        out_tif = output_path(a, b)
        print(f"[{index}/{len(tasks)}] A={a:.2f} m, B={b:.2f} m")

        if is_completed_raster(out_tif):
            print("    Existing FlowAcc found, skip.")
            skip_count += 1
            continue

        delete_raster(out_tif)
        task_scratch = None
        burned_dem = None
        flowacc = None

        try:
            task_scratch = make_task_scratch(a, b)
            arcpy.env.scratchWorkspace = task_scratch

            burned_dem = raw_dem - (mask_field * float(a)) - (mask_channel * float(b))

            flowacc = DeriveContinuousFlow(
                in_surface_raster=burned_dem,
                in_depressions_data="",
                in_weight_raster="",
                out_flow_direction_raster="",
                flow_direction_type=flow_direction_type,
                force_flow=force_flow,
            )
            flowacc.save(out_tif)

            if not is_completed_raster(out_tif):
                raise RuntimeError("FlowAcc output is empty or unreadable.")

            success_count += 1
            print("    Done:", out_tif)

        except Exception as exc:
            failed_count += 1
            print("    Failed:", exc)
            try:
                print(arcpy.GetMessages(2))
            except Exception:
                pass
            delete_raster(out_tif)

        finally:
            if flowacc is not None:
                del flowacc
            if burned_dem is not None:
                del burned_dem
            gc.collect()
            arcpy.management.ClearWorkspaceCache()
            arcpy.env.scratchWorkspace = scratch_root
            if task_scratch is not None:
                delete_path(task_scratch)

    del raw_dem
    del mask_field
    del mask_channel
    clean_scratch_root()

    elapsed = time.time() - start_time
    print("\n========================================")
    print("Finished")
    print("========================================")
    print("New success:", success_count)
    print("Skipped:", skip_count)
    print("Failed:", failed_count)
    print("Elapsed minutes:", round(elapsed / 60.0, 2))
    print("FlowAcc folder:", flowacc_dir)


if __name__ == "__main__":
    main()
