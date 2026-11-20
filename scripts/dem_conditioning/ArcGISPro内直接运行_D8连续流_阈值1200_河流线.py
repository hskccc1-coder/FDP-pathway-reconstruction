# -*- coding: utf-8 -*-
"""
在 ArcGIS Pro 内部直接运行的 ArcPy 脚本。

算法：
    烧录 DEM = 原始 DEM - A * 田沟掩膜 - B * 沟渠掩膜
    -> DeriveContinuousFlow (D8)
    -> FlowAcc > 1200
    -> StreamToFeature

运行方式：在 ArcGIS Pro 的 Python 环境中运行本文件。大型原始DEM可通过
环境变量 FDP_RAW_DEM 指定；未指定时读取 data/external/raw_dem.tif。

脚本只保留最终 SHP，中间的烧录 DEM、FlowAcc、D8 流向和河网
栅格都会删除。
"""

import gc
import glob
import os
import shutil
import time

import arcpy
from arcpy.sa import Con, DeriveContinuousFlow, IsNull, Raster, StreamToFeature


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 1. 只需要在这里修改配置
# ============================================================
# SMOKE_TEST：只运行 A=0, B=0，用于首次测试。
# SINGLE：只运行下面 SINGLE_A、SINGLE_B 指定的一组。
# FORMAL_2D：运行 A/B=0~2 m、步长 0.2 m 的 121 组。
RUN_MODE = "FORMAL_2D"

# SINGLE 模式参数；1.4/2.0 是以前 MFD 实验的总体 recall 最佳组合。
SINGLE_A = 1.4
SINGLE_B = 2.0

RAW_DEM = os.environ.get(
    "FDP_RAW_DEM", os.path.join(PROJECT_ROOT, "data", "external", "raw_dem.tif")
)
FIELD_SHP = os.path.join(
    PROJECT_ROOT, "data", "fdp_constraints", "burn_features", "field_ditch_burn.shp"
)
CHANNEL_SHP = os.path.join(
    PROJECT_ROOT, "data", "fdp_constraints", "burn_features", "channel_burn.shp"
)

OUT_ROOT = os.environ.get(
    "FDP_D8_OUTPUT",
    os.path.join(PROJECT_ROOT, "results", "pathway_metrics", "dem_routing_runtime", "d8"),
)
SCRATCH_ROOT = os.path.join(OUT_ROOT, "_scratch_delete_ok")

FLOWACC_THRESHOLD = 1200.0
# False 与之前评价的 FlowAcc > 1200 一致。
# 如需 FlowAcc >= 1200，改为 True。
THRESHOLD_INCLUSIVE = False

BURN_MIN = 0.0
BURN_MAX = 2.0
BURN_STEP = 0.2

# 默认不把 121 个输出全部加入当前地图。
# SINGLE/SMOKE_TEST 时如需自动加入地图，改为 True。
ADD_OUTPUT_TO_CURRENT_MAP = False


# ============================================================
# 2. 通用函数
# ============================================================
def log(message):
    text = str(message)
    print(text)
    try:
        arcpy.AddMessage(text)
    except Exception:
        pass


def log_error(message):
    text = str(message)
    print(text)
    try:
        arcpy.AddError(text)
    except Exception:
        pass


def make_values(start, stop, step):
    count = int(round((stop - start) / step))
    return [round(start + i * step, 2) for i in range(count + 1)]


def value_tag(value):
    return f"{float(value):.2f}".replace(".", "p")


def output_name(a, b):
    return f"StreamLine_D8_thr1200_A{value_tag(a)}_B{value_tag(b)}.shp"


def delete_path(path):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def delete_shapefile(shp_path):
    if arcpy.Exists(shp_path):
        try:
            arcpy.management.Delete(shp_path)
            return
        except Exception:
            pass
    for path in glob.glob(os.path.splitext(shp_path)[0] + ".*"):
        delete_path(path)


def output_is_valid(shp_path):
    try:
        return arcpy.Exists(shp_path) and int(arcpy.management.GetCount(shp_path)[0]) > 0
    except Exception:
        return False


def clean_scratch():
    for workspace in ("memory", "in_memory"):
        try:
            arcpy.management.Delete(workspace)
        except Exception:
            pass
    try:
        arcpy.management.ClearWorkspaceCache()
    except Exception:
        pass

    os.makedirs(SCRATCH_ROOT, exist_ok=True)
    for name in os.listdir(SCRATCH_ROOT):
        delete_path(os.path.join(SCRATCH_ROOT, name))


def create_binary_mask(feature_path, name):
    oid_field = arcpy.Describe(feature_path).OIDFieldName
    oid_raster = f"memory\\{name}_oid"
    mask_path = f"memory\\{name}_mask"

    for path in (oid_raster, mask_path):
        if arcpy.Exists(path):
            arcpy.management.Delete(path)

    log(f"正在创建 {name} 0/1 掩膜...")
    arcpy.conversion.FeatureToRaster(
        in_features=feature_path,
        field=oid_field,
        out_raster=oid_raster,
        cell_size=RAW_DEM,
    )

    oid = Raster(oid_raster)
    mask = Con(IsNull(oid), 0, 1)
    mask.save(mask_path)
    del mask
    del oid
    arcpy.management.Delete(oid_raster)
    return Raster(mask_path)


def build_tasks():
    mode = RUN_MODE.upper()
    if mode == "SMOKE_TEST":
        return "00_SmokeTest", [(0.0, 0.0)]
    if mode == "SINGLE":
        return "01_Single", [(float(SINGLE_A), float(SINGLE_B))]
    if mode == "FORMAL_2D":
        values = make_values(BURN_MIN, BURN_MAX, BURN_STEP)
        return (
            "02_Formal_2D_A0_2_B0_2_step0p2",
            [(a, b) for a in values for b in values],
        )
    raise ValueError("RUN_MODE 只能是 SMOKE_TEST、SINGLE 或 FORMAL_2D。")


def add_to_current_map(shp_path):
    if not ADD_OUTPUT_TO_CURRENT_MAP:
        return
    try:
        project = arcpy.mp.ArcGISProject("CURRENT")
        active_map = project.activeMap
        if active_map is not None:
            active_map.addDataFromPath(shp_path)
            log(f"已加入当前地图: {shp_path}")
    except Exception as exc:
        log(f"输出已生成，但加入当前地图失败: {exc}")


# ============================================================
# 3. 主程序
# ============================================================
def main():
    for input_path in (RAW_DEM, FIELD_SHP, CHANNEL_SHP):
        if not arcpy.Exists(input_path):
            raise FileNotFoundError(f"输入不存在: {input_path}")

    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(SCRATCH_ROOT, exist_ok=True)

    spatial_status = arcpy.CheckExtension("Spatial")
    if spatial_status not in ("Available", "CheckedOut"):
        raise RuntimeError(f"Spatial Analyst 不可用: {spatial_status}")

    checked_out_here = spatial_status == "Available"
    if checked_out_here:
        arcpy.CheckOutExtension("Spatial")

    try:
        arcpy.env.overwriteOutput = True
        arcpy.env.addOutputsToMap = False
        arcpy.env.snapRaster = RAW_DEM
        arcpy.env.cellSize = RAW_DEM
        arcpy.env.extent = RAW_DEM
        arcpy.env.mask = RAW_DEM
        arcpy.env.outputCoordinateSystem = arcpy.Describe(RAW_DEM).spatialReference
        arcpy.env.scratchWorkspace = SCRATCH_ROOT

        output_subdir, tasks = build_tasks()
        output_dir = os.path.join(OUT_ROOT, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        comparator = ">=" if THRESHOLD_INCLUSIVE else ">"
        log("=" * 55)
        log("ArcGIS Pro 内运行：D8 派生连续流生成河流线")
        log(f"运行模式: {RUN_MODE}")
        log(f"河网条件: FlowAcc {comparator} {FLOWACC_THRESHOLD:g}")
        log(f"任务数: {len(tasks)}")
        log(f"输出目录: {output_dir}")
        log("=" * 55)

        clean_scratch()
        field_mask = create_binary_mask(FIELD_SHP, "field")
        channel_mask = create_binary_mask(CHANNEL_SHP, "channel")
        raw_dem = Raster(RAW_DEM)

        success_count = 0
        skip_count = 0
        failed_count = 0
        start_time = time.time()

        for index, (a, b) in enumerate(tasks, 1):
            output_shp = os.path.join(output_dir, output_name(a, b))
            log(f"[{index}/{len(tasks)}] A={a:.2f} m, B={b:.2f} m")

            if output_is_valid(output_shp):
                log("    已存在，跳过。")
                skip_count += 1
                continue

            delete_shapefile(output_shp)
            task_dir = os.path.join(
                SCRATCH_ROOT,
                f"task_A{value_tag(a)}_B{value_tag(b)}",
            )
            delete_path(task_dir)
            os.makedirs(task_dir, exist_ok=True)

            burned_dem = None
            flow_acc = None
            stream_raster = None
            try:
                arcpy.env.scratchWorkspace = task_dir
                flow_direction = os.path.join(task_dir, "flowdir_d8.tif")

                burned_dem = (
                    raw_dem
                    - field_mask * float(a)
                    - channel_mask * float(b)
                )

                flow_acc = DeriveContinuousFlow(
                    in_surface_raster=burned_dem,
                    in_depressions_data="",
                    in_weight_raster="",
                    out_flow_direction_raster=flow_direction,
                    flow_direction_type="D8",
                    force_flow="NORMAL",
                )

                if THRESHOLD_INCLUSIVE:
                    stream_raster = Con(flow_acc >= FLOWACC_THRESHOLD, 1)
                else:
                    stream_raster = Con(flow_acc > FLOWACC_THRESHOLD, 1)

                StreamToFeature(
                    in_stream_raster=stream_raster,
                    in_flow_direction_raster=flow_direction,
                    out_polyline_features=output_shp,
                    simplify="NO_SIMPLIFY",
                )

                if not output_is_valid(output_shp):
                    raise RuntimeError("河流线输出为空或无法读取。")

                feature_count = int(arcpy.management.GetCount(output_shp)[0])
                log(f"    完成，线要素数: {feature_count}")
                success_count += 1
                add_to_current_map(output_shp)

            except Exception as exc:
                failed_count += 1
                log_error(f"    失败: {exc}")
                try:
                    log_error(arcpy.GetMessages(2))
                except Exception:
                    pass
                delete_shapefile(output_shp)

            finally:
                if stream_raster is not None:
                    del stream_raster
                if flow_acc is not None:
                    del flow_acc
                if burned_dem is not None:
                    del burned_dem
                gc.collect()

                try:
                    arcpy.management.ClearWorkspaceCache()
                except Exception:
                    pass
                arcpy.env.scratchWorkspace = SCRATCH_ROOT
                delete_path(task_dir)

        del raw_dem
        del field_mask
        del channel_mask
        clean_scratch()

        elapsed = (time.time() - start_time) / 60.0
        log("=" * 55)
        log("处理完成")
        log(f"新完成: {success_count}")
        log(f"跳过: {skip_count}")
        log(f"失败: {failed_count}")
        log(f"耗时: {elapsed:.2f} 分钟")
        log(f"河流线目录: {output_dir}")
        log("=" * 55)

    finally:
        if checked_out_here:
            try:
                arcpy.CheckInExtension("Spatial")
            except Exception:
                pass


main()
