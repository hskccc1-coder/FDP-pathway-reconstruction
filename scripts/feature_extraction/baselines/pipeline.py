from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any, Iterable, Iterator


def _fix_conda_geo_environment() -> None:
    """Prefer the active conda environment's GDAL/PROJ data over global stale paths."""
    env_root = Path(sys.executable).resolve().parent
    proj = env_root / "Library" / "share" / "proj"
    gdal = env_root / "Library" / "share" / "gdal"
    if (proj / "proj.db").is_file():
        os.environ["PROJ_LIB"] = str(proj)
    if gdal.is_dir():
        os.environ["GDAL_DATA"] = str(gdal)


_fix_conda_geo_environment()

import geopandas as gpd
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds, transform as window_transform
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from .models import build_model, freeze_encoder_batchnorm
except ImportError:  # Direct script execution from this directory.
    from models import build_model, freeze_encoder_batchnorm


IGNORE_INDEX = 255
LABEL_CACHE_VERSION = "v2_valid_region_clip"
MODEL_NAMES = ("unetpp", "deeplabv3plus", "seanet")
FEATURE_NAMES = ("rgb", "slope")


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(
        config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    )
    return cfg, config_path


def _resolve(config_dir: str | Path, value: str | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else (Path(config_dir) / path).resolve()


def output_root(cfg: dict[str, Any]) -> Path:
    root = _resolve(cfg["_config_dir"], cfg.get("output_dir", "outputs"))
    assert root is not None
    return root


def split_reference_path(cfg: dict[str, Any], split_name: str) -> Path:
    """Use the split RGB raster as the canonical geospatial grid."""
    value = cfg["splits"][split_name]["features"]["rgb"]
    path = _resolve(cfg["_config_dir"], value)
    if path is None:
        raise ValueError(f"splits.{split_name}.features.rgb is required as the reference grid")
    return path


def config_fingerprint(cfg: dict[str, Any]) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    payload = json.dumps(clean, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def seed_everything(seed: int, cfg: dict[str, Any] | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    runtime = (cfg or {}).get("runtime", {})
    deterministic = bool(runtime.get("deterministic", True))
    benchmark = bool(runtime.get("cudnn_benchmark", False)) and not deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
    if torch.cuda.is_available():
        allow_tf32 = bool(runtime.get("allow_tf32", False))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def _precision(cfg: dict[str, Any], device: torch.device) -> tuple[bool, torch.dtype | None]:
    name = str(cfg.get("runtime", {}).get("precision", "fp32")).lower()
    if device.type != "cuda" or name == "fp32":
        return False, None
    if name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("runtime.precision=bf16 but this CUDA device does not support BF16")
        return True, torch.bfloat16
    if name == "fp16":
        return True, torch.float16
    raise ValueError("runtime.precision must be fp32, fp16 or bf16")


def _autocast(cfg: dict[str, Any], device: torch.device):
    enabled, dtype = _precision(cfg, device)
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _channels_last(cfg: dict[str, Any], device: torch.device) -> bool:
    return device.type == "cuda" and bool(cfg.get("runtime", {}).get("channels_last", False))


def validate_config(cfg: dict[str, Any], stages: Iterable[str]) -> None:
    stages = set(stages)
    errors: list[str] = []
    classes = cfg.get("classes", [])
    ids = [int(item["id"]) for item in classes]
    if not classes or sorted(ids) != list(range(1, len(classes) + 1)):
        errors.append("classes must use contiguous foreground ids starting at 1")
    if cfg.get("selection", {}).get("primary_metric") != "mean_iou":
        errors.append("selection.primary_metric must be mean_iou for this experiment")
    if set(cfg.get("models", [])) != set(MODEL_NAMES):
        errors.append("models must contain exactly unetpp, deeplabv3plus and seanet")
    initialization_policy = cfg.get("initialization_policy", "random_all")
    if initialization_policy not in {"random_all", "official_pretrained"}:
        errors.append("initialization_policy must be random_all or official_pretrained")
    if str(cfg.get("runtime", {}).get("precision", "fp32")).lower() not in {"fp32", "fp16", "bf16"}:
        errors.append("runtime.precision must be fp32, fp16 or bf16")
    scales = [int(v) for v in cfg.get("scales", [])]
    if sorted(scales, reverse=True) != [4096, 2048, 1024, 512]:
        errors.append("scales must contain exactly 4096, 2048, 1024 and 512")
    if set(cfg.get("features", {}).keys()) != set(FEATURE_NAMES):
        errors.append("features must contain exactly rgb and slope")
    for feature_name, feature_cfg in cfg.get("features", {}).items():
        bands = feature_cfg.get("bands", [])
        mean, std = feature_cfg.get("mean", []), feature_cfg.get("std", [])
        if len(bands) != 3 or len(mean) != 3 or len(std) != 3:
            errors.append(f"features.{feature_name} must define three bands, means and stds")
    source_dir = _resolve(cfg.get("_config_dir", "."), cfg.get("seanet", {}).get("source_dir"))
    if source_dir is None or not (source_dir / "models.py").is_file():
        errors.append(f"seanet.source_dir must contain the official models.py: {source_dir}")
    seanet_cfg = cfg.get("seanet", {})
    if "encoder_name" in cfg:
        errors.append("encoder_name is replaced by encoder_by_model")
    encoders = cfg.get("encoder_by_model", {})
    if set(encoders) != {"unetpp", "deeplabv3plus"}:
        errors.append("encoder_by_model must define unetpp and deeplabv3plus")
    if initialization_policy == "official_pretrained":
        source = seanet_cfg.get("pretrained_source")
        if source == "official_vgg16_mat":
            # The SEANet/RCF .mat weights come from Caffe (BGR, mean subtraction on 0-255);
            # this pipeline feeds ImageNet-normalised RGB, so loading them would be silently wrong.
            errors.append("seanet.pretrained_source=official_vgg16_mat needs Caffe BGR preprocessing, which is not implemented")
        elif source != "torchvision_vgg16_imagenet":
            errors.append("seanet.pretrained_source must be torchvision_vgg16_imagenet")
        rgb = cfg.get("features", {}).get("rgb", {})
        if [round(v, 3) for v in rgb.get("mean", [])] != [0.485, 0.456, 0.406] or \
                [round(v, 3) for v in rgb.get("std", [])] != [0.229, 0.224, 0.225]:
            errors.append("official_pretrained requires ImageNet mean/std for the rgb feature")
    needed_splits = {"audit": ("train", "validation", "test"), "train": ("train", "validation"),
                     "validate": ("validation",), "test": ("test",)}
    required = set()
    for stage in stages:
        required.update(needed_splits.get(stage, ()))
    for split_name in required:
        split = cfg.get("splits", {}).get(split_name, {})
        labels_value = split.get("labels")
        label_sources = split.get("label_sources", [])
        if not labels_value and not label_sources:
            errors.append(f"splits.{split_name} requires labels or label_sources")
        if labels_value:
            if str(labels_value).startswith("REQUIRED_"):
                errors.append(f"splits.{split_name}.labels is required")
            else:
                path = _resolve(cfg["_config_dir"], labels_value)
                if path is None or not path.exists():
                    errors.append(f"missing splits.{split_name}.labels: {path}")
        for index, source in enumerate(label_sources):
            value = source.get("path")
            path = _resolve(cfg["_config_dir"], value)
            if not value or str(value).startswith("REQUIRED_") or path is None or not path.exists():
                errors.append(f"missing splits.{split_name}.label_sources[{index}].path: {path}")
            if int(source.get("class_id", -1)) not in {0, *[int(item["id"]) for item in classes]}:
                errors.append(f"invalid class_id in splits.{split_name}.label_sources[{index}]")
        valid_region = split.get("valid_region")
        if valid_region:
            region_path = _resolve(cfg["_config_dir"], valid_region)
            if region_path is None or not region_path.exists():
                errors.append(f"missing splits.{split_name}.valid_region: {region_path}")
        stratify_region = split.get("stratify_region")
        if stratify_region:
            stratify_path = _resolve(cfg["_config_dir"], stratify_region)
            if stratify_path is None or not stratify_path.exists():
                errors.append(f"missing splits.{split_name}.stratify_region: {stratify_path}")
        for feature in FEATURE_NAMES:
            value = split.get("features", {}).get(feature)
            if not value or str(value).startswith("REQUIRED_"):
                errors.append(f"splits.{split_name}.features.{feature} is required")
                continue
            path = _resolve(cfg["_config_dir"], value)
            if path is None or not path.exists():
                errors.append(f"missing splits.{split_name}.features.{feature}: {path}")
    if errors:
        raise ValueError("Invalid experiment configuration:\n- " + "\n- ".join(errors))


def audit_spatial_splits(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fail when labelled train/validation/test areas overlap by positive area."""
    if not bool(cfg.get("audit", {}).get("enforce", True)):
        result = {
            "status": "skipped_by_user_configuration",
            "config_fingerprint": config_fingerprint(cfg),
            "warning": "Spatial leakage checks were explicitly disabled; test overlap may exist.",
        }
        save_json(output_root(cfg) / "audit.json", result)
        return result
    reports: list[dict[str, Any]] = []
    loaded: dict[str, gpd.GeoDataFrame] = {}
    for split_name in ("train", "validation", "test"):
        split = cfg["splits"][split_name]
        audit_value = split.get("valid_region") or split.get("labels")
        if not audit_value:
            raise ValueError(f"Cannot audit {split_name}: no valid_region or labels")
        path = _resolve(cfg["_config_dir"], audit_value)
        assert path is not None
        frame = gpd.read_file(path)
        if frame.crs is None:
            raise ValueError(f"Labels have no CRS: {path}")
        frame = frame.loc[~frame.geometry.is_empty & frame.geometry.notna(), ["geometry"]]
        loaded[split_name] = frame
    names = list(loaded)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            left, right = loaded[left_name], loaded[right_name]
            right = right.to_crs(left.crs)
            intersection = left.geometry.unary_union.intersection(right.geometry.unary_union)
            area = float(intersection.area)
            reports.append({"left": left_name, "right": right_name, "intersection_area": area})
            if area > float(cfg.get("audit", {}).get("max_label_overlap_area", 0.0)):
                raise RuntimeError(
                    f"Spatial leakage: {left_name} and {right_name} labels overlap by {area:.6f} "
                    f"square CRS units. Supply disjoint regions before training."
                )
    result = {"status": "passed", "config_fingerprint": config_fingerprint(cfg), "pairs": reports}
    save_json(output_root(cfg) / "audit.json", result)
    return result


def _positions(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    positions = list(range(0, length - size + 1, stride))
    final = length - size
    if positions[-1] != final:
        positions.append(final)
    return positions


def iter_windows(width: int, height: int, size: int, overlap: float) -> Iterator[Window]:
    stride = max(1, int(round(size * (1.0 - overlap))))
    for row in _positions(height, size, stride):
        for col in _positions(width, size, stride):
            yield Window(col, row, min(size, width - col), min(size, height - row))


def _exact_grid(dataset: rasterio.DatasetReader, reference: rasterio.DatasetReader) -> bool:
    return (
        dataset.crs == reference.crs
        and dataset.transform.almost_equals(reference.transform)
        and dataset.width == reference.width
        and dataset.height == reference.height
    )


class AlignedRaster:
    def __init__(self, reference_path: Path, feature_path: Path, bands: list[int]) -> None:
        self.stack = ExitStack()
        self.reference = self.stack.enter_context(rasterio.open(reference_path))
        self.source = self.stack.enter_context(rasterio.open(feature_path))
        if _exact_grid(self.source, self.reference):
            self.dataset = self.source
        else:
            self.dataset = self.stack.enter_context(
                WarpedVRT(
                    self.source,
                    crs=self.reference.crs,
                    transform=self.reference.transform,
                    width=self.reference.width,
                    height=self.reference.height,
                    resampling=Resampling.bilinear,
                )
            )
        self.bands = bands
        if any(v < 1 or v > self.dataset.count for v in bands):
            raise ValueError(f"Invalid bands {bands} for {feature_path} ({self.dataset.count} bands)")

    def close(self) -> None:
        self.stack.close()

    def read(self, window: Window, out_height: int | None = None, out_width: int | None = None) -> np.ndarray:
        kwargs: dict[str, Any] = {"window": window}
        if out_height is not None and out_width is not None:
            kwargs["out_shape"] = (len(self.bands), out_height, out_width)
            kwargs["resampling"] = Resampling.bilinear
        return self.dataset.read(self.bands, **kwargs).astype(np.float32)


def _normalize(array: np.ndarray, feature_cfg: dict[str, Any]) -> np.ndarray:
    value_min = float(feature_cfg.get("value_min", 0.0))
    value_max = float(feature_cfg.get("value_max", 255.0))
    if value_max <= value_min:
        raise ValueError("feature value_max must be greater than value_min")
    array = np.clip((array - value_min) / (value_max - value_min), 0.0, 1.0)
    mean = np.asarray(feature_cfg["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(feature_cfg["std"], dtype=np.float32)[:, None, None]
    if mean.shape[0] != array.shape[0] or np.any(std <= 0):
        raise ValueError("mean/std must match the configured bands and std must be positive")
    return (array - mean) / std


def _brightness_contrast(
    array: np.ndarray,
    feature_cfg: dict[str, Any],
    brightness_limit: float,
    contrast_limit: float,
) -> np.ndarray:
    value_min = float(feature_cfg.get("value_min", 0.0))
    value_max = float(feature_cfg.get("value_max", 255.0))
    unit = np.clip((array - value_min) / (value_max - value_min), 0.0, 1.0)
    contrast = 1.0 + random.uniform(-contrast_limit, contrast_limit)
    brightness = random.uniform(-brightness_limit, brightness_limit)
    center = unit.mean(axis=(1, 2), keepdims=True)
    unit = np.clip((unit - center) * contrast + center + brightness, 0.0, 1.0)
    return unit * (value_max - value_min) + value_min


def create_label_cache(cfg: dict[str, Any], split_name: str) -> Path:
    split = cfg["splits"][split_name]
    reference_path = split_reference_path(cfg, split_name)
    cache_path = (
        output_root(cfg)
        / "cache"
        / f"{split_name}_{config_fingerprint(cfg)[:12]}_{LABEL_CACHE_VERSION}_labels.tif"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not cfg.get("rebuild_label_cache", False):
        return cache_path
    with rasterio.open(reference_path) as reference:
        base = np.full(
            (reference.height, reference.width),
            0 if split.get("labels_are_exhaustive", True) else IGNORE_INDEX,
            dtype=np.uint8,
        )
        valid_region_mask: np.ndarray | None = None
        valid_region = _resolve(cfg["_config_dir"], split.get("valid_region"))
        if valid_region is not None:
            valid = gpd.read_file(valid_region).to_crs(reference.crs)
            base.fill(IGNORE_INDEX)
            valid_region_mask = rasterize(
                ((geom, 1) for geom in valid.geometry if geom is not None),
                out_shape=base.shape,
                transform=reference.transform,
                fill=0,
                dtype=np.uint8,
            ).astype(bool)
            base[valid_region_mask] = 0
        allowed = {0, *[int(item["id"]) for item in cfg["classes"]]}
        grouped: dict[int, list[Any]] = {value: [] for value in sorted(allowed)}
        labels_value = split.get("labels")
        if labels_value:
            labels_path = _resolve(cfg["_config_dir"], labels_value)
            assert labels_path is not None
            labels = gpd.read_file(labels_path).to_crs(reference.crs)
            class_field = split.get("class_field", "class_id")
            if class_field not in labels.columns:
                raise KeyError(f"{labels_path} has no class field '{class_field}'")
            value_map = {str(k): int(v) for k, v in split.get("class_value_map", {}).items()}
            for raw_value, geometry in zip(labels[class_field], labels.geometry):
                class_id = value_map.get(str(raw_value), None)
                if class_id is None:
                    try:
                        class_id = int(raw_value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Unmapped class value {raw_value!r} in {labels_path}; add class_value_map"
                        ) from exc
                if class_id not in allowed:
                    raise ValueError(f"Unexpected class id {class_id} in {labels_path}")
                grouped[class_id].append(geometry)
        for source in split.get("label_sources", []):
            source_path = _resolve(cfg["_config_dir"], source["path"])
            assert source_path is not None
            class_id = int(source["class_id"])
            source_frame = gpd.read_file(source_path).to_crs(reference.crs)
            grouped[class_id].extend(
                geometry for geometry in source_frame.geometry if geometry is not None and not geometry.is_empty
            )
        for class_id, geometries in grouped.items():
            if not geometries:
                continue
            class_mask = rasterize(
                ((geometry, 1) for geometry in geometries if geometry is not None),
                out_shape=base.shape,
                transform=reference.transform,
                fill=0,
                dtype=np.uint8,
                all_touched=bool(split.get("all_touched", True)),
            ).astype(bool)
            base[class_mask] = class_id
        # Label sources may extend beyond the validation/test polygon. Clip after
        # rasterizing every class so inference padding never becomes evaluation.
        if valid_region_mask is not None:
            base[~valid_region_mask] = IGNORE_INDEX
        profile = reference.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=IGNORE_INDEX,
                       compress="LZW", tiled=True, blockxsize=512, blockysize=512)
        with rasterio.open(cache_path, "w", **profile) as dst:
            dst.write(base, 1)
    return cache_path


def confusion_matrix(target: np.ndarray, prediction: np.ndarray, num_classes: int) -> np.ndarray:
    valid = target != IGNORE_INDEX
    values = target[valid].astype(np.int64) * num_classes + prediction[valid].astype(np.int64)
    return np.bincount(values, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(cm: np.ndarray) -> dict[str, Any]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
    # Background is reported, but the selection average is over foreground classes.
    foreground_active = (tp[1:] + fp[1:] + fn[1:]) > 0
    mean_iou = float(np.mean(iou[1:][foreground_active])) if foreground_active.any() else 0.0
    mean_f1 = float(np.mean(f1[1:][foreground_active])) if foreground_active.any() else 0.0
    return {
        "mean_iou": mean_iou,
        "mean_f1": mean_f1,
        "pixel_accuracy": float(tp.sum() / max(1, cm.sum())),
        "per_class_iou": iou.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "confusion_matrix": cm.astype(int).tolist(),
    }


def boundary_iou(target: np.ndarray, prediction: np.ndarray, num_classes: int, width: int) -> float:
    valid = target != IGNORE_INDEX
    core_valid = ndimage.binary_erosion(valid, iterations=width, border_value=0)
    values: list[float] = []
    structure = ndimage.generate_binary_structure(2, 1)
    for class_id in range(1, num_classes):
        truth = (target == class_id) & valid
        pred = (prediction == class_id) & valid
        truth_boundary = truth ^ ndimage.binary_erosion(truth, structure=structure, iterations=width)
        pred_boundary = pred ^ ndimage.binary_erosion(pred, structure=structure, iterations=width)
        truth_band = ndimage.binary_dilation(truth_boundary, structure=structure, iterations=width)
        pred_band = ndimage.binary_dilation(pred_boundary, structure=structure, iterations=width)
        union = (truth_band | pred_band) & core_valid
        if union.any():
            values.append(float(((truth_band & pred_band) & core_valid).sum() / union.sum()))
    return float(np.mean(values)) if values else 0.0


def _edge_and_distance(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = mask != IGNORE_INDEX
    edge = np.zeros(mask.shape, dtype=np.float32)
    for class_id in np.unique(mask[valid]):
        region = (mask == class_id) & valid
        edge += (region ^ ndimage.binary_erosion(region)).astype(np.float32)
    edge = np.clip(edge, 0.0, 1.0)
    distance = ndimage.distance_transform_edt(edge < 0.5).astype(np.float32)
    if distance.max() > 0:
        distance /= distance.max()
    edge[~valid] = 0
    distance[~valid] = 0
    return edge, distance


class PatchDataset(Dataset):
    def __init__(
        self,
        cfg: dict[str, Any],
        split_name: str,
        feature_name: str,
        patch_size: int,
        overlap: float,
        augment: bool,
    ) -> None:
        self.cfg = cfg
        self.split_name = split_name
        self.feature_name = feature_name
        self.patch_size = patch_size
        self.augment = augment
        split = cfg["splits"][split_name]
        self.reference_path = split_reference_path(cfg, split_name)
        self.feature_path = _resolve(cfg["_config_dir"], split["features"][feature_name])
        self.label_path = create_label_cache(cfg, split_name)
        assert self.feature_path is not None
        self.feature_cfg = cfg["features"][feature_name]
        with rasterio.open(self.reference_path) as reference:
            candidates = list(iter_windows(reference.width, reference.height, patch_size, overlap))
        self.windows: list[Window] = []
        min_valid = float(cfg.get("training", {}).get("min_valid_fraction", 0.5))
        min_foreground = float(cfg.get("training", {}).get("min_foreground_fraction", 0.0))
        with rasterio.open(self.label_path) as labels:
            for window in candidates:
                mask = labels.read(1, window=window)
                if np.mean(mask != IGNORE_INDEX) < min_valid:
                    continue
                if augment and np.mean((mask > 0) & (mask != IGNORE_INDEX)) < min_foreground:
                    continue
                self.windows.append(window)
        if not self.windows:
            raise RuntimeError(f"No usable {split_name}/{feature_name} patches were found")
        self._aligned: AlignedRaster | None = None
        self._labels: rasterio.DatasetReader | None = None

    def __len__(self) -> int:
        return len(self.windows)

    def _open(self) -> None:
        if self._aligned is None:
            bands = [int(v) for v in self.feature_cfg.get("bands", [1, 2, 3])]
            self._aligned = AlignedRaster(self.reference_path, self.feature_path, bands)
            self._labels = rasterio.open(self.label_path)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self._open()
        assert self._aligned is not None and self._labels is not None
        window = self.windows[index]
        image = self._aligned.read(window)
        mask = self._labels.read(1, window=window).astype(np.int64)
        if self.augment:
            augmentation = self.cfg.get("training", {}).get("augmentation", {})
            rotation = int(random.choice(augmentation.get("rotation_degrees", [0, 90, 180, 270])))
            if rotation % 90 != 0:
                raise ValueError("Only 90-degree rotation augmentation is supported")
            k = (rotation // 90) % 4
            image, mask = np.rot90(image, k, axes=(1, 2)), np.rot90(mask, k)
            if random.random() < float(augmentation.get("horizontal_flip_probability", 0.5)):
                image, mask = image[:, :, ::-1], mask[:, ::-1]
            if random.random() < float(augmentation.get("vertical_flip_probability", 0.5)):
                image, mask = image[:, ::-1, :], mask[::-1, :]
            if bool(augmentation.get("brightness_contrast", True)):
                image = _brightness_contrast(
                    image,
                    self.feature_cfg,
                    float(augmentation.get("brightness_limit", 0.2)),
                    float(augmentation.get("contrast_limit", 0.2)),
                )
        image = _normalize(image, self.feature_cfg)
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        edge, distance = _edge_and_distance(mask)
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "edge": torch.from_numpy(edge[None]),
            "distance": torch.from_numpy(distance[None]),
        }


def _inverse_frequency_class_weights(dataset: PatchDataset, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)
    with rasterio.open(dataset.label_path) as labels:
        for window in dataset.windows:
            values = labels.read(1, window=window)
            values = values[values != IGNORE_INDEX]
            counts += np.bincount(values, minlength=num_classes)[:num_classes]
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training labels contain no pixels for classes: {missing}")
    weights = counts.sum() / (num_classes * counts.astype(np.float64))
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    safe_target = target.clone()
    safe_target[~valid] = 0
    one_hot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).float()
    valid_mask = valid[:, None].float()
    probs = torch.softmax(logits, dim=1) * valid_mask
    one_hot = one_hot * valid_mask
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    denominator = (probs + one_hot).sum(dim=(0, 2, 3))
    score = (2 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - score[1:].mean()


def _loss(
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    num_classes: int,
    cfg: dict[str, Any],
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = output["mask"]
    assert logits is not None
    ce = F.cross_entropy(logits, batch["mask"], weight=class_weights, ignore_index=IGNORE_INDEX)
    dice = _dice_loss(logits, batch["mask"], num_classes)
    train_cfg = cfg["training"]
    total = float(train_cfg.get("ce_weight", 1.0)) * ce + float(train_cfg.get("dice_weight", 1.0)) * dice
    parts = {"ce": float(ce.detach()), "dice": float(dice.detach())}
    valid = (batch["mask"] != IGNORE_INDEX)[:, None]
    if output.get("edge") is not None:
        edge_outputs = output["edge"] if isinstance(output["edge"], (tuple, list)) else [output["edge"]]
        edge_loss = torch.stack([
            F.binary_cross_entropy_with_logits(edge_output[valid], batch["edge"][valid])
            for edge_output in edge_outputs
        ]).mean()
        total = total + float(train_cfg.get("seanet_edge_weight", 1.0)) * edge_loss
        parts["edge"] = float(edge_loss.detach())
    if output.get("distance") is not None:
        pred_distance = torch.sigmoid(output["distance"])
        distance_loss = F.smooth_l1_loss(pred_distance[valid], batch["distance"][valid])
        total = total + float(train_cfg.get("seanet_distance_weight", 1.0)) * distance_loss
        parts["distance"] = float(distance_loss.detach())
    return total, parts


def _device(cfg: dict[str, Any]) -> torch.device:
    requested = cfg.get("device", "cuda")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def _model_bundle(cfg: dict[str, Any], model_name: str, pretrained: bool):
    seanet_cfg = cfg.get("seanet", {})
    deeplab_cfg = cfg.get("deeplabv3plus", {})
    source_dir = _resolve(cfg["_config_dir"], seanet_cfg.get("source_dir"))
    vgg_path = _resolve(cfg["_config_dir"], seanet_cfg.get("vgg16_mat"))
    return build_model(
        model_name,
        1 + len(cfg["classes"]),
        pretrained=pretrained,
        encoder_name=cfg.get("encoder_by_model", {}).get(model_name, "resnet34"),
        deeplab_output_stride=int(deeplab_cfg.get("output_stride", 16)),
        deeplab_atrous_rates=deeplab_cfg.get("atrous_rates"),
        seanet_source_dir=str(source_dir) if source_dir else None,
        seanet_pretrained_source=seanet_cfg.get("pretrained_source", "torchvision_vgg16_imagenet"),
        seanet_vgg16_mat=str(vgg_path) if vgg_path else None,
    )


def _loader(dataset: Dataset, cfg: dict[str, Any], training: bool, model_name: str) -> DataLoader:
    train_cfg = cfg["training"]
    batch_size = int(train_cfg.get("batch_size_by_model", {}).get(model_name, train_cfg.get("batch_size", 4)))
    workers = int(train_cfg.get("workers", 0))
    kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training and len(dataset) >= batch_size,
    )
    if workers > 0:
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
    return DataLoader(**kwargs)


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        if _channels_last(cfg, device):
            image = image.contiguous(memory_format=torch.channels_last)
        with _autocast(cfg, device):
            prediction = model(image)["mask"].argmax(1).cpu().numpy()
        target = batch["mask"].numpy()
        for truth, pred in zip(target, prediction):
            cm += confusion_matrix(truth, pred, num_classes)
    return metrics_from_confusion(cm)


def train_one(cfg: dict[str, Any], model_name: str, feature_name: str, force: bool = False) -> Path:
    destination = output_root(cfg) / "training" / model_name / feature_name
    checkpoint_path = destination / "best.pt"
    manifest_path = destination / "manifest.json"
    fingerprint = config_fingerprint(cfg)
    if not force and checkpoint_path.exists() and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            if json.load(handle).get("config_fingerprint") == fingerprint:
                return checkpoint_path
    destination.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg.get("seed", 3407)), cfg)
    patch_size = int(cfg["training"].get("patch_size", 512))
    train_dataset = PatchDataset(cfg, "train", feature_name, patch_size, 0.0, True)
    val_dataset = PatchDataset(cfg, "validation", feature_name, patch_size, 0.0, False)
    train_loader = _loader(train_dataset, cfg, True, model_name)
    val_loader = _loader(val_dataset, cfg, False, model_name)
    device = _device(cfg)
    if device.type == "cuda" and bool(cfg.get("runtime", {}).get("empty_cache_between_runs", False)):
        torch.cuda.empty_cache()
    use_pretrained = cfg.get("initialization_policy", "random_all") == "official_pretrained"
    bundle = _model_bundle(cfg, model_name, pretrained=use_pretrained)
    model = bundle.model.to(device)
    freeze_bn = use_pretrained and bool(cfg["training"].get("freeze_pretrained_encoder_batchnorm", False))
    frozen_bn_layers = freeze_encoder_batchnorm(model) if freeze_bn else 0
    if _channels_last(cfg, device):
        model = model.to(memory_format=torch.channels_last)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    learning_rates = cfg["training"].get("learning_rate_by_model", {})
    learning_rate = float(learning_rates.get(model_name, cfg["training"].get("learning_rate", 1e-4)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    epochs = int(cfg["training"].get("epochs", 50))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    _, amp_dtype = _precision(cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)
    num_classes = 1 + len(cfg["classes"])
    weighting = cfg["training"].get("class_weighting", "inverse_frequency")
    if weighting != "inverse_frequency":
        raise ValueError("training.class_weighting must be inverse_frequency")
    class_weights = _inverse_frequency_class_weights(train_dataset, num_classes).to(device)
    best_iou = -math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_bn:
            freeze_encoder_batchnorm(model)
        running = 0.0
        for batch in tqdm(train_loader, desc=f"train {model_name}/{feature_name} {epoch}/{epochs}"):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            if _channels_last(cfg, device):
                batch["image"] = batch["image"].contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg, device):
                loss, _ = _loss(model(batch["image"]), batch, num_classes, cfg, class_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
        scheduler.step()
        metrics = evaluate_loader(model, val_loader, device, num_classes, cfg)
        record = {"epoch": epoch, "train_loss": running / max(1, len(train_loader)), **metrics}
        history.append(record)
        save_json(destination / "history.json", history)
        print(
            f"epoch_summary model={model_name} feature={feature_name} epoch={epoch}/{epochs} "
            f"train_loss={record['train_loss']:.6f} val_mean_iou={metrics['mean_iou']:.6f} "
            f"val_mean_f1={metrics['mean_f1']:.6f} val_pixel_accuracy={metrics['pixel_accuracy']:.6f}",
            flush=True,
        )
        if metrics["mean_iou"] > best_iou:
            best_iou = metrics["mean_iou"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model": model_name,
                    "feature": feature_name,
                    "num_classes": num_classes,
                    "config_fingerprint": fingerprint,
                    "initialization": bundle.initialization,
                    "best_epoch": epoch,
                    "best_validation_mean_iou": best_iou,
                    "class_weights": class_weights.detach().cpu().tolist(),
                    "learning_rate": learning_rate,
                },
                checkpoint_path,
            )
    save_json(
        manifest_path,
        {
            "config_fingerprint": fingerprint,
            "model": model_name,
            "feature": feature_name,
            "initialization": bundle.initialization,
            "initialization_policy": cfg.get("initialization_policy", "random_all"),
            "frozen_encoder_batchnorm_layers": frozen_bn_layers,
            "checkpoint_selection": "validation foreground mean_iou",
            "best_validation_mean_iou": best_iou,
            "class_weights": class_weights.detach().cpu().tolist(),
            "class_weighting": weighting,
            "learning_rate": learning_rate,
            "train_patches": len(train_dataset),
            "validation_patches": len(val_dataset),
            "peak_cuda_memory_gb": (
                float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0
            ),
        },
    )
    return checkpoint_path


def _blend_weight(height: int, width: int) -> np.ndarray:
    row = np.hanning(height) if height > 2 else np.ones(height, dtype=np.float32)
    col = np.hanning(width) if width > 2 else np.ones(width, dtype=np.float32)
    weight = np.outer(row, col).astype(np.float32)
    return np.maximum(weight, 1e-3)


def _load_checkpoint_model(
    cfg: dict[str, Any], model_name: str, feature_name: str, device: torch.device
) -> torch.nn.Module:
    path = output_root(cfg) / "training" / model_name / feature_name / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {path}. Run the train stage first.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != model_name or checkpoint.get("feature") != feature_name:
        raise RuntimeError(f"Checkpoint identity does not match requested run: {path}")
    if checkpoint.get("config_fingerprint") != config_fingerprint(cfg):
        raise RuntimeError(f"Checkpoint was created with a different config: {path}")
    model = _model_bundle(cfg, model_name, pretrained=False).model
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    if _channels_last(cfg, device):
        model = model.to(memory_format=torch.channels_last)
    return model


def _metrics_from_prediction(
    prediction_path: Path,
    label_path: Path,
    num_classes: int,
    boundary_width: int,
    label_window: Window | None = None,
    stratify_region: Path | None = None,
) -> dict[str, Any]:
    with rasterio.open(prediction_path) as pred_ds, rasterio.open(label_path) as label_ds:
        pred = pred_ds.read(1)
        target = label_ds.read(1, window=label_window) if label_window is not None else label_ds.read(1)
        pred_transform = pred_ds.transform
        pred_crs = pred_ds.crs
    if pred.shape != target.shape:
        raise RuntimeError(f"Prediction/label shape mismatch: {pred.shape} vs {target.shape}")
    metrics = metrics_from_confusion(confusion_matrix(target, pred, num_classes))
    metrics["boundary_iou"] = boundary_iou(target, pred, num_classes, boundary_width)
    if stratify_region is not None:
        zone_frame = gpd.read_file(stratify_region).to_crs(pred_crs)
        shed = rasterize(
            ((geometry, 1) for geometry in zone_frame.geometry if geometry is not None),
            out_shape=target.shape,
            transform=pred_transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        ).astype(bool)
        zones: dict[str, Any] = {}
        for zone_name, zone_mask in (("shed_zone", shed), ("ordinary_zone", ~shed)):
            zone_target = target.copy()
            zone_target[~zone_mask] = IGNORE_INDEX
            zone_metrics = metrics_from_confusion(confusion_matrix(zone_target, pred, num_classes))
            zone_metrics["boundary_iou"] = boundary_iou(
                zone_target, pred, num_classes, boundary_width
            )
            zones[zone_name] = zone_metrics
            metrics[f"{zone_name}_mean_iou"] = zone_metrics["mean_iou"]
            metrics[f"{zone_name}_mean_f1"] = zone_metrics["mean_f1"]
            metrics[f"{zone_name}_boundary_iou"] = zone_metrics["boundary_iou"]
        metrics["zones"] = zones
    return metrics


def _roi_window(
    cfg: dict[str, Any], split_name: str, reference: rasterio.DatasetReader, scale: int
) -> Window:
    valid_region = _resolve(cfg["_config_dir"], cfg["splits"][split_name].get("valid_region"))
    if valid_region is None:
        return Window(0, 0, reference.width, reference.height)
    frame = gpd.read_file(valid_region).to_crs(reference.crs)
    left, bottom, right, top = frame.total_bounds
    raw = from_bounds(left, bottom, right, top, reference.transform)
    padding = int(scale) // 2
    col0 = max(0, int(math.floor(raw.col_off)) - padding)
    row0 = max(0, int(math.floor(raw.row_off)) - padding)
    col1 = min(reference.width, int(math.ceil(raw.col_off + raw.width)) + padding)
    row1 = min(reference.height, int(math.ceil(raw.row_off + raw.height)) + padding)
    return Window(col0, row0, col1 - col0, row1 - row0)


@torch.no_grad()
def infer_one(
    cfg: dict[str, Any],
    split_name: str,
    model_name: str,
    feature_name: str,
    scale: int,
    force: bool = False,
) -> dict[str, Any]:
    stage_name = "validation" if split_name == "validation" else "test"
    destination = output_root(cfg) / stage_name / model_name / feature_name / f"scale_{scale}"
    metrics_path = destination / "metrics.json"
    prediction_path = destination / "prediction.tif"
    if not force and metrics_path.exists() and prediction_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("config_fingerprint") == config_fingerprint(cfg):
            return cached
    destination.mkdir(parents=True, exist_ok=True)
    split = cfg["splits"][split_name]
    reference_path = split_reference_path(cfg, split_name)
    feature_path = _resolve(cfg["_config_dir"], split["features"][feature_name])
    assert feature_path is not None
    label_path = create_label_cache(cfg, split_name)
    feature_cfg = cfg["features"][feature_name]
    bands = [int(v) for v in feature_cfg.get("bands", [1, 2, 3])]
    overlap = float(cfg.get("inference", {}).get("overlap", 0.5))
    # ``scale`` is the source-image context/window size, not the tensor size.
    # All scale candidates must enter the network at the same spatial size used
    # during training; otherwise larger contexts change GPU memory/architecture
    # behaviour and 2048/4096 windows are impractical even on a large GPU.
    network_input_size = int(
        cfg.get("inference", {}).get(
            "network_input_size", cfg.get("training", {}).get("patch_size", 512)
        )
    )
    if network_input_size <= 0:
        raise ValueError("inference.network_input_size must be positive")
    device = _device(cfg)
    seed_everything(int(cfg.get("seed", 3407)), cfg)
    if device.type == "cuda" and bool(cfg.get("runtime", {}).get("empty_cache_between_runs", False)):
        torch.cuda.empty_cache()
    model = _load_checkpoint_model(cfg, model_name, feature_name, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    num_classes = 1 + len(cfg["classes"])
    with rasterio.open(reference_path) as reference:
        roi_window = _roi_window(cfg, split_name, reference, int(scale))
        height, width = int(roi_window.height), int(roi_window.width)
        profile = reference.profile.copy()
        profile.update(
            width=width,
            height=height,
            transform=window_transform(roi_window, reference.transform),
        )
    prob_path = destination / "probability_sum.dat"
    weight_path = destination / "weight_sum.dat"
    prob_sum = np.memmap(prob_path, mode="w+", dtype="float32", shape=(num_classes, height, width))
    weight_sum = np.memmap(weight_path, mode="w+", dtype="float32", shape=(height, width))
    prob_sum[:] = 0
    weight_sum[:] = 0
    reader = AlignedRaster(reference_path, feature_path, bands)
    try:
        windows = list(iter_windows(width, height, int(scale), overlap))
        for local_window in tqdm(windows, desc=f"{stage_name} {model_name}/{feature_name}/{scale}"):
            source_window = Window(
                int(roi_window.col_off + local_window.col_off),
                int(roi_window.row_off + local_window.row_off),
                int(local_window.width),
                int(local_window.height),
            )
            image = _normalize(
                reader.read(
                    source_window,
                    out_height=network_input_size,
                    out_width=network_input_size,
                ),
                feature_cfg,
            )
            tensor = torch.from_numpy(np.ascontiguousarray(image))[None].to(device)
            if _channels_last(cfg, device):
                tensor = tensor.contiguous(memory_format=torch.channels_last)
            with _autocast(cfg, device):
                logits = model(tensor)["mask"]
            probability = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
            h, w = probability.shape[-2:]
            if (h, w) != (int(local_window.height), int(local_window.width)):
                probability = F.interpolate(
                    torch.from_numpy(probability)[None],
                    size=(int(local_window.height), int(local_window.width)),
                    mode="bilinear",
                    align_corners=False,
                )[0].numpy()
                h, w = probability.shape[-2:]
            weight = _blend_weight(h, w)
            row, col = int(local_window.row_off), int(local_window.col_off)
            prob_sum[:, row:row + h, col:col + w] += probability * weight[None]
            weight_sum[row:row + h, col:col + w] += weight
        prob_sum.flush()
        weight_sum.flush()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=IGNORE_INDEX,
                       compress="LZW", tiled=True, blockxsize=512, blockysize=512)
        with rasterio.open(prediction_path, "w", **profile) as dst:
            block_rows = 512
            for row in range(0, height, block_rows):
                h = min(block_rows, height - row)
                denom = np.maximum(weight_sum[row:row + h], 1e-8)
                pred = np.argmax(prob_sum[:, row:row + h] / denom[None], axis=0).astype(np.uint8)
                dst.write(pred, 1, window=Window(0, row, width, h))
    finally:
        reader.close()
        del prob_sum
        del weight_sum
        for temp_path in (prob_path, weight_path):
            if temp_path.exists():
                temp_path.unlink()
    peak_cuda_memory_gb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0
    )
    metrics = _metrics_from_prediction(
        prediction_path,
        label_path,
        num_classes,
        int(cfg.get("inference", {}).get("boundary_width_pixels", 3)),
        label_window=roi_window,
        stratify_region=_resolve(cfg["_config_dir"], split.get("stratify_region")),
    )
    metrics.update(
        {
            "split": split_name,
            "model": model_name,
            "feature": feature_name,
            "scale": int(scale),
            "network_input_size": network_input_size,
            "checkpoint": str(output_root(cfg) / "training" / model_name / feature_name / "best.pt"),
            "selection_eligible": split_name == "validation",
            "config_fingerprint": config_fingerprint(cfg),
            "peak_cuda_memory_gb": peak_cuda_memory_gb,
        }
    )
    save_json(metrics_path, metrics)
    del model
    if device.type == "cuda" and bool(cfg.get("runtime", {}).get("empty_cache_between_runs", False)):
        torch.cuda.empty_cache()
    return metrics


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "split", "model", "feature", "scale", "mean_iou", "mean_f1",
        "boundary_iou", "pixel_accuracy", "ordinary_zone_mean_iou",
        "shed_zone_mean_iou", "peak_cuda_memory_gb",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_all(
    cfg: dict[str, Any],
    models: Iterable[str] = MODEL_NAMES,
    features: Iterable[str] = FEATURE_NAMES,
    scales: Iterable[int] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    models = list(models)
    features = list(features)
    scales = list(scales or cfg["scales"])
    formal_features = list(cfg.get("experiment_features", FEATURE_NAMES))
    formal = (
        len(models) == len(MODEL_NAMES)
        and set(models) == set(MODEL_NAMES)
        and len(features) == len(formal_features)
        and set(features) == set(formal_features)
        and len(scales) == 4
        and set(int(value) for value in scales) == {4096, 2048, 1024, 512}
    )
    rows: list[dict[str, Any]] = []
    for model_name in models:
        for feature_name in features:
            for scale in scales:
                rows.append(infer_one(cfg, "validation", model_name, feature_name, int(scale), force))
    if not formal:
        _write_csv(output_root(cfg) / "validation_results.partial.csv", rows)
        return {
            "partial": True,
            "message": "Partial smoke-test results were saved; no deployable selection was created.",
            "models": {},
        }
    _write_csv(output_root(cfg) / "validation_results.csv", rows)
    selected: dict[str, Any] = {}
    for model_name in models:
        candidates = [row for row in rows if row["model"] == model_name]
        expected = len(features) * len(scales)
        if len(candidates) != expected:
            raise RuntimeError(f"{model_name} has {len(candidates)} validation runs; expected {expected}")
        best = max(
            candidates,
            key=lambda row: (float(row["mean_iou"]), float(row["mean_f1"]), float(row["boundary_iou"])),
        )
        selected[model_name] = {
            "feature": best["feature"],
            "scale": int(best["scale"]),
            "validation_mean_iou": float(best["mean_iou"]),
            "validation_mean_f1": float(best["mean_f1"]),
            "validation_boundary_iou": float(best["boundary_iou"]),
            "selection_rule": "max mean_iou; ties: mean_f1 then boundary_iou",
        }
    payload = {
        "selected_on": "validation",
        "primary_metric": "mean_iou",
        "test_metrics_used_for_selection": False,
        "config_fingerprint": config_fingerprint(cfg),
        "models": selected,
    }
    save_json(output_root(cfg) / "selected_configs.json", payload)
    return payload


def test_selected(cfg: dict[str, Any], models: Iterable[str] = MODEL_NAMES, force: bool = False) -> list[dict[str, Any]]:
    models = list(models)
    if len(models) != len(MODEL_NAMES) or set(models) != set(MODEL_NAMES):
        raise ValueError("The locked test stage must evaluate all three models exactly once; model subsets are forbidden.")
    selection_path = output_root(cfg) / "selected_configs.json"
    if not selection_path.is_file():
        raise FileNotFoundError("selected_configs.json is missing; complete validation selection first")
    with selection_path.open("r", encoding="utf-8") as handle:
        selected = json.load(handle)
    if selected.get("selected_on") != "validation" or selected.get("test_metrics_used_for_selection") is not False:
        raise RuntimeError("Refusing test: selection file does not prove validation-only selection")
    if selected.get("config_fingerprint") != config_fingerprint(cfg):
        raise RuntimeError("Selection file was produced by a different configuration")
    rows: list[dict[str, Any]] = []
    for model_name in models:
        choice = selected["models"][model_name]
        rows.append(
            infer_one(cfg, "test", model_name, choice["feature"], int(choice["scale"]), force)
        )
    _write_csv(output_root(cfg) / "test_results.csv", rows)
    return rows


def copy_config_snapshot(cfg: dict[str, Any]) -> None:
    source = Path(cfg["_config_path"])
    destination = output_root(cfg) / "config.snapshot.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
