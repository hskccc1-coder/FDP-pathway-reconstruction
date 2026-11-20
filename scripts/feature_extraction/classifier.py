from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image
from rasterio.mask import mask
from shapely.geometry import box
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import load_config, resolve, save_json, seed_everything


def expanded_bbox(geometry, factor: float):
    minx, miny, maxx, maxy = geometry.bounds
    width = max(maxx - minx, 1e-9) * factor
    height = max(maxy - miny, 1e-9) * factor
    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0
    return box(
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


def geometry_hash(geometry) -> str:
    return hashlib.sha256(geometry.wkb).hexdigest()


def object_id(index: int, geometry) -> str:
    return f"obj_{index:04d}_{geometry_hash(geometry)[:12]}"


def _positive_intersection_area(geometry, region) -> float:
    if geometry is None or geometry.is_empty or not geometry.intersects(region):
        return 0.0
    return float(geometry.intersection(region).area)


def prepare_manifest(config: dict, base: Path, output: Path) -> pd.DataFrame:
    settings = config["classifier"]
    labels_path = resolve(base, config["paths"]["classification_objects"])
    test_region_path = resolve(base, config["paths"]["independent_test_region"])
    frame = gpd.read_file(labels_path).reset_index(drop=True)
    test_region = gpd.read_file(test_region_path)
    if frame.crs is None or test_region.crs is None:
        raise ValueError("Classification objects and the independent test region require a CRS")
    if test_region.crs != frame.crs:
        test_region = test_region.to_crs(frame.crs)

    label_field = settings["label_field"]
    if label_field not in frame.columns:
        raise KeyError(f"Missing label field: {label_field}")
    classes = list(settings["classes"])
    unknown = sorted(set(frame[label_field].astype(str)).difference(classes))
    if unknown:
        raise ValueError(f"Unrecognized classes: {unknown}")
    if frame.geometry.isna().any() or frame.geometry.is_empty.any():
        raise ValueError("The object library contains null or empty geometries")
    if (~frame.geometry.is_valid).any():
        raise ValueError("The object library contains invalid geometries")

    frame["geometry_hash"] = frame.geometry.map(geometry_hash)
    duplicate = frame["geometry_hash"].duplicated(keep=False)
    if duplicate.any():
        raise ValueError(f"Duplicate geometries detected: {int(duplicate.sum())}")
    observed_source = frame[label_field].value_counts().reindex(classes, fill_value=0).to_dict()
    expected_source = settings.get("expected_source_counts")
    if expected_source and observed_source != expected_source:
        raise ValueError(f"Source class counts differ: observed={observed_source}, expected={expected_source}")

    test_union = test_region.geometry.unary_union
    frame["test_overlap_m2"] = frame.geometry.map(lambda value: _positive_intersection_area(value, test_union))
    excluded = frame[frame["test_overlap_m2"] > 0].copy()
    retained = frame[frame["test_overlap_m2"] <= 0].copy()
    excluded_counts = excluded[label_field].value_counts().reindex(classes, fill_value=0).to_dict()
    expected_excluded = settings.get("expected_excluded_test_overlap_counts")
    if expected_excluded and excluded_counts != expected_excluded:
        raise ValueError(
            f"Test-overlap exclusions differ: observed={excluded_counts}, expected={expected_excluded}"
        )

    retained["source_row"] = retained.index.astype(int)
    retained["object_id"] = [object_id(int(index), geometry) for index, geometry in retained.geometry.items()]
    train_rows, validation_rows = train_test_split(
        retained.index.to_numpy(),
        test_size=float(settings["validation_fraction"]),
        random_state=int(config["seed"]),
        stratify=retained[label_field],
        shuffle=True,
    )
    retained["split"] = "validation"
    retained.loc[train_rows, "split"] = "train"
    manifest = retained[
        ["source_row", "object_id", "geometry_hash", label_field, "split", "test_overlap_m2"]
    ].rename(columns={label_field: "label"})
    if manifest.groupby("object_id")["split"].nunique().max() != 1:
        raise RuntimeError("An object was assigned to more than one subset")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False, encoding="utf-8-sig")
    excluded_output = output.with_name("excluded_test_overlap_objects.csv")
    excluded.assign(source_row=excluded.index.astype(int))[
        ["source_row", label_field, "geometry_hash", "test_overlap_m2"]
    ].rename(columns={label_field: "label"}).to_csv(excluded_output, index=False, encoding="utf-8-sig")

    summary = {
        "source_objects": int(len(frame)),
        "source_class_counts": observed_source,
        "excluded_test_overlap_objects": int(len(excluded)),
        "excluded_test_overlap_class_counts": excluded_counts,
        "excluded_test_overlap_area_m2": float(excluded["test_overlap_m2"].sum()),
        "retained_objects": int(len(retained)),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "split_class_counts": {
            split: subset["label"].value_counts().reindex(classes, fill_value=0).to_dict()
            for split, subset in manifest.groupby("split")
        },
        "seed": int(config["seed"]),
        "validation_fraction": float(settings["validation_fraction"]),
    }
    save_json(output.with_name("split_summary.json"), summary)
    return manifest


def _crop_geometries(geometry, mode: str, expansion: float) -> list:
    tight = box(*geometry.bounds)
    expanded = expanded_bbox(geometry, expansion)
    if mode == "tight":
        return [tight]
    if mode == "expanded":
        return [expanded]
    if mode == "dual_average":
        return [tight, expanded]
    raise ValueError(f"Unsupported crop mode: {mode}")


def _rgb_uint8(array: np.ndarray, value_min: float, value_max: float) -> np.ndarray:
    image = np.moveaxis(array[:3], 0, -1).astype(np.float32)
    image = np.nan_to_num(image, nan=value_min, posinf=value_max, neginf=value_min)
    image = np.clip((image - value_min) / (value_max - value_min), 0.0, 1.0)
    return np.round(image * 255.0).astype(np.uint8)


class ObjectDataset(Dataset):
    def __init__(
        self,
        objects: gpd.GeoDataFrame,
        raster_path: Path,
        classes: list[str],
        transform,
        crop_mode: str,
        expansion: float,
        value_min: float,
        value_max: float,
    ) -> None:
        self.objects = objects.reset_index(drop=True)
        self.raster_path = raster_path
        self.classes = classes
        self.class_to_index = {value: index for index, value in enumerate(classes)}
        self.transform = transform
        self.crop_mode = crop_mode
        self.expansion = expansion
        self.value_min = value_min
        self.value_max = value_max
        self.source = None

    def __len__(self) -> int:
        return len(self.objects)

    def _open(self) -> None:
        if self.source is None:
            self.source = rasterio.open(self.raster_path)

    def __getitem__(self, index: int):
        self._open()
        row = self.objects.iloc[index]
        views = []
        for crop_geometry in _crop_geometries(row.geometry, self.crop_mode, self.expansion):
            array, _ = mask(self.source, [crop_geometry], crop=True, filled=True, all_touched=True)
            image = Image.fromarray(_rgb_uint8(array, self.value_min, self.value_max), mode="RGB")
            views.append(self.transform(image))
        label = self.class_to_index[str(row["label"])] if "label" in row else -1
        return torch.stack(views), label, int(index)

    def close(self) -> None:
        if self.source is not None:
            self.source.close()
            self.source = None

    def __del__(self):
        self.close()


def _transform(settings: dict):
    return transforms.Compose(
        [
            transforms.Resize((int(settings["input_size_px"]), int(settings["input_size_px"]))),
            transforms.ToTensor(),
            transforms.Normalize(settings["normalization_mean"], settings["normalization_std"]),
        ]
    )


def _load_objects(config: dict, base: Path, manifest_path: Path, split: str) -> gpd.GeoDataFrame:
    settings = config["classifier"]
    source = gpd.read_file(resolve(base, config["paths"]["classification_objects"]))
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["split"] == split].copy()
    selected = source.iloc[manifest["source_row"].astype(int).to_numpy()].copy().reset_index(drop=True)
    selected["object_id"] = manifest["object_id"].to_numpy()
    selected["label"] = manifest["label"].to_numpy()
    raster_path = resolve(base, config["paths"]["rgb_raster"])
    with rasterio.open(raster_path) as raster:
        if selected.crs is None:
            raise ValueError("Classification objects require a CRS")
        if selected.crs != raster.crs:
            selected = selected.to_crs(raster.crs)
    observed = selected["label"].value_counts().reindex(settings["classes"], fill_value=0)
    if (observed == 0).any():
        raise ValueError(f"Split {split} is missing classes: {observed[observed == 0].index.tolist()}")
    return selected


def make_model(number_of_classes: int, dropout: float, pretrained: bool) -> nn.Module:
    weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet101(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, number_of_classes))
    return model


def _object_probabilities(model: nn.Module, views: torch.Tensor) -> torch.Tensor:
    batch, number_of_views, channels, height, width = views.shape
    logits = model(views.reshape(batch * number_of_views, channels, height, width))
    return torch.softmax(logits.reshape(batch, number_of_views, -1), dim=2).mean(dim=1)


def _evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    loss_total = 0.0
    correct = 0
    count = 0
    with torch.inference_mode():
        for views, labels, _ in loader:
            views = views.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            probabilities = _object_probabilities(model, views)
            loss = nn.functional.nll_loss(torch.log(probabilities.clamp_min(1e-8)), labels)
            loss_total += float(loss) * len(labels)
            correct += int((probabilities.argmax(1) == labels).sum())
            count += len(labels)
    return loss_total / max(count, 1), correct / max(count, 1)


def train(config: dict, base: Path, manifest_path: Path, output_dir: Path) -> None:
    settings = config["classifier"]
    seed_everything(int(config["seed"]))
    classes = list(settings["classes"])
    raster_path = resolve(base, config["paths"]["rgb_raster"])
    train_objects = _load_objects(config, base, manifest_path, "train")
    validation_objects = _load_objects(config, base, manifest_path, "validation")
    transform = _transform(settings)
    dataset_arguments = dict(
        raster_path=raster_path,
        classes=classes,
        transform=transform,
        crop_mode=settings["crop_mode"],
        expansion=float(settings["crop_expansion"]),
        value_min=float(settings["value_min"]),
        value_max=float(settings["value_max"]),
    )
    train_dataset = ObjectDataset(train_objects, **dataset_arguments)
    validation_dataset = ObjectDataset(validation_objects, **dataset_arguments)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        num_workers=int(settings["workers"]),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        num_workers=int(settings["workers"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(len(classes), float(settings["dropout"]), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    history = []
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        item_count = 0
        for views, labels, _ in train_loader:
            views = views.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch, number_of_views, channels, height, width = views.shape
            repeated_labels = labels.repeat_interleave(number_of_views)
            optimizer.zero_grad(set_to_none=True)
            logits = model(views.reshape(batch * number_of_views, channels, height, width))
            loss = criterion(logits, repeated_labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss) * len(repeated_labels)
            item_count += len(repeated_labels)
        validation_loss, validation_accuracy = _evaluate(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(item_count, 1),
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(record)
        print(
            f"epoch={epoch}/{settings['epochs']} train_loss={record['train_loss']:.6f} "
            f"validation_loss={validation_loss:.6f} validation_accuracy={validation_accuracy:.6f}",
            flush=True,
        )
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "classes": classes,
            "crop_mode": settings["crop_mode"],
            "crop_expansion": float(settings["crop_expansion"]),
            "validation_accuracy": validation_accuracy,
            "seed": int(config["seed"]),
        }
        torch.save(state, output_dir / "last_resnet101.pt")
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(state, output_dir / "best_resnet101.pt")
    save_json(output_dir / "history.json", history)
    save_json(
        output_dir / "training_manifest.json",
        {
            "architecture": "ResNet-101",
            "pretrained_weights": "ImageNet1K_V2",
            "classes": classes,
            "train_objects": len(train_dataset),
            "validation_objects": len(validation_dataset),
            "crop_mode": settings["crop_mode"],
            "number_of_views_per_object": 2 if settings["crop_mode"] == "dual_average" else 1,
            "epochs": int(settings["epochs"]),
            "batch_size_objects": int(settings["batch_size"]),
            "optimizer": "Adam",
            "learning_rate": float(settings["learning_rate"]),
            "loss": "cross_entropy",
            "checkpoint_selection": "highest object-level validation accuracy",
            "best_validation_accuracy": best_accuracy,
        },
    )


def _load_checkpoint(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def predict(config: dict, base: Path, checkpoint: Path, objects_path: Path, output: Path) -> None:
    settings = config["classifier"]
    classes = list(settings["classes"])
    frame = gpd.read_file(objects_path).reset_index(drop=True)
    raster_path = resolve(base, config["paths"]["rgb_raster"])
    with rasterio.open(raster_path) as raster:
        if frame.crs is None:
            raise ValueError("Prediction objects require a CRS")
        raster_crs = raster.crs
    original_crs = frame.crs
    working = frame if frame.crs == raster_crs else frame.to_crs(raster_crs)
    working = working.copy()
    working["label"] = classes[0]
    dataset = ObjectDataset(
        working,
        raster_path,
        classes,
        _transform(settings),
        settings["crop_mode"],
        float(settings["crop_expansion"]),
        float(settings["value_min"]),
        float(settings["value_max"]),
    )
    loader = DataLoader(dataset, batch_size=int(settings["batch_size"]), shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = _load_checkpoint(checkpoint, device)
    saved_classes = list(saved.get("classes", classes))
    if saved_classes != classes:
        raise ValueError(f"Checkpoint class order differs: {saved_classes} != {classes}")
    if saved.get("crop_mode", settings["crop_mode"]) != settings["crop_mode"]:
        raise ValueError("Checkpoint and prediction crop modes differ")
    model = make_model(len(classes), float(settings["dropout"]), pretrained=False).to(device)
    model.load_state_dict(saved["model_state"] if "model_state" in saved else saved)
    model.eval()
    all_probabilities = []
    with torch.inference_mode():
        for views, _, _ in loader:
            all_probabilities.append(_object_probabilities(model, views.to(device)).cpu().numpy())
    probabilities = np.concatenate(all_probabilities, axis=0)
    predictions = probabilities.argmax(axis=1)
    result = frame.copy()
    result["class_pred"] = [classes[index] for index in predictions]
    result["class_prob"] = probabilities.max(axis=1)
    for index, class_name in enumerate(classes):
        result[f"p_{class_name[:8].lower()}"] = probabilities[:, index]
    result = result.set_crs(original_crs, allow_override=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(output, driver="GPKG")


def main() -> None:
    parser = argparse.ArgumentParser(description="Object-level ResNet-101 training and prediction")
    parser.add_argument("--config", default="config.json")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--manifest", default="results/segmentation_metrics/runtime/classifier/split_manifest.csv")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--manifest", default="results/segmentation_metrics/runtime/classifier/split_manifest.csv")
    train_parser.add_argument("--output-dir", default="results/segmentation_metrics/runtime/classifier")
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--checkpoint", default="results/segmentation_metrics/runtime/classifier/best_resnet101.pt")
    predict_parser.add_argument("--objects", required=True)
    predict_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, base = load_config(args.config)
    if args.command == "prepare":
        prepare_manifest(config, base, resolve(base, args.manifest))
    elif args.command == "train":
        train(config, base, resolve(base, args.manifest), resolve(base, args.output_dir))
    else:
        predict(
            config,
            base,
            resolve(base, args.checkpoint),
            Path(args.objects).resolve(),
            Path(args.output).resolve(),
        )


if __name__ == "__main__":
    main()
