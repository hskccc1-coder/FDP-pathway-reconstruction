from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    # Configurations are grouped under project_root/configs in this package.
    # Keep compatibility with the original layout, where the JSON lived at root.
    base = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    return config, base


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def ensure_parent(path: str | Path) -> Path:
    result = Path(path)
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def save_json(path: str | Path, value: Any) -> None:
    output = ensure_parent(path)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_dataset(path: str | Path) -> str:
    value = Path(path)
    if value.suffix.lower() != ".shp":
        return sha256_file(value)
    members = sorted(
        candidate
        for candidate in value.parent.glob(f"{value.stem}.*")
        if candidate.is_file()
    )
    if not members:
        raise FileNotFoundError(value)
    digest = hashlib.sha256()
    for member in members:
        digest.update(member.name.lower().encode("utf-8"))
        digest.update(b"\0")
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest().upper()


def nsr_class_priority(config: dict[str, Any], requested: list[str] | None) -> list[str]:
    """Return the author-recorded Field/Water priority; a differing CLI value is rejected."""
    protocol = config.get("nsr_evaluation", {})
    priority = protocol.get("class_priority")
    if sorted(priority or []) != ["Field", "Water"]:
        raise KeyError("config nsr_evaluation.class_priority must list Field and Water exactly once")
    if protocol.get("water_inside_interstrip") != "excluded_from_evaluation":
        raise ValueError("Only nsr_evaluation.water_inside_interstrip=excluded_from_evaluation is implemented")
    if requested is not None and list(requested) != list(priority):
        raise ValueError(f"--class-priority {requested} contradicts the recorded protocol {priority}")
    return list(priority)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
