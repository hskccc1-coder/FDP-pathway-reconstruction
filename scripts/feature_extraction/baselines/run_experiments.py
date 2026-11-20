from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from pipeline import (
    FEATURE_NAMES,
    MODEL_NAMES,
    audit_spatial_splits,
    copy_config_snapshot,
    load_config,
    output_root,
    test_selected,
    train_one,
    validate_all,
    validate_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Four-scale comparison for U-Net++, DeepLabV3+ and SEANet; features come from experiment_features in the config."
    )
    parser.add_argument("--config", required=True, help="Path to experiment JSON configuration")
    parser.add_argument(
        "--stage",
        choices=("audit", "train", "validate", "test", "all"),
        default="all",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--features", nargs="+", choices=FEATURE_NAMES, default=None)
    parser.add_argument("--scales", nargs="+", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Rebuild matching completed outputs")
    parser.add_argument("--log-file", default=None, help="Optional combined stdout/stderr log path")
    return parser.parse_args()


class Tee:
    def __init__(self, terminal, log_handle) -> None:
        self.terminal = terminal
        self.log_handle = log_handle

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log_handle.write(value)
        self.log_handle.flush()
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_handle.flush()

    def isatty(self) -> bool:
        return False


def run(args: argparse.Namespace, cfg: dict, config_path: Path) -> None:
    if args.features is None:
        args.features = list(cfg.get("experiment_features", FEATURE_NAMES))
    stages = ["audit", "train", "validate", "test"] if args.stage == "all" else [args.stage]
    # Every executable stage is gated by the three-way spatial leakage audit.
    validate_config(cfg, set(stages) | {"audit"})
    output_root(cfg).mkdir(parents=True, exist_ok=True)
    copy_config_snapshot(cfg)
    print(f"Configuration: {config_path}")
    print(f"Output: {output_root(cfg)}")

    report = audit_spatial_splits(cfg)
    print(f"Spatial split audit: {report['status']}")

    if "train" in stages:
        for model_name in args.models:
            for feature_name in args.features:
                checkpoint = train_one(cfg, model_name, feature_name, args.force)
                print(f"Checkpoint: {checkpoint}")

    if "validate" in stages:
        selected = validate_all(cfg, args.models, args.features, args.scales, args.force)
        if selected.get("partial"):
            print(selected["message"])
        else:
            print("Validation selections:")
            for model_name, choice in selected["models"].items():
                print(
                    f"  {model_name}: feature={choice['feature']}, scale={choice['scale']}, "
                    f"mean_iou={choice['validation_mean_iou']:.6f}"
                )

    if "test" in stages:
        rows = test_selected(cfg, args.models, args.force)
        print("Locked test results:")
        for row in rows:
            print(
                f"  {row['model']}: feature={row['feature']}, scale={row['scale']}, "
                f"mean_iou={row['mean_iou']:.6f}"
            )


def main() -> None:
    args = parse_args()
    cfg, config_path = load_config(args.config)
    root = output_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_file).resolve() if args.log_file else root / "logs" / f"{timestamp}_{args.stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        sys.stdout = Tee(original_stdout, log_handle)
        sys.stderr = Tee(original_stderr, log_handle)
        try:
            print(f"Log: {log_path}")
            run(args, cfg, config_path)
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    main()
