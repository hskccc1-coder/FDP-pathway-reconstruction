from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

from common import save_json, sha256_dataset


def load_evaluation(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("split") != "validation":
        raise ValueError(f"Selection accepts validation results only: {path}")
    required = {"candidate_id", "feature", "scale_px", "prediction", "per_class"}
    missing = sorted(required.difference(value))
    if missing:
        raise KeyError(f"Missing evaluation fields in {path}: {missing}")
    if not value.get("registered_roi"):
        raise ValueError(f"Evaluation was not checked against a registered validation region: {path}")
    prediction = Path(value["prediction"])
    if not prediction.is_file() or sha256_dataset(prediction) != value["input_sha256"]["prediction"]:
        raise ValueError(f"Prediction is missing or changed after evaluation: {prediction}")
    value["evaluation_file"] = str(path.resolve())
    return value


def select(evaluations: list[dict]) -> dict:
    if not evaluations:
        raise ValueError("At least one validation evaluation is required")
    selected = {}
    for class_name in ["Field", "Interstrip", "Water"]:
        candidates = []
        for evaluation in evaluations:
            row = next(
                (item for item in evaluation["per_class"] if item["class_name"] == class_name),
                None,
            )
            if row is None:
                raise KeyError(f"{evaluation['candidate_id']} has no {class_name} metrics")
            if not math.isfinite(float(row["IoU"])) or not math.isfinite(float(row["F1"])):
                raise ValueError(
                    f"{evaluation['candidate_id']} has non-finite {class_name} metrics"
                )
            candidates.append(
                {
                    "candidate_id": evaluation["candidate_id"],
                    "feature": evaluation["feature"],
                    "scale_px": int(evaluation["scale_px"]),
                    "prediction": evaluation["prediction"],
                    "prediction_sha256": evaluation["input_sha256"]["prediction"],
                    "evaluation_file": evaluation["evaluation_file"],
                    "IoU": float(row["IoU"]),
                    "F1": float(row["F1"]),
                }
            )
        candidates.sort(key=lambda item: (-item["IoU"], -item["F1"], item["candidate_id"]))
        selected[class_name] = candidates[0]
    return {
        "selection_split": "validation",
        "selection_rule": "maximum_IoU_then_F1_then_candidate_id",
        "selected": selected,
        "candidate_count": len(evaluations),
        "candidate_ids": sorted(value["candidate_id"] for value in evaluations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select class-specific N-SR candidates on validation data")
    parser.add_argument("--evaluations", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expanded = []
    for value in args.evaluations:
        matches = sorted(glob.glob(value))
        expanded.extend(matches if matches else [value])
    evaluations = [load_evaluation(Path(value).resolve()) for value in expanded]
    identifiers = [value["candidate_id"] for value in evaluations]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Candidate identifiers must be unique")
    expected_grid = {(feature, scale) for feature in ["rgb", "slope"] for scale in [512, 1024, 2048, 4096]}
    actual_grid = {(value["feature"], int(value["scale_px"])) for value in evaluations}
    if actual_grid != expected_grid:
        missing = sorted(expected_grid.difference(actual_grid))
        extra = sorted(actual_grid.difference(expected_grid))
        raise ValueError(f"Selection requires the complete eight-candidate grid; missing={missing}, extra={extra}")
    protocol_fields = [
        "field_truth",
        "water_truth",
        "roi",
        "interstrip_region",
        "class_field",
        "class_values",
        "class_priority",
        "overlap_tolerance_m2",
    ]
    reference = {field: evaluations[0][field] for field in protocol_fields}
    reference_hashes = {
        name: evaluations[0]["input_sha256"][name]
        for name in ["field_truth", "water_truth", "roi", "interstrip_region"]
    }
    for evaluation in evaluations[1:]:
        current = {field: evaluation[field] for field in protocol_fields}
        current_hashes = {
            name: evaluation["input_sha256"][name]
            for name in ["field_truth", "water_truth", "roi", "interstrip_region"]
        }
        if current != reference or current_hashes != reference_hashes:
            raise ValueError("All validation candidates must use the same evaluation protocol and input hashes")
    result = select(evaluations)
    result["evaluation_protocol"] = reference
    result["evaluation_input_sha256"] = reference_hashes
    save_json(Path(args.output).resolve(), result)


if __name__ == "__main__":
    main()
