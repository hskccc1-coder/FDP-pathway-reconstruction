from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import rasterio

from prepare_nsr_raster import _aligned_array, _rasterize_mask, _sha256_dataset


def _binary_metrics(truth: np.ndarray, prediction: np.ndarray, region: np.ndarray) -> dict[str, float]:
    target = truth & region
    predicted = prediction & region
    tp = int(np.count_nonzero(target & predicted))
    fp = int(np.count_nonzero(~target & predicted & region))
    fn = int(np.count_nonzero(target & ~predicted))
    denominator = tp + fp + fn
    iou = tp / denominator if denominator else math.nan
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else math.nan
    return {"IoU": iou, "F1": f1, "TP_pixels": tp, "FP_pixels": fp, "FN_pixels": fn}


def select(args: argparse.Namespace) -> None:
    config_path = Path(args.candidates).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    if not candidates:
        raise ValueError("Candidate configuration contains no candidates")
    reference_path = Path(args.reference_raster).resolve()
    truth_path = Path(args.truth_raster).resolve()
    zone_path = Path(args.interstrip_region).resolve()
    for path in [reference_path, truth_path, zone_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    with rasterio.open(reference_path) as reference:
        truth = _aligned_array(truth_path, reference).astype(np.uint8)
        valid = truth != args.nodata
        zone = _rasterize_mask(zone_path, reference, all_touched=args.all_touched) & valid
        regions = {
            "Field": valid & ~zone,
            "Interstrip": valid & zone,
            "Water": valid & ~zone,
        }
        truth_masks = {
            "Field": truth == args.field_class_id,
            "Interstrip": truth == args.field_class_id,
            "Water": truth == args.water_class_id,
        }

        rows: list[dict[str, object]] = []
        candidate_lookup: dict[str, dict[str, object]] = {}
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in candidate_lookup:
                raise ValueError(f"Duplicate candidate_id: {candidate_id}")
            vector_path = Path(candidate["prediction"]).resolve()
            if not vector_path.is_file():
                raise FileNotFoundError(vector_path)
            candidate_lookup[candidate_id] = {**candidate, "prediction": str(vector_path)}
            field_mask = _rasterize_mask(
                vector_path,
                reference,
                str(candidate["class_field"]),
                [str(value) for value in candidate["field_values"]],
                args.all_touched,
            )
            water_mask = _rasterize_mask(
                vector_path,
                reference,
                str(candidate["class_field"]),
                [str(value) for value in candidate["water_values"]],
                args.all_touched,
            )
            for evaluation_class, prediction in [
                ("Field", field_mask),
                ("Interstrip", field_mask),
                ("Water", water_mask),
            ]:
                metrics = _binary_metrics(
                    truth_masks[evaluation_class], prediction, regions[evaluation_class]
                )
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "feature": candidate["feature"],
                        "scale_px": int(candidate["scale_px"]),
                        "evaluation_class": evaluation_class,
                        **metrics,
                        "region_valid_pixels": int(np.count_nonzero(regions[evaluation_class])),
                        "prediction": str(vector_path),
                    }
                )

    selected: dict[str, dict[str, object]] = {}
    for evaluation_class in ["Field", "Interstrip", "Water"]:
        options = [row for row in rows if row["evaluation_class"] == evaluation_class]
        options.sort(
            key=lambda row: (
                -float(row["IoU"]),
                -float(row["F1"]),
                str(row["candidate_id"]),
            )
        )
        best = options[0]
        candidate = candidate_lookup[str(best["candidate_id"])]
        selected[evaluation_class] = {
            **best,
            "class_field": candidate["class_field"],
            "field_values": candidate["field_values"],
            "water_values": candidate["water_values"],
            "prediction_sha256": _sha256_dataset(Path(str(best["prediction"]))),
        }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    selection = {
        "selection_split": "validation",
        "selection_rule": "maximum IoU; ties by F1 then candidate_id",
        "water_inside_interstrip": "excluded_from_selection_and_evaluation",
        "class_ids": {"field": args.field_class_id, "water": args.water_class_id},
        "all_touched": args.all_touched,
        "reference_raster": str(reference_path),
        "truth_raster": str(truth_path),
        "interstrip_region": str(zone_path),
        "selected": selected,
        "input_sha256": {
            "reference_raster": _sha256_dataset(reference_path),
            "truth_raster": _sha256_dataset(truth_path),
            "interstrip_region": _sha256_dataset(zone_path),
        },
    }
    output.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    table_path = output.with_suffix(".csv")
    with table_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(selection, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select N-SR feature-scale candidates on a locked validation raster"
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--reference-raster", required=True)
    parser.add_argument("--truth-raster", required=True)
    parser.add_argument("--interstrip-region", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--field-class-id", type=int, default=1)
    parser.add_argument("--water-class-id", type=int, default=2)
    parser.add_argument("--nodata", type=int, default=255)
    parser.add_argument("--all-touched", action="store_true")
    select(parser.parse_args())


if __name__ == "__main__":
    main()
