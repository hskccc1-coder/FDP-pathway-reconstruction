"""Validate bundled inputs, Python syntax, portability, and package checksums."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REQUIRED = [
    "README.md",
    "environment.yml",
    "data/four_season_TN_TP/water_TN_TP_four_seasons_all_valid.csv",
    "data/four_season_TN_TP/water_TN_TP_four_seasons_balanced_12_sites.csv",
    "data/土壤采样点/土壤采样点.shp",
    "data/土地利用类型/小流域土地利用类型.shp",
    "data/矫正后河流线/river.shp",
    "data/outlet/outlet_snapped.shp",
    "data/单位侵蚀量/水土总氮多流向总氮1.tif",
    "code/run_all.py",
]
DRIVE_PATH = re.compile(r"(?:[rR][\"']|[\"'])[A-Za-z]:[\\/]")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate() -> list[str]:
    errors: list[str] = []
    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            errors.append(f"missing required file: {rel}")

    scripts = sorted((ROOT / "code").glob("*.py"))
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        try:
            ast.parse(text, filename=str(script))
        except SyntaxError as exc:
            errors.append(f"syntax error in {script.name}: {exc}")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if DRIVE_PATH.search(line) and "DRIVE_PATH" not in line:
                errors.append(f"absolute drive path in {script.name}:{line_no}")

    water_shapes = list((ROOT / "data" / "soil_water_measurements" / "water_seasonal_points" / "shp").glob("*.shp"))
    if len(water_shapes) != 8:
        errors.append(f"expected 8 seasonal water shapefiles, found {len(water_shapes)}")
    return errors


def write_manifest() -> Path:
    output = ROOT / "MANIFEST.csv"
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == output or "__pycache__" in path.parts:
            continue
        rows.append((path.relative_to(ROOT).as_posix(), path.stat().st_size, sha256(path)))
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "bytes", "sha256"])
        writer.writerows(rows)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    errors = validate()
    if errors:
        print("Package validation FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1
    if args.write_manifest:
        print(f"Wrote {write_manifest().relative_to(ROOT)}")
    print("Package validation PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
