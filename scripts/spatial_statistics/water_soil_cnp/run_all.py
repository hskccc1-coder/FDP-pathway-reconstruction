"""Run the Ecological Informatics reproducibility workflow in dependency order."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent

STAGES = [
    ("01_network_distance", "figure3_network_distance_stage1.py", "primary"),
    ("04_multivariable_diagnostics", "water_soil_multivariable_phase1.py", "primary"),
    ("05_initial_models", "water_soil_multivariable_phase2.py", "primary"),
    ("06_soil_covariates", "water_soil_multivariable_phase3_soil_covariates.py", "primary"),
    ("07_soil_models", "water_soil_multivariable_phase4_soil_final.py", "primary"),
    ("08_distance_decision", "water_soil_multivariable_phase4_distance_decision.py", "primary"),
    ("09_final_models", "water_soil_multivariable_phase5_final.py", "primary"),
    ("10_raw_vs_residual_moran", "raw_vs_residual_moransI.py", "primary"),
    ("11_robustness", "water_soil_multivariable_phase6_statistical_robustness.py", "primary"),
    ("12_manuscript_figures_cd", "manuscript_backhalf_figures_final.py", "primary"),
    ("13_figure_e_n_transport", "manuscript_figureE_nload_real.py", "primary"),
    ("14_figure_e_panel_b", "manuscript_figureE_panelB_optimize.py", "primary"),
    ("15_flowacc_sensitivity", "manuscript_figureE_flowacc_pathway.py", "external"),
    ("16_publication_figures", "nature_style_recommended_figures.py", "primary"),
    ("17_combined_figure", "combined_nature_figure.py", "primary"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List stages and exit.")
    parser.add_argument("--only", choices=[s[0] for s in STAGES], help="Run one stage only.")
    parser.add_argument(
        "--include-optional-flowacc",
        action="store_true",
        help="Run the FlowAcc sensitivity stage (requires ECOINFO_FLOWACC_RASTER or the documented file).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        for name, script, category in STAGES:
            print(f"{name:30s} {script} [{category}]")
        return 0

    selected = STAGES
    if args.only:
        selected = [stage for stage in STAGES if stage[0] == args.only]

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    for name, script, category in selected:
        if category == "external" and not args.include_optional_flowacc and not args.only:
            print(f"SKIP {name}: optional external FlowAcc raster not bundled")
            continue
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable, str(CODE / script)], cwd=str(ROOT), env=env, check=True)
    print("\nWorkflow completed. Generated files are under results/pathway_metrics/water_soil_cnp/runtime/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
