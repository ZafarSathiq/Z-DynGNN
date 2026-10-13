#!/usr/bin/env python3
"""Phase I: Data Foundation, Experimental Noise Ceilings & 9-Dose Pharmacodynamics.

Consolidates Engineering Steps 1, 2, and 8 into a single unified data foundation:
  1. Ingests PDB 3SPU C-alpha coordinates (Chain A, residues 43-270).
  2. Ingests all 4,271 missense variants across all 9 drug conditions (38,439 measurements).
  3. Computes the formal Law of Total Biological Variance decomposition for all 9 conditions.
  4. Audits physical dose-response monotonicity (y_high <= y_mid <= y_low) with replicate noise tolerance.
  5. Exports curated 9-condition dataset and Phase I master manifest.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from zdyngnn.data.dms_loader import (
    DMS_CONDITIONS,
    filter_missense_dms_data,
    load_raw_dms_excel,
    melt_to_long_format,
)
from zdyngnn.data.noise_ceiling import compute_condition_noise_ceiling
from zdyngnn.data.pdb_parser import parse_pdb_structure

CONDITIONS_9 = [
    ("Ampicillin", "32.0_AMP", 32.0),
    ("Ampicillin", "256.0_AMP", 256.0),
    ("Ampicillin", "512.0_AMP", 512.0),
    ("Cefotaxime", "2.0_CTX", 2.0),
    ("Cefotaxime", "16.0_CTX", 16.0),
    ("Cefotaxime", "32.0_CTX", 32.0),
    ("Meropenem", "0.0625_MEM", 0.0625),
    ("Meropenem", "0.25_MEM", 0.25),
    ("Meropenem", "0.5_MEM", 0.5),
]

DRUG_SERIES = {
    "Ampicillin": ["32.0_AMP", "256.0_AMP", "512.0_AMP"],
    "Cefotaxime": ["2.0_CTX", "16.0_CTX", "32.0_CTX"],
    "Meropenem": ["0.0625_MEM", "0.25_MEM", "0.5_MEM"],
}


def run_phase1_pipeline(
    pdb_path: str = "DMS Data NDM-1/Strutural data/3SPU.pdb",
    dms_path: str = "DMS Data NDM-1/41467_2024_52614_MOESM6_ESM.xlsx",
    output_dir: str = "data/processed",
    results_dir: str = "results",
    distance_cutoff: float = 8.0,
    noise_tolerance: float = 0.51,
) -> dict:
    """Execute Phase I unified data foundation and pharmacodynamics pipeline."""
    out_p = Path(output_dir)
    res_p = Path(results_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    res_p.mkdir(parents=True, exist_ok=True)

    print("=====================================================================")
    print("PHASE I: DATA FOUNDATION, NOISE CEILINGS & 9-DOSE PHARMACODYNAMICS")
    print("=====================================================================")

    # 1. Structural Parsing: PDB 3SPU C-alpha coordinates
    print(f"\n[*] Parsing native reference structure: {pdb_path}...")
    pdb_data = parse_pdb_structure(
        pdb_path=pdb_path,
        chain_id="A",
        res_start=43,
        res_end=270,
        distance_cutoff=distance_cutoff,
    )
    n_res = pdb_data.num_residues
    print(f"[+] Successfully resolved Chain A: {n_res} residues (positions {pdb_data.residue_numbers[0]} to {pdb_data.residue_numbers[-1]}).")
    print(f"    Contact degree mean: {pdb_data.degrees.mean():.2f} (range {pdb_data.degrees.min()} to {pdb_data.degrees.max()} contacts at {distance_cutoff} Å).")

    # 2. Ingest Full 9-Condition DMS Landscape
    print(f"\n[*] Ingesting Chen et al. 2024 DMS supplement: {dms_path}...")
    raw_dms = load_raw_dms_excel(file_path=dms_path)
    dms_wide = filter_missense_dms_data(
        df=raw_dms,
        res_start=int(pdb_data.residue_numbers[0]),
        res_end=int(pdb_data.residue_numbers[-1]),
    )
    n_muts = len(dms_wide)
    print(f"[+] Loaded {n_muts} missense substitutions across all 9 drug concentration conditions.")
    print(f"    Total measurement points: {n_muts * 9}")

    # 3. Law of Total Biological Variance & Noise Ceilings across all 9 conditions
    print(f"\n[*] Evaluating Law of Total Biological Variance across all 9 conditions...")
    noise_ceilings = {}
    for drug_name, cond_name, conc_ug_ml in CONDITIONS_9:
        score_col = f"{cond_name}_score"
        sd_col = f"{cond_name}_score_sd"
        res = compute_condition_noise_ceiling(
            scores=dms_wide[score_col],
            sds=dms_wide[sd_col],
            condition_name=cond_name,
        )
        noise_ceilings[cond_name] = {
            "drug": drug_name,
            "concentration_ug_ml": conc_ug_ml,
            "total_variance": float(res.total_variance),
            "replicate_error_variance": float(res.error_variance),
            "explainable_signal_variance": float(res.signal_variance),
            "r2_max": float(res.r2_max),
            "r_max": float(res.r_max),
            "snr": float(res.snr),
            "num_valid_measurements": int(res.num_samples),
        }
        print(f"    {cond_name:12s} ({conc_ug_ml:8.4f} ug/mL): Var(y)={res.total_variance:.3f}, Var(err)={res.error_variance:.3f} | r_max={res.r_max:.4f}, SNR={res.snr:.2f}")

    # 4. Dose-Response Monotonicity Auditing across 3 drug series
    print(f"\n[*] Auditing physical dose-response monotonicity (y_high <= y_mid <= y_low)...")
    monotonicity_audit = {}

    for drug, cond_list in DRUG_SERIES.items():
        y_low = dms_wide[f"{cond_list[0]}_score"].values
        y_mid = dms_wide[f"{cond_list[1]}_score"].values
        y_high = dms_wide[f"{cond_list[2]}_score"].values

        # Strict monotonicity
        strict_full = np.mean((y_high <= y_mid) & (y_mid <= y_low))
        # Monotonicity with experimental replicate noise tolerance
        tol_full = np.mean((y_high <= y_mid + noise_tolerance) & (y_mid <= y_low + noise_tolerance))
        low_to_mid_tol = np.mean(y_mid <= y_low + noise_tolerance)
        mid_to_high_tol = np.mean(y_high <= y_mid + noise_tolerance)

        monotonicity_audit[drug] = {
            "conditions": cond_list,
            "total_mutations_evaluated": n_muts,
            "strict_monotonicity_rate": float(strict_full),
            "noise_tolerant_monotonicity_rate": float(tol_full),
            "low_to_mid_dose_compliance": float(low_to_mid_tol),
            "mid_to_high_dose_compliance": float(mid_to_high_tol),
        }
        print(f"    {drug:12s}: Low->Mid compliance = {low_to_mid_tol*100:.2f}%, Overall 3-step compliance = {tol_full*100:.2f}% (eps={noise_tolerance} SD)")

    # 5. Export Phase I Curated Datasets
    curated_csv_path = out_p / "phase1_dms_9cond_curated.csv"
    dms_wide.to_csv(curated_csv_path, index=False)
    print(f"\n[+] Saved curated 9-condition DMS dataset to {curated_csv_path}")

    phase1_manifest = {
        "phase": "PHASE I: DATA FOUNDATION & 9-DOSE PHARMACODYNAMICS",
        "reference_structure": {
            "pdb_id": pdb_data.pdb_id,
            "chain_id": pdb_data.chain_id,
            "num_residues": n_res,
            "residue_range": [int(pdb_data.residue_numbers[0]), int(pdb_data.residue_numbers[-1])],
            "degree_mean": float(pdb_data.degrees.mean()),
            "degree_min": int(pdb_data.degrees.min()),
            "degree_max": int(pdb_data.degrees.max()),
            "cutoff_angstroms": distance_cutoff,
        },
        "dms_landscape": {
            "num_missense_variants": n_muts,
            "num_conditions": len(CONDITIONS_9),
            "total_measurements": n_muts * 9,
        },
        "noise_ceilings": noise_ceilings,
        "dose_response_monotonicity": monotonicity_audit,
        "files_generated": [
            str(curated_csv_path.name),
            "phase1_pharmacodynamics_manifest.json",
        ],
    }

    manifest_json_path = res_p / "phase1_pharmacodynamics_manifest.json"
    with open(manifest_json_path, "w") as f:
        json.dump(phase1_manifest, f, indent=2)
    print(f"[+] Saved Phase I master manifest to {manifest_json_path}")

    return phase1_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase I Data Foundation & Pharmacodynamics CLI.")
    parser.add_argument("--pdb", default="DMS Data NDM-1/Strutural data/3SPU.pdb", help="Path to PDB file.")
    parser.add_argument("--dms", default="DMS Data NDM-1/41467_2024_52614_MOESM6_ESM.xlsx", help="Path to DMS Excel file.")
    parser.add_argument("--out", default="data/processed", help="Output directory.")
    parser.add_argument("--cutoff", type=float, default=8.0, help="Distance cutoff in Angstroms.")
    parser.add_argument("--noise_tol", type=float, default=0.51, help="Replicate noise tolerance.")
    args = parser.parse_args()

    run_phase1_pipeline(
        pdb_path=args.pdb,
        dms_path=args.dms,
        output_dir=args.out,
        distance_cutoff=args.cutoff,
        noise_tolerance=args.noise_tol,
    )
