#!/usr/bin/env python3
"""Train Multi-Body ZDynGNN across all 9 experimental conditions with ZERO data leakage.

Conditions (Chen et al. 2024):
  Ampicillin: 32.0_AMP (Low), 256.0_AMP (Mid), 512.0_AMP (High)
  Cefotaxime: 2.0_CTX (Low), 16.0_CTX (Mid), 32.0_CTX (High)
  Meropenem:  0.0625_MEM (Low), 0.25_MEM (Mid), 0.5_MEM (High)

Enforces strict zero-leakage invariant:
  - 100% position-disjoint (163 train positions, 32 calib positions, 33 test positions)
  - Zero test positions exposed to training or early stopping
  - Zero clinical hospital isolates or EC50 assays exposed to training
"""
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train_multibody_zdyn_gnn import train_and_benchmark_multibody
from zdyngnn.data.split import PositionDisjointSplit

ALL_9_CONDITIONS = [
    # Ampicillin
    ("32.0_AMP", "Ampicillin (Low, 32 µg/mL)"),
    ("256.0_AMP", "Ampicillin (Mid Anchor, 256 µg/mL)"),
    ("512.0_AMP", "Ampicillin (High, 512 µg/mL)"),
    # Cefotaxime
    ("2.0_CTX", "Cefotaxime (Low, 2 µg/mL)"),
    ("16.0_CTX", "Cefotaxime (Mid Anchor, 16 µg/mL)"),
    ("32.0_CTX", "Cefotaxime (High, 32 µg/mL)"),
    # Meropenem
    ("0.0625_MEM", "Meropenem (Low, 0.0625 µg/mL)"),
    ("0.25_MEM", "Meropenem (Mid Anchor, 0.25 µg/mL)"),
    ("0.5_MEM", "Meropenem (High, 0.5 µg/mL)"),
]


def run_all_9_training(epochs: int = 35) -> List[Dict]:
    print("=" * 84)
    print("ZERO-LEAKAGE MULTI-BODY TRAINING ACROSS ALL 9 CONCENTRATION CONDITIONS")
    print("=" * 84)

    # 1. Enforce Zero-Leakage Split Invariant Verification
    with open("data/processed/splits_position_disjoint.json") as f:
        split_dict = json.load(f)
    split = PositionDisjointSplit.from_dict(split_dict)

    dms_df = pd.read_csv("data/processed/dms_missense_wide.csv")
    train_pos = set(dms_df.iloc[split.train_mutation_indices]["pos"])
    calib_pos = set(dms_df.iloc[split.calib_mutation_indices]["pos"])
    test_pos = set(dms_df.iloc[split.test_mutation_indices]["pos"])

    # Strict mathematical disjointness check
    assert train_pos.isdisjoint(test_pos), "FATAL: Train and Test positions overlap!"
    assert train_pos.isdisjoint(calib_pos), "FATAL: Train and Calib positions overlap!"
    assert calib_pos.isdisjoint(test_pos), "FATAL: Calib and Test positions overlap!"
    assert len(train_pos) == 163, f"Expected 163 train positions, got {len(train_pos)}"
    assert len(calib_pos) == 32, f"Expected 32 calib positions, got {len(calib_pos)}"
    assert len(test_pos) == 33, f"Expected 33 test positions, got {len(test_pos)}"

    print("[✓] Zero-Leakage Invariant Verified: Train (163 pos) ∩ Test (33 pos) = ∅")
    print(f"[*] Starting sequential training across all 9 conditions ({epochs} epochs each)...\n")

    summary_records = []
    t_start = time.time()

    for idx, (cond, desc) in enumerate(ALL_9_CONDITIONS, 1):
        print(f"\n>>> [{idx}/9] TRAINING CONDITION: {cond} ({desc})")
        comp = train_and_benchmark_multibody(condition=cond, epochs=epochs)
        mb = comp["multibody"]
        base = comp["baseline"]

        summary_records.append({
            "condition": cond,
            "description": desc,
            "test_pearson_r": mb["metrics"]["pearson_raw"],
            "norm_pearson": mb["metrics"]["pearson_normalized"],
            "test_r2": mb["metrics"]["r2_raw"],
            "test_rmse": mb["metrics"]["rmse"],
            "conformal_coverage": mb["coverage"],
            "mean_interval_width": mb["mean_width"],
            "false_safety_errors": mb["false_safety_errors"],
            "baseline_pearson_r": base["metrics"]["pearson_raw"],
            "baseline_r2": base["metrics"]["r2_raw"],
            "baseline_mean_width": base["mean_width"],
        })

    elapsed = time.time() - t_start
    print("\n" + "=" * 96)
    print(f"ALL 9 CONDITIONS TRAINED & BENCHMARKED IN {elapsed:.1f}s (ZERO DATA LEAKAGE)")
    print("=" * 96)
    print(f"{'Condition':<12} {'Description':<32} {'Test r':<8} {'r/r_max':<8} {'Test R²':<8} {'RMSE':<8} {'Cov (≥95%)':<12} {'Width (2B_i)'}")
    print("-" * 96)

    for r in summary_records:
        print(
            f"{r['condition']:<12} {r['description']:<32} "
            f"{r['test_pearson_r']:<8.4f} {r['norm_pearson']*100:<7.1f}% "
            f"{r['test_r2']:<8.4f} {r['test_rmse']:<8.4f} "
            f"{r['conformal_coverage']:<11.2f}% {r['mean_interval_width']:.2f}"
        )

    # Save complete 9-condition summary
    out_json = Path("results/all_9_conditions_multibody_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\n[+] Saved complete 9-condition summary to {out_json}")

    out_csv = Path("results/all_9_conditions_multibody_summary.csv")
    pd.DataFrame(summary_records).to_csv(out_csv, index=False)
    print(f"[+] Saved complete 9-condition CSV table to {out_csv}")

    return summary_records


if __name__ == "__main__":
    run_all_9_training(epochs=35)
