#!/usr/bin/env python3
"""Train the 6 Supporting Pharmacodynamic Layer Models with Zero Data Leakage.

Implements the Dual-Tier Architecture:
  1. Primary Clinical Production Anchors: 256.0_AMP, 16.0_CTX, 0.25_MEM (in checkpoints/)
  2. Supporting Pharmacodynamic Layer: Low and High doses for each drug (in checkpoints/dose_response/)
     - Ampicillin: 32.0_AMP (Low), 512.0_AMP (High)
     - Cefotaxime: 2.0_CTX (Low), 32.0_CTX (High)
     - Meropenem:  0.0625_MEM (Low), 0.5_MEM (High)

Enforces strict zero-leakage invariant:
  - 100% position-disjoint (163 train positions, 32 calib positions, 33 test positions)
  - Zero test positions exposed to training or early stopping
  - Zero clinical hospital isolates or EC50 assays exposed to training
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

# Ensure scripts and src are on path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train_multibody_zdyn_gnn import train_and_benchmark_multibody
from zdyngnn.data.split import PositionDisjointSplit

SUPPORTING_CONDITIONS = [
    ("32.0_AMP", "Ampicillin (Low, 32 µg/mL)", "AMP"),
    ("512.0_AMP", "Ampicillin (High, 512 µg/mL)", "AMP"),
    ("2.0_CTX", "Cefotaxime (Low, 2 µg/mL)", "CTX"),
    ("32.0_CTX", "Cefotaxime (High, 32 µg/mL)", "CTX"),
    ("0.0625_MEM", "Meropenem (Low, 0.0625 µg/mL)", "MEM"),
    ("0.5_MEM", "Meropenem (High, 0.5 µg/mL)", "MEM"),
]

PRIMARY_ANCHORS = [
    ("256.0_AMP", "Ampicillin (Mid Anchor, 256 µg/mL)", "AMP"),
    ("16.0_CTX", "Cefotaxime (Mid Anchor, 16 µg/mL)", "CTX"),
    ("0.25_MEM", "Meropenem (Mid Anchor, 0.25 µg/mL)", "MEM"),
]


def run_pharmacodynamic_layer_training(epochs: int = 35) -> Dict:
    print("=" * 84)
    print("PHARMODYNAMIC LAYER: ZERO-LEAKAGE TRAINING OF 6 SUPPORTING DOSE MODELS")
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
    print(f"[*] Training 6 Supporting Pharmacodynamic Models ({epochs} epochs each)...\n")

    dose_dir = Path("checkpoints/dose_response")
    dose_dir.mkdir(parents=True, exist_ok=True)

    supporting_records = []
    t_start = time.time()

    for idx, (cond, desc, drug) in enumerate(SUPPORTING_CONDITIONS, 1):
        print(f"\n>>> [{idx}/6] TRAINING SUPPORTING DOSE: {cond} ({desc})")
        comp = train_and_benchmark_multibody(condition=cond, epochs=epochs)
        mb = comp["multibody"]
        base = comp["baseline"]

        # Move/copy checkpoint into checkpoints/dose_response/
        src_ckpt = Path(f"checkpoints/zdyn_gnn_multibody_{cond}.pt")
        dst_ckpt = dose_dir / f"zdyn_gnn_multibody_{cond}.pt"
        if src_ckpt.exists() and src_ckpt != dst_ckpt:
            import shutil
            shutil.copy2(src_ckpt, dst_ckpt)

        supporting_records.append({
            "tier": "Supporting Pharmacodynamic Layer",
            "drug": drug,
            "condition": cond,
            "description": desc,
            "test_pearson_r": mb["metrics"]["pearson_raw"],
            "norm_pearson": mb["metrics"]["pearson_normalized"],
            "test_r2": mb["metrics"]["r2_raw"],
            "test_rmse": mb["metrics"]["rmse"],
            "conformal_coverage": mb["coverage"],
            "mean_interval_width": mb["mean_width"],
            "false_safety_errors": mb["false_safety_errors"],
            "checkpoint_path": str(dst_ckpt),
        })

    elapsed = time.time() - t_start
    print("\n" + "=" * 96)
    print(f"SUPPORTING PHARMACODYNAMIC LAYER TRAINED IN {elapsed:.1f}s (ZERO DATA LEAKAGE)")
    print("=" * 96)
    print(f"{'Condition':<12} {'Description':<32} {'Test r':<8} {'r/r_max':<8} {'Test R²':<8} {'RMSE':<8} {'Cov (≥95%)':<12} {'Width (2B_i)'}")
    print("-" * 96)

    for r in supporting_records:
        print(
            f"{r['condition']:<12} {r['description']:<32} "
            f"{r['test_pearson_r']:<8.4f} {r['norm_pearson']*100:<7.1f}% "
            f"{r['test_r2']:<8.4f} {r['test_rmse']:<8.4f} "
            f"{r['conformal_coverage']:<11.2f}% {r['mean_interval_width']:.2f}"
        )

    # 2. Build Master 9-Condition Dose-Response Explorer Manifest
    # Ingest primary anchors metrics
    all_9_records = []

    # Read existing 3 primary anchors from comparison JSON or evaluate
    for cond, desc, drug in PRIMARY_ANCHORS:
        pred_csv = Path(f"results/multibody_predictions_{cond}.csv")
        if pred_csv.exists():
            df = pd.read_csv(pred_csv)
            y = df["fitness_score"].values
            mb_mu = df["mb_mu"].values
            r_val = float(np.corrcoef(y, mb_mu)[0, 1])
            r2_val = float(1.0 - np.sum((y - mb_mu)**2) / np.sum((y - np.mean(y))**2))
            rmse_val = float(np.sqrt(np.mean((y - mb_mu)**2)))
            cov_val = float(np.mean((y >= df["mb_lower"].values) & (y <= df["mb_upper"].values)) * 100.0)
            w_val = float(np.mean(df["mb_upper"].values - df["mb_lower"].values))
            fs_val = int(np.sum((df["mb_upper"].values < -2.0) & (y >= -2.0)))

            all_9_records.append({
                "tier": "Primary Clinical Production Anchor",
                "drug": drug,
                "condition": cond,
                "description": desc,
                "test_pearson_r": r_val,
                "norm_pearson": r_val / 0.98,  # approximate ceiling
                "test_r2": r2_val,
                "test_rmse": rmse_val,
                "conformal_coverage": cov_val,
                "mean_interval_width": w_val,
                "false_safety_errors": fs_val,
                "checkpoint_path": f"checkpoints/zdyn_gnn_multibody_{cond}.pt",
            })

    all_9_records.extend(supporting_records)

    # Sort logically: AMP (32, 256, 512), CTX (2, 16, 32), MEM (0.0625, 0.25, 0.5)
    sort_order = {
        "32.0_AMP": 1, "256.0_AMP": 2, "512.0_AMP": 3,
        "2.0_CTX": 4, "16.0_CTX": 5, "32.0_CTX": 6,
        "0.0625_MEM": 7, "0.25_MEM": 8, "0.5_MEM": 9,
    }
    all_9_records.sort(key=lambda x: sort_order.get(x["condition"], 99))

    out_json = Path("results/dose_response_explorer_9cond.json")
    with open(out_json, "w") as f:
        json.dump(all_9_records, f, indent=2)
    print(f"\n[+] Saved complete 9-condition Dose-Response Explorer JSON to {out_json}")

    out_csv = Path("results/dose_response_explorer_9cond.csv")
    pd.DataFrame(all_9_records).to_csv(out_csv, index=False)
    print(f"[+] Saved complete 9-condition Dose-Response Explorer CSV to {out_csv}")

    return {
        "supporting_records": supporting_records,
        "all_9_records": all_9_records,
    }


if __name__ == "__main__":
    run_pharmacodynamic_layer_training(epochs=35)
