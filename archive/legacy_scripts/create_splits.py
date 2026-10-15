#!/usr/bin/env python3
"""CLI script to generate and cache strict position-disjoint splits for NDM-1."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from zdyngnn.data.split import (
    PositionDisjointSplit,
    split_positions_disjoint,
    split_positions_kfold,
    verify_disjoint_invariants,
)


def generate_and_cache_splits(
    dynamics_path: str = "data/processed/dynamics_3spu.npz",
    dms_path: str = "data/processed/dms_missense_wide.csv",
    output_dir: str = "data/processed",
    seed: int = 42,
) -> dict:
    """Generate canonical 70/15/15 and 5-fold CV position-disjoint partitions."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading dynamics from {dynamics_path}...")
    dyn = np.load(dynamics_path)
    res_nums = dyn["residue_numbers"]
    strata = dyn["strata_bins"]
    n_res = len(res_nums)

    print(f"[*] Loading DMS missense table from {dms_path}...")
    dms_df = pd.read_csv(dms_path)
    n_mut = len(dms_df)

    print(f"    Residues: {n_res}, Missense Mutations: {n_mut}")

    # 1. Generate Canonical Single 70/15/15 Split
    print(f"[*] Generating canonical stratified 70/15/15 position-disjoint split (seed={seed})...")
    single_split = split_positions_disjoint(
        positions=res_nums,
        strata=strata,
        train_ratio=0.70,
        calib_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
        mutation_df=dms_df,
    )

    verify_disjoint_invariants(single_split, dms_df)
    print(f"    Positions: Train={single_split.num_train_positions} ({single_split.num_train_positions/n_res*100:.1f}%), "
          f"Calib={single_split.num_calib_positions} ({single_split.num_calib_positions/n_res*100:.1f}%), "
          f"Test={single_split.num_test_positions} ({single_split.num_test_positions/n_res*100:.1f}%)")
    print(f"    Mutations: Train={single_split.num_train_mutations} ({single_split.num_train_mutations/n_mut*100:.1f}%), "
          f"Calib={single_split.num_calib_mutations} ({single_split.num_calib_mutations/n_mut*100:.1f}%), "
          f"Test={single_split.num_test_mutations} ({single_split.num_test_mutations/n_mut*100:.1f}%)")

    single_split_path = out_dir / "splits_position_disjoint.json"
    with open(single_split_path, "w") as f:
        json.dump(single_split.to_dict(), f, indent=2)

    # 2. Generate 5-Fold Cross-Validation Splits
    print(f"[*] Generating 5-fold cross-validation position-disjoint splits (seed={seed})...")
    kfold_splits = split_positions_kfold(
        positions=res_nums,
        strata=strata,
        k_folds=5,
        seed=seed,
        mutation_df=dms_df,
    )

    kfold_dicts = []
    all_test_positions = []
    for fold_id, f_split in enumerate(kfold_splits):
        verify_disjoint_invariants(f_split, dms_df)
        all_test_positions.extend(f_split.test_positions)
        print(f"    Fold {fold_id}: Train={f_split.num_train_positions} pos ({f_split.num_train_mutations} mut), "
              f"Calib={f_split.num_calib_positions} pos ({f_split.num_calib_mutations} mut), "
              f"Test={f_split.num_test_positions} pos ({f_split.num_test_mutations} mut)")
        kfold_dicts.append(f_split.to_dict())

    # Invariant check: Across 5 folds, every position appears in test exactly once
    assert sorted(all_test_positions) == sorted(res_nums.tolist()), "5-fold test coverage must partition all positions exactly once."

    kfold_path = out_dir / "splits_5fold.json"
    with open(kfold_path, "w") as f:
        json.dump({"folds": kfold_dicts, "num_folds": 5}, f, indent=2)

    # 3. Summary Manifest
    summary = {
        "num_total_positions": int(n_res),
        "num_total_mutations": int(n_mut),
        "canonical_split": {
            "train_positions": int(single_split.num_train_positions),
            "calib_positions": int(single_split.num_calib_positions),
            "test_positions": int(single_split.num_test_positions),
            "train_mutations": int(single_split.num_train_mutations),
            "calib_mutations": int(single_split.num_calib_mutations),
            "test_mutations": int(single_split.num_test_mutations),
            "disjoint_verified": True,
        },
        "kfold_splits": {
            "num_folds": 5,
            "test_coverage_complete": True,
            "all_folds_disjoint_verified": True,
        },
        "files_generated": [
            str(single_split_path.name),
            str(kfold_path.name),
        ],
    }

    summary_path = out_dir / "splits_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Create position-disjoint splits for NDM-1.")
    parser.add_argument("--dynamics", default="data/processed/dynamics_3spu.npz", help="Path to dynamics cache.")
    parser.add_argument("--dms", default="data/processed/dms_missense_wide.csv", help="Path to DMS missense table.")
    parser.add_argument("--out", default="data/processed", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    summary = generate_and_cache_splits(args.dynamics, args.dms, args.out, args.seed)
    print("\n[+] Step 4 Position-Disjoint Splits Complete!")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
