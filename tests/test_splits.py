"""Unit and integration tests for Step 4 Position-Disjoint Splitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zdyngnn.data.split import (
    PositionDisjointSplit,
    map_split_to_mutation_indices,
    split_positions_disjoint,
    split_positions_kfold,
    verify_disjoint_invariants,
)

DYNAMICS_CACHE = "data/processed/dynamics_3spu.npz"
DMS_CACHE = "data/processed/dms_missense_wide.csv"
SPLIT_CANONICAL = "data/processed/splits_position_disjoint.json"
SPLIT_KFOLD = "data/processed/splits_5fold.json"


@pytest.fixture
def dynamics_data():
    assert Path(DYNAMICS_CACHE).exists(), "Run scripts/compute_dynamics.py first."
    return np.load(DYNAMICS_CACHE)


@pytest.fixture
def dms_data():
    assert Path(DMS_CACHE).exists(), "Run scripts/preprocess_data.py first."
    return pd.read_csv(DMS_CACHE)


def test_position_disjointness_invariant(dynamics_data, dms_data):
    """Verify that positional partitions are strictly disjoint and cover all 228 positions."""
    res_nums = dynamics_data["residue_numbers"]
    strata = dynamics_data["strata_bins"]
    n_res = len(res_nums)

    split = split_positions_disjoint(
        positions=res_nums,
        strata=strata,
        train_ratio=0.70,
        calib_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        mutation_df=dms_data,
    )

    s_train = set(split.train_positions)
    s_calib = set(split.calib_positions)
    s_test = set(split.test_positions)

    # 1. Zero positional leakage
    assert s_train.isdisjoint(s_calib), "Train and Calib must be disjoint."
    assert s_train.isdisjoint(s_test), "Train and Test must be disjoint."
    assert s_calib.isdisjoint(s_test), "Calib and Test must be disjoint."

    # 2. Complete coverage
    assert s_train | s_calib | s_test == set(res_nums)
    assert len(s_train) + len(s_calib) + len(s_test) == n_res

    # 3. Ratio sanity (70 / 15 / 15 +/- 3%)
    assert 0.65 <= len(s_train) / n_res <= 0.75
    assert 0.12 <= len(s_calib) / n_res <= 0.18
    assert 0.12 <= len(s_test) / n_res <= 0.18


def test_mutation_index_disjointness(dynamics_data, dms_data):
    """Verify that mutation row indices mapped from positions are strictly disjoint and complete."""
    res_nums = dynamics_data["residue_numbers"]
    strata = dynamics_data["strata_bins"]
    n_mut = len(dms_data)

    split = split_positions_disjoint(
        positions=res_nums,
        strata=strata,
        seed=42,
        mutation_df=dms_data,
    )

    idx_train = set(split.train_mutation_indices)
    idx_calib = set(split.calib_mutation_indices)
    idx_test = set(split.test_mutation_indices)

    # Disjointness
    assert idx_train.isdisjoint(idx_calib)
    assert idx_train.isdisjoint(idx_test)
    assert idx_calib.isdisjoint(idx_test)

    # Completeness
    assert len(idx_train) + len(idx_calib) + len(idx_test) == n_mut
    assert idx_train | idx_calib | idx_test == set(range(n_mut))

    # Verify positions of mutations in test partition NEVER appear in train or calib
    test_positions_in_mutations = set(dms_data.iloc[list(idx_test)]["pos"])
    train_positions_in_mutations = set(dms_data.iloc[list(idx_train)]["pos"])
    calib_positions_in_mutations = set(dms_data.iloc[list(idx_calib)]["pos"])

    assert test_positions_in_mutations.isdisjoint(train_positions_in_mutations)
    assert test_positions_in_mutations.isdisjoint(calib_positions_in_mutations)


def test_kfold_cross_validation_invariants(dynamics_data, dms_data):
    """Verify that 5-fold CV splits form a valid partition where each position is tested once."""
    res_nums = dynamics_data["residue_numbers"]
    strata = dynamics_data["strata_bins"]

    kfold_splits = split_positions_kfold(
        positions=res_nums,
        strata=strata,
        k_folds=5,
        seed=42,
        mutation_df=dms_data,
    )

    assert len(kfold_splits) == 5

    all_test_positions = []
    all_test_mutation_indices = []

    for fold_id, f_split in enumerate(kfold_splits):
        assert verify_disjoint_invariants(f_split, dms_data) is True
        all_test_positions.extend(f_split.test_positions)
        all_test_mutation_indices.extend(f_split.test_mutation_indices)

        # Check fold sizes: ~20% of 228 is ~45 positions
        assert 35 <= f_split.num_test_positions <= 55
        assert 130 <= f_split.num_train_positions <= 165
        assert 25 <= f_split.num_calib_positions <= 40

    # Invariant: Every position appears in test partition of exactly one fold
    assert sorted(all_test_positions) == sorted(res_nums.tolist())
    assert len(set(all_test_positions)) == len(res_nums)

    # Invariant: Every mutation appears in test partition of exactly one fold
    assert sorted(all_test_mutation_indices) == list(range(len(dms_data)))
    assert len(set(all_test_mutation_indices)) == len(dms_data)


def test_serialization_and_cache_integrity():
    """Verify JSON serialization round-trip and cached files on disk."""
    split = PositionDisjointSplit(
        train_positions=[43, 44, 45],
        calib_positions=[46],
        test_positions=[47],
        train_mutation_indices=[0, 1],
        calib_mutation_indices=[2],
        test_mutation_indices=[3],
    )
    d = split.to_dict()
    reconstructed = PositionDisjointSplit.from_dict(d)
    assert reconstructed.train_positions == split.train_positions
    assert reconstructed.calib_positions == split.calib_positions
    assert reconstructed.test_positions == split.test_positions
    assert reconstructed.train_mutation_indices == split.train_mutation_indices

    # Check cached canonical split on disk
    assert Path(SPLIT_CANONICAL).exists()
    with open(SPLIT_CANONICAL) as f:
        data_canon = json.load(f)
    canon_split = PositionDisjointSplit.from_dict(data_canon)
    assert canon_split.num_train_positions == 163
    assert canon_split.num_calib_positions == 32
    assert canon_split.num_test_positions == 33
    assert canon_split.num_train_mutations == 3062
    assert canon_split.num_calib_mutations == 597
    assert canon_split.num_test_mutations == 612

    # Check cached 5-fold split on disk
    assert Path(SPLIT_KFOLD).exists()
    with open(SPLIT_KFOLD) as f:
        data_kfold = json.load(f)
    assert len(data_kfold["folds"]) == 5
