"""Unit and integration tests for Phase III Spatial Modeling & Benchmarking."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MANIFEST_PATH = Path("results/phase3_modeling_manifest.json")
COMPARISON_CSV_PATH = Path("results/phase3_model_comparisons_summary.csv")


def test_phase3_artifacts_exist():
    """Verify that Phase III artifacts exist on disk and are valid."""
    assert MANIFEST_PATH.exists(), "Run scripts/phase3_spatial_modeling.py first."
    assert COMPARISON_CSV_PATH.exists(), "Run scripts/phase3_spatial_modeling.py first."

    with open(MANIFEST_PATH) as f:
        data = json.load(f)
    assert data["phase"] == "PHASE III: SPATIAL INDEPENDENCE & Z-DYNGNN MODELING"
    assert data["spatial_partition"]["train_positions"] == 163
    assert data["spatial_partition"]["calib_positions"] == 32
    assert data["spatial_partition"]["test_positions"] == 33
    assert data["spatial_partition"]["disjoint_verified"] is True


def test_site_mean_collapse_invariant():
    """Verify that Site-Mean baseline collapsed (R2 < 0 and Pearson is NaN) on unseen positions."""
    df = pd.read_csv(COMPARISON_CSV_PATH)
    sm_rows = df[df["model"] == "Site-Mean Baseline"]
    assert len(sm_rows) == 3

    for _, row in sm_rows.iterrows():
        assert row["r2_raw"] < 0.0, f"Site-Mean R2 was not negative on {row['condition']}: {row['r2_raw']}"
        assert np.isnan(row["pearson_raw"]), f"Site-Mean Pearson was not NaN on {row['condition']}"


def test_zdyn_gnn_noise_ceiling_capture_invariant():
    """Verify that Z-DynGNN captures >55% of the physical experimental noise ceiling."""
    df = pd.read_csv(COMPARISON_CSV_PATH)
    zdyn_rows = df[df["model"] == "Z-DynGNN (Ours)"]
    assert len(zdyn_rows) == 3

    for _, row in zdyn_rows.iterrows():
        assert row["pearson_normalized"] >= 0.55, f"Normalized Pearson < 0.55 on {row['condition']}"
        assert row["pearson_raw"] >= 0.57, f"Raw Pearson < 0.57 on {row['condition']}"
        assert row["r2_raw"] > 0.15, f"Raw R2 < 0.15 on {row['condition']}"

    # Specifically assert Ampicillin captures >= 64% of ceiling
    amp_row = zdyn_rows[zdyn_rows["condition"] == "256.0_AMP"].iloc[0]
    assert amp_row["pearson_normalized"] >= 0.64


def test_comparison_csv_schema():
    """Verify the columns and types in phase3_model_comparisons_summary.csv."""
    df = pd.read_csv(COMPARISON_CSV_PATH)
    assert len(df) == 9  # 3 conditions x 3 models
    assert set(df["condition"].unique()) == {"256.0_AMP", "16.0_CTX", "0.25_MEM"}
    assert set(df["model"].unique()) == {"Site-Mean Baseline", "Ridge Regression", "Z-DynGNN (Ours)"}
    assert "pearson_normalized" in df.columns
    assert "r2_normalized" in df.columns
    assert "rmse" in df.columns
