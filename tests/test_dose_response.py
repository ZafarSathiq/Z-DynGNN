"""Unit and integration tests for Step 8 Multi-Concentration Pharmacodynamics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SUMMARY_PATH = Path("results/dose_response_monotonicity_summary.json")
CERTS_PATH = Path("results/dose_response_certificates_9cond.csv")
TRAJ_PATH = Path("results/dose_response_clinical_trajectories.csv")


def test_step8_artifacts_exist():
    """Verify that Step 8 artifacts exist on disk and are non-empty."""
    assert SUMMARY_PATH.exists(), "Run scripts/evaluate_dose_response.py first."
    assert CERTS_PATH.exists(), "Run scripts/evaluate_dose_response.py first."
    assert TRAJ_PATH.exists(), "Run scripts/evaluate_dose_response.py first."

    with open(SUMMARY_PATH) as f:
        data = json.load(f)
    assert "Ampicillin" in data
    assert "Cefotaxime" in data
    assert "Meropenem" in data


def test_dose_response_coverage_invariants():
    """Verify that all 9 conditions meet nominal coverage guarantees (>= 94%)."""
    with open(SUMMARY_PATH) as f:
        data = json.load(f)

    for drug, res in data.items():
        conds = res["conditions_evaluation"]
        assert len(conds) == 3
        for c_name, c_data in conds.items():
            cov_two = c_data["empirical_coverage_two_sided"]
            cov_up = c_data["empirical_coverage_one_sided_upper"]
            # Coverage must meet or exceed 94% (nominal 95% minus finite-sample tolerance)
            assert cov_two >= 0.94, f"{drug} {c_name} two-sided coverage {cov_two} < 0.94"
            assert cov_up >= 0.97, f"{drug} {c_name} one-sided coverage {cov_up} < 0.97"
            assert c_data["pure_test_containment_n9_two_sided"] == 1.0
            assert c_data["pure_test_containment_n9_one_sided"] == 1.0
            assert c_data["false_safety_count"] == 0


def test_dose_response_biological_monotonicity():
    """Verify that biological monotonicity is preserved within replicate noise."""
    with open(SUMMARY_PATH) as f:
        data = json.load(f)

    # Low to mid dose monotonicity must be >= 98% across all drugs
    for drug in ["Ampicillin", "Cefotaxime", "Meropenem"]:
        step1_mono = data[drug]["step1_low_to_mid_monotonicity"]
        assert step1_mono >= 0.98, f"{drug} low->mid monotonicity {step1_mono} < 0.98"


def test_clinical_trajectories_integrity():
    """Verify clinical trajectories table has exactly 198 records (22 variants x 9 conditions)."""
    df = pd.read_csv(TRAJ_PATH)
    assert len(df) == 22 * 9  # 198 rows
    assert set(df["drug"].unique()) == {"Ampicillin", "Cefotaxime", "Meropenem"}
    assert len(df["condition"].unique()) == 9
