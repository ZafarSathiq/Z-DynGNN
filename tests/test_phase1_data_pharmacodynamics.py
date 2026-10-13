"""Unit and integration tests for Phase I Data Foundation & Pharmacodynamics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MANIFEST_PATH = Path("results/phase1_pharmacodynamics_manifest.json")
CURATED_CSV_PATH = Path("data/processed/phase1_dms_9cond_curated.csv")


def test_phase1_artifacts_exist():
    """Verify that Phase I artifacts exist on disk and are valid."""
    assert MANIFEST_PATH.exists(), "Run scripts/phase1_data_pharmacodynamics.py first."
    assert CURATED_CSV_PATH.exists(), "Run scripts/phase1_data_pharmacodynamics.py first."

    with open(MANIFEST_PATH) as f:
        data = json.load(f)
    assert data["phase"] == "PHASE I: DATA FOUNDATION & 9-DOSE PHARMACODYNAMICS"
    assert data["reference_structure"]["num_residues"] == 228
    assert data["dms_landscape"]["num_conditions"] == 9
    assert data["dms_landscape"]["total_measurements"] == 4271 * 9


def test_phase1_noise_ceilings_invariants():
    """Verify that all 9 conditions have valid positive noise ceilings and SNRs."""
    with open(MANIFEST_PATH) as f:
        data = json.load(f)

    ceilings = data["noise_ceilings"]
    assert len(ceilings) == 9
    for cond_name, c_data in ceilings.items():
        assert c_data["r_max"] > 0.90, f"r_max too low for {cond_name}: {c_data['r_max']}"
        assert c_data["r2_max"] > 0.80, f"r2_max too low for {cond_name}: {c_data['r2_max']}"
        assert c_data["snr"] > 5.0, f"SNR too low for {cond_name}: {c_data['snr']}"
        assert c_data["total_variance"] > c_data["replicate_error_variance"]


def test_phase1_dose_response_monotonicity_invariants():
    """Verify that low-to-mid dose monotonicity exceeds 98% across all three drug series."""
    with open(MANIFEST_PATH) as f:
        data = json.load(f)

    mono = data["dose_response_monotonicity"]
    assert "Ampicillin" in mono
    assert "Cefotaxime" in mono
    assert "Meropenem" in mono

    for drug in ["Ampicillin", "Cefotaxime", "Meropenem"]:
        low_to_mid = mono[drug]["low_to_mid_dose_compliance"]
        assert low_to_mid >= 0.98, f"{drug} low->mid compliance {low_to_mid} < 0.98"


def test_phase1_curated_csv_integrity():
    """Verify schema and non-emptiness of phase1_dms_9cond_curated.csv."""
    df = pd.read_csv(CURATED_CSV_PATH)
    assert len(df) == 4271
    assert "mutation" in df.columns
    assert "pos" in df.columns
    assert "wt" in df.columns
    assert "mut" in df.columns

    # Verify that all 9 score and sd columns are present and numeric
    for cond in ["32.0_AMP", "256.0_AMP", "512.0_AMP", "2.0_CTX", "16.0_CTX", "32.0_CTX", "0.0625_MEM", "0.25_MEM", "0.5_MEM"]:
        assert f"{cond}_score" in df.columns
        assert f"{cond}_score_sd" in df.columns
        assert not df[f"{cond}_score"].isna().all()
