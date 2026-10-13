"""Unit and integration tests for Phase V Comprehensive Real-World Clinical & Wet-Lab Validation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MANIFEST_PATH = Path("results/phase5_validation_manifest.json")
CERTS_PATH = Path("results/phase5_clinical_and_ec50_certificates.csv")


def test_phase5_artifacts_exist():
    """Verify that Phase V artifacts exist on disk and are valid."""
    assert MANIFEST_PATH.exists(), "Run scripts/phase5_clinical_wetlab_validation.py first."
    assert CERTS_PATH.exists(), "Run scripts/phase5_clinical_wetlab_validation.py first."

    with open(MANIFEST_PATH) as f:
        data = json.load(f)
    assert data["phase"] == "PHASE V: COMPREHENSIVE REAL-WORLD CLINICAL & WET-LAB VALIDATION"
    assert "global_bldb_library_n48" in data
    assert "wet_lab_ec50_microdilutions_n21" in data


def test_phase5_pure_test_containment_invariants():
    """Verify 100% containment on pure held-out TEST hospital variants across all conditions."""
    with open(MANIFEST_PATH) as f:
        data = json.load(f)

    bldb_data = data["global_bldb_library_n48"]
    for cond in ["256.0_AMP", "16.0_CTX", "0.25_MEM"]:
        c_summary = bldb_data[cond]
        assert c_summary["pure_test_n7_two_sided_containment"] == 1.0, f"Pure test 2-sided failed on {cond}"
        assert c_summary["pure_test_n7_one_sided_upper_coverage"] == 1.0, f"Pure test 1-sided failed on {cond}"
        assert c_summary["false_safety_errors"] == 0, f"False safety detected on {cond}"


def test_phase5_wetlab_ec50_containment_invariants():
    """Verify 100% containment on physical microdilution EC50 assays and multi-mutants."""
    with open(MANIFEST_PATH) as f:
        data = json.load(f)

    ec50_data = data["wet_lab_ec50_microdilutions_n21"]
    assert ec50_data["total_evaluated"] == 21
    assert ec50_data["containment_two_sided"] == 1.0
    assert ec50_data["containment_one_sided"] == 1.0
    assert ec50_data["pure_test_singles_containment"] == 1.0
    assert ec50_data["combinatorial_multi_mutants_containment"] == 1.0
    assert ec50_data["triple_mutant_safe_certified"] is True


def test_phase5_certificates_dataframe_schema():
    """Verify columns and integrity of phase5_clinical_and_ec50_certificates.csv."""
    df = pd.read_csv(CERTS_PATH)
    assert len(df) == 48 * 3 + 21  # 144 BLDB evaluations + 21 EC50 evaluations = 165
    assert "source_benchmark" in df.columns
    assert "condition" in df.columns
    assert "mutation" in df.columns
    assert "partition" in df.columns
    assert "covered_two_sided" in df.columns
    assert "covered_one_sided" in df.columns

    # Verify zero false safety errors across the entire dataset
    resistant_mask = df["fitness_true"] >= -2.0
    safe_calls = df["clinical_call_one_sided"] == "Certified Safe / Non-Resistant"
    critical_errors = df[resistant_mask & safe_calls]
    assert len(critical_errors) == 0
