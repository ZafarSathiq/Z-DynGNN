"""Unit and integration tests for Step 10 Complete BLDB Clinical Variant Expansion."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CATALOG_PATH = Path("data/processed/bldb_full_ndm_catalog.json")
SUMMARY_PATH = Path("results/bldb_full_validation_summary.json")
CERTS_PATH = Path("results/bldb_full_clinical_certificates.csv")


def test_step10_artifacts_exist():
    """Verify that Step 10 full BLDB artifacts exist and are non-empty."""
    assert CATALOG_PATH.exists(), "Run scripts/fetch_bldb_catalog.py first."
    assert SUMMARY_PATH.exists(), "Run scripts/evaluate_full_bldb.py first."
    assert CERTS_PATH.exists(), "Run scripts/evaluate_full_bldb.py first."

    with open(SUMMARY_PATH) as f:
        data = json.load(f)
    assert data["total_variants_cataloged"] >= 70
    assert data["total_mature_substitutions_evaluated"] == 48
    assert data["total_evaluations_across_conditions"] == 48 * 3  # 144 evaluations


def test_pure_test_containment_invariants():
    """Verify 100% containment on pure held-out TEST hospital variants (n=7)."""
    with open(SUMMARY_PATH) as f:
        data = json.load(f)

    for cond, c_data in data["condition_summaries"].items():
        test_info = c_data["pure_test_n7"]
        assert test_info["count"] == 7
        assert test_info["containment_two_sided"] == 1.0, f"Pure test 2-sided containment failed on {cond}"
        assert test_info["containment_one_sided"] == 1.0, f"Pure test 1-sided upper coverage failed on {cond}"
        assert c_data["total_false_safety_errors"] == 0, f"False safety errors detected on {cond}"


def test_partition_isolation_and_no_leakage():
    """Verify that Bucket 3 (TRAIN_CONTROL) variants are strictly segregated from TEST."""
    df = pd.read_csv(CERTS_PATH)
    test_muts = set(df[df["partition"] == "TEST"]["mutation"].unique())
    train_muts = set(df[df["partition"] == "TRAIN_CONTROL"]["mutation"].unique())
    calib_muts = set(df[df["partition"] == "CALIB"]["mutation"].unique())

    # Assert mutually exclusive partitions (zero data leakage)
    assert len(test_muts.intersection(train_muts)) == 0, "Test and Train partitions intersect!"
    assert len(test_muts.intersection(calib_muts)) == 0, "Test and Calib partitions intersect!"
    assert len(calib_muts.intersection(train_muts)) == 0, "Calib and Train partitions intersect!"
    assert len(test_muts) == 7
    assert len(calib_muts) == 11
    assert len(train_muts) == 30


def test_patient_safety_zero_critical_errors():
    """Verify that zero resistant hospital variants receive a certified safe call."""
    df = pd.read_csv(CERTS_PATH)
    resistant_mask = df["fitness_true"] >= -2.0
    safe_calls = df["clinical_call_one_sided"] == "Certified Safe / Non-Resistant"
    critical_errors = df[resistant_mask & safe_calls]
    assert len(critical_errors) == 0, f"Critical false safety errors found: {critical_errors}"
