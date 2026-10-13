"""Unit and integration tests for Step 9 EC50 Microdilution & Combinatorial Epistasis."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SUMMARY_PATH = Path("results/epistasis_ec50_validation_summary.json")
CERTS_PATH = Path("results/epistasis_combinatorial_certificates.csv")


def test_step9_artifacts_exist():
    """Verify that Step 9 artifacts exist on disk and are valid."""
    assert SUMMARY_PATH.exists(), "Run scripts/evaluate_epistasis_ec50.py first."
    assert CERTS_PATH.exists(), "Run scripts/evaluate_epistasis_ec50.py first."

    with open(SUMMARY_PATH) as f:
        data = json.load(f)
    assert data["dataset"] == "Chen et al. 2024 Supplementary Data 5 (EC50 Assays)"
    assert data["overall_physical_ec50_containment"]["total_evaluated"] == 21


def test_single_mutants_containment_invariants():
    """Verify that all single mutants in SD5 are 100% contained."""
    with open(SUMMARY_PATH) as f:
        data = json.load(f)

    singles = data["single_mutants_n12"]
    assert singles["total_count"] == 12
    assert singles["containment_two_sided"] == 1.0
    assert singles["containment_one_sided"] == 1.0
    assert singles["pure_test_n2"]["containment_two_sided"] == 1.0
    assert singles["pure_test_n2"]["containment_one_sided"] == 1.0
    assert set(singles["pure_test_n2"]["mutations"]) == {"G197Y", "I259L"}


def test_combinatorial_multi_mutants_containment_invariants():
    """Verify that all 9 combinatorial multi-mutants are 100% contained."""
    with open(SUMMARY_PATH) as f:
        data = json.load(f)

    combs = data["combinatorial_multi_mutants_n9"]
    assert combs["total_count"] == 9
    assert combs["containment_two_sided"] == 1.0
    assert combs["containment_one_sided"] == 1.0

    # Verify that both synergistic rescue and negative epistasis events are detected
    epi_events = combs["epistasis_events"]
    assert len(epi_events) == 9
    interps = [e["interpretation"] for e in epi_events]
    assert "Synergistic Buffering" in interps
    assert "Deleterious Interference" in interps


def test_certificates_csv_schema():
    """Verify the columns and types in epistasis_combinatorial_certificates.csv."""
    df = pd.read_csv(CERTS_PATH)
    assert len(df) == 21
    assert "mean_ec50_ug_ml" in df.columns
    assert "log2_fold_ec50_vs_wt" in df.columns
    assert "expected_additive_ec50" in df.columns
    assert "epistasis_delta" in df.columns
    assert "covered_two_sided" in df.columns
    assert "covered_one_sided" in df.columns
    assert df["covered_two_sided"].all()
    assert df["covered_one_sided"].all()
