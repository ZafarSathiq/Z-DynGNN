"""Unit and integration tests for Step 12 Decoupled Dual One-Sided Hypothesis Testing."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SUMMARY_PATH = Path("results/decoupled_testing_summary.json")
DECISIONS_PATH = Path("results/decoupled_testing_decisions.csv")


def test_step12_artifacts_exist():
    """Verify that Step 12 decoupled testing artifacts exist and are non-empty."""
    assert SUMMARY_PATH.exists(), "Run scripts/evaluate_decoupled_testing.py first."
    assert DECISIONS_PATH.exists(), "Run scripts/evaluate_decoupled_testing.py first."

    with open(SUMMARY_PATH) as f:
        data = json.load(f)
    assert "256.0_AMP" in data
    assert "16.0_CTX" in data
    assert "0.25_MEM" in data


def test_decisive_rate_invariants():
    """Verify that decoupled testing achieves significant triage reduction (>10%)."""
    with open(SUMMARY_PATH) as f:
        data = json.load(f)

    # Cefotaxime must achieve >50% decisive calls
    ctx_test = data["16.0_CTX"]["test_set_n612"]
    assert ctx_test["decisive_rate"] >= 0.50, f"CTX decisive rate {ctx_test['decisive_rate']} < 0.50"
    assert ctx_test["decisive_calls_total"] >= 300

    # Ampicillin must achieve >25% decisive calls
    amp_test = data["256.0_AMP"]["test_set_n612"]
    assert amp_test["decisive_rate"] >= 0.25, f"AMP decisive rate {amp_test['decisive_rate']} < 0.25"

    # Meropenem must achieve >10% decisive calls
    mem_test = data["0.25_MEM"]["test_set_n612"]
    assert mem_test["decisive_rate"] >= 0.10, f"MEM decisive rate {mem_test['decisive_rate']} < 0.10"


def test_zero_false_safety_errors():
    """Verify that zero resistant test mutations receive a false safety call."""
    with open(SUMMARY_PATH) as f:
        data = json.load(f)

    for cond, c_data in data.items():
        assert c_data["test_set_n612"]["false_safety_count"] == 0, f"False safety detected on {cond}"


def test_decisions_csv_schema():
    """Verify the columns and rows in decoupled_testing_decisions.csv."""
    df = pd.read_csv(DECISIONS_PATH)
    assert len(df) == 612 * 3  # 1836 rows
    assert "decoupled_clinical_call" in df.columns
    assert "is_decisive" in df.columns
    assert "lower_bound_one_sided" in df.columns
    assert "upper_bound_one_sided" in df.columns

    # Verify that decisive calls exist in the CSV
    decisive_rows = df[df["is_decisive"]]
    assert len(decisive_rows) > 500  # Total across all 3 conditions
