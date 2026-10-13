"""Unit and integration tests for Phase IV Conformal Calibration & Decoupled Decision Engine."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MANIFEST_PATH = Path("results/phase4_conformal_decision_manifest.json")
ABLATION_CSV_PATH = Path("results/phase4_conformal_comparative_ablation.csv")
DECISIONS_CSV_PATH = Path("results/phase4_decoupled_test_decisions.csv")


def test_phase4_artifacts_exist():
    """Verify that Phase IV artifacts exist on disk and are valid."""
    assert MANIFEST_PATH.exists(), "Run scripts/phase4_conformal_decision_engine.py first."
    assert ABLATION_CSV_PATH.exists(), "Run scripts/phase4_conformal_decision_engine.py first."
    assert DECISIONS_CSV_PATH.exists(), "Run scripts/phase4_conformal_decision_engine.py first."

    with open(MANIFEST_PATH) as f:
        data = json.load(f)
    assert data["phase"] == "PHASE IV: CONFORMAL CALIBRATION & DECOUPLED DECISION SUPPORT ENGINE"
    assert "256.0_AMP" in data["conditions_summary"]
    assert "16.0_CTX" in data["conditions_summary"]
    assert "0.25_MEM" in data["conditions_summary"]


def test_phase4_coverage_guarantees():
    """Verify that all conditions satisfy nominal coverage guarantees (>= 94%)."""
    with open(MANIFEST_PATH) as f:
        data = json.load(f)

    for cond, c_data in data["conditions_summary"].items():
        two_cov = c_data["two_sided"]["test_coverage"]
        up_cov = c_data["one_sided_upper"]["test_coverage"]
        assert two_cov >= 0.94, f"{cond} two-sided coverage {two_cov} < 0.94"
        assert up_cov >= 0.99, f"{cond} one-sided upper coverage {up_cov} < 0.99"
        assert c_data["decoupled_decisions"]["false_safety_count"] == 0


def test_phase4_decoupled_triage_invariants():
    """Verify that decoupled testing achieves significant triage rate on CTX and AMP."""
    with open(MANIFEST_PATH) as f:
        data = json.load(f)

    ctx_res = data["conditions_summary"]["16.0_CTX"]["decoupled_decisions"]
    amp_res = data["conditions_summary"]["256.0_AMP"]["decoupled_decisions"]

    # Cefotaxime must achieve > 50% decisive triage
    assert ctx_res["decisive_rate"] >= 0.50
    assert ctx_res["decisive_calls_total"] == 326

    # Ampicillin must achieve > 25% decisive triage
    assert amp_res["decisive_rate"] >= 0.25
    assert amp_res["decisive_calls_total"] == 164


def test_comparative_ablation_gnn_superiority():
    """Verify that Z-DynGNN achieves tighter certified margins than ZDynEGNN."""
    df = pd.read_csv(ABLATION_CSV_PATH)
    assert len(df) == 2

    gnn_row = df[df["model"].str.contains("Z-DynGNN")].iloc[0]
    egnn_row = df[df["model"].str.contains("ZDynEGNN")].iloc[0]

    # Z-DynGNN margin must be strictly narrower than ZDynEGNN
    assert gnn_row["mean_certified_margin_B"] < egnn_row["mean_certified_margin_B"]
    assert gnn_row["quantile_one_sided_upper"] < egnn_row["quantile_one_sided_upper"]
