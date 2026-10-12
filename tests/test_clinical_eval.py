"""Unit and integration tests for Step 7 Clinical Validation and Resistance Decisions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zdyngnn.eval.clinical import (
    BLDB_CLINICAL_MUTATIONS,
    ClinicalDecisionMetrics,
    classify_clinical_decision,
    evaluate_clinical_decision_metrics,
    extract_bldb_variant_evaluation,
)


def test_bldb_catalog_integrity():
    """Verify that the BLDB clinical variant catalog is complete and structurally valid."""
    assert len(BLDB_CLINICAL_MUTATIONS) == 22
    for item in BLDB_CLINICAL_MUTATIONS:
        assert "mutation" in item
        assert "pos" in item
        assert "wt" in item
        assert "mut" in item
        assert "source" in item or "variants" in item
        assert "phenotype" in item
        assert 43 <= item["pos"] <= 270  # Within resolved 3SPU crystal structure
        assert item["mutation"] == f"{item['wt']}{item['pos']}{item['mut']}"


def test_classify_clinical_decision_logic():
    """Verify categorical decision logic against clinical resistance thresholds."""
    # 1. Certified Resistant (lower bound >= -2.0)
    assert classify_clinical_decision(lower=-1.5, upper=2.0, mu=0.5) == "Certified Resistant"
    assert classify_clinical_decision(lower=-2.0, upper=1.0, mu=-0.5) == "Certified Resistant"

    # 2. Certified Susceptible (upper bound < -5.0)
    assert classify_clinical_decision(lower=-10.0, upper=-5.5, mu=-7.0) == "Certified Susceptible"

    # 3. Ambiguous / Triage (spans threshold)
    assert classify_clinical_decision(lower=-4.0, upper=1.0, mu=-1.5) == "Ambiguous (Triage for Lab)"
    assert classify_clinical_decision(lower=-8.0, upper=-3.0, mu=-5.5) == "Ambiguous (Triage for Lab)"


def test_evaluate_clinical_decision_metrics():
    """Verify sensitivity, specificity, and safety error metrics."""
    y_true = np.array([0.5, 0.2, -6.0, -7.0, -1.0])
    mu = np.array([0.4, 0.3, -6.2, -6.8, -1.2])
    lower = np.array([-1.0, -2.5, -9.0, -10.0, -3.0])
    upper = np.array([1.8, 3.1, -3.4, -4.6, 0.6])

    metrics = evaluate_clinical_decision_metrics(
        y_true=y_true,
        lower=lower,
        upper=upper,
        mu=mu,
        condition="test",
    )

    assert metrics.num_samples == 5
    assert metrics.containment_rate == 1.0  # All y_true lie in [lower, upper]
    assert metrics.false_safety_count == 0  # No true resistant called susceptible


def test_extract_bldb_variant_evaluation():
    """Verify filtering and certificate construction on clinical mutations."""
    mock_df = pd.DataFrame({
        "mutation": ["M154L", "A74T", "E170K", "XYZ123"],
        "pos": [154, 74, 170, 50],
        "wt": ["M", "A", "E", "X"],
        "mut": ["L", "T", "K", "Z"],
        "fitness_score": [0.48, -0.32, -0.17, -5.0],
        "pred_mu": [0.40, -0.40, -0.20, -5.1],
        "pred_sigma": [1.0, 1.2, 0.8, 2.0],
    })

    bldb_df, stats = extract_bldb_variant_evaluation(mock_df, condition="test", quantile=2.0)
    assert len(bldb_df) == 3  # Matches 3 real BLDB variants, drops XYZ123
    assert stats["containment_rate"] == 1.0
    assert "variants" in bldb_df.columns
    assert "phenotype" in bldb_df.columns


def test_clinical_artifacts_on_disk():
    """Verify generated Step 7 clinical certificates and profile files."""
    bldb_csv = Path("results/bldb_variant_certificates.csv")
    m154l_json = Path("results/m154l_clinical_profile.json")
    dec_csv = Path("results/clinical_decision_metrics.csv")
    sum_json = Path("results/clinical_validation_summary.json")

    assert bldb_csv.exists()
    assert m154l_json.exists()
    assert dec_csv.exists()
    assert sum_json.exists()

    df_bldb = pd.read_csv(bldb_csv)
    assert len(df_bldb) == 66  # 22 unexposed clinical mutations * 3 conditions
    assert (df_bldb["partition"] != "TRAIN").all(), "Zero training mutations allowed in validation table."
    assert df_bldb["covered"].mean() >= 0.95, "At least 95% of unexposed BLDB variants must be contained."
    with open(m154l_json) as f:
        m154l_data = json.load(f)
    assert m154l_data["variant"] == "M154L"
    assert "allosteric_coupling_to_zinc_catalytic_pocket" in m154l_data
    assert "His120" in m154l_data["allosteric_coupling_to_zinc_catalytic_pocket"]
