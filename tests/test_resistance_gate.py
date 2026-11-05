"""Unit and integration tests for src/zdyngnn/csd/resistance_gate.py."""
import pytest

from zdyngnn.csd.resistance_gate import (
    CandidateScreenResult,
    ConformalResistanceGate,
    HospitalVariant,
)


def test_conformal_resistance_gate_catalog_loading():
    """Verify that clinical mutations from circulating superbug strains (NDM-1..NDM-77) are loaded."""
    gate = ConformalResistanceGate(
        catalog_path="data/processed/bldb_full_ndm_catalog.json",
        default_condition="0.25_MEM",
    )

    # 49 mature mutations cataloged across hospital strains
    assert gate.num_variants >= 40

    # Verify critical outbreak mutations are present
    var_muts = {v.mutation for v in gate.variants}
    assert "M154L" in var_muts  # Global dominant allosteric outbreak variant
    assert "M248L" in var_muts  # Active-site pocket loop substitution
    assert "A74T" in var_muts   # Loop L3 variant


def test_conformal_clinical_safety_decisions():
    """Verify one-sided conformal upper bound decisions against the -2.0 resistance threshold."""
    gate = ConformalResistanceGate(
        default_condition="0.25_MEM",
        custom_q_upper=1.8325,
        custom_q_two_sided=2.2100,
        resistance_threshold=-2.0,
    )

    # Case 1: Potent lead with certified safety margin
    # mu = -5.5, sigma = 0.5 -> U_upper = -5.5 + 1.8325 * 0.5 = -4.584 <= -2.0 -> Certified Safe
    res_safe = gate.evaluate_candidate(
        candidate_id="LEAD_CERTIFIED_SAFE",
        smiles="c1ccccc1C(=O)NC(CS)C(=O)O",
        base_affinity=-5.5,
        base_sigma=0.5,
    )
    assert res_safe.passed_gate is True
    assert res_safe.clinical_call == "Certified Safe"
    assert res_safe.max_u_upper <= -2.0
    assert res_safe.min_safety_margin > 0.0

    # Case 2: Inactive or resistant candidate
    # mu = -0.5, sigma = 0.4 -> U_upper = -0.5 + 1.8325 * 0.4 = +0.233 > -2.0 -> Failed gate
    res_resist = gate.evaluate_candidate(
        candidate_id="LEAD_RESISTANT_ESCAPE",
        smiles="CCNCC",
        base_affinity=-0.5,
        base_sigma=0.4,
    )
    assert res_resist.passed_gate is False
    assert res_resist.clinical_call == "Resistant Escape Risk"
    assert res_resist.max_u_upper > -2.0
    assert res_resist.min_safety_margin < 0.0

    # Case 3: Ambiguous / borderline candidate
    # mu = -2.5, sigma = 0.8 -> U_upper = -2.5 + 1.8325 * 0.8 = -1.034 > -2.0, lower = -4.268 < -2.0
    res_ambig = gate.evaluate_candidate(
        candidate_id="LEAD_AMBIGUOUS_BORDERLINE",
        smiles="c1ccccc1O",
        base_affinity=-2.5,
        base_sigma=0.8,
    )
    assert res_ambig.passed_gate is False
    assert res_ambig.clinical_call == "Ambiguous / Borderline"


def test_batch_screening_and_filtering():
    """Verify batch filtering of candidate libraries through the resistance gate."""
    gate = ConformalResistanceGate(default_condition="0.25_MEM")

    candidates = [
        {"synthon_assembly_id": "CAND_01_POTENT", "smiles": "SCC(=O)N1CCCC1", "mu_ensemble": -6.0, "sigma_ensemble": 0.3},
        {"synthon_assembly_id": "CAND_02_BORDERLINE", "smiles": "c1ccccc1N", "mu_ensemble": -2.2, "sigma_ensemble": 0.6},
        {"synthon_assembly_id": "CAND_03_WEAK", "smiles": "c1ccncc1", "mu_ensemble": -0.8, "sigma_ensemble": 0.5},
    ]

    passed, summary = gate.filter_candidates(candidates)

    assert len(passed) == 1
    assert passed[0].candidate_id == "CAND_01_POTENT"
    assert summary["total_candidates_screened"] == 3
    assert summary["passed_gate_count"] == 1
    assert pytest.approx(summary["pass_rate"], 1e-3) == 0.333
