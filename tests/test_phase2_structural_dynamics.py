"""Unit and integration tests for Phase II Multi-Conformational Structural Dynamics."""
import json
from pathlib import Path

import numpy as np
import pytest

MANIFEST_PATH = Path("results/phase2_structural_dynamics_manifest.json")
CANONICAL_3SPU = Path("data/processed/dynamics_3spu.npz")
CACHE_4HL2 = Path("data/processed/dynamics_4hl2.npz")
CACHE_4RL0 = Path("data/processed/dynamics_4rl0.npz")
CACHE_5ZGY = Path("data/processed/dynamics_5zgy.npz")
CACHE_5O2E = Path("data/processed/dynamics_5o2e.npz")


def test_phase2_artifacts_exist():
    """Verify that Phase II manifest and all parallel caches exist."""
    assert MANIFEST_PATH.exists(), "Run scripts/phase2_structural_dynamics.py first."
    assert CANONICAL_3SPU.exists(), "Canonical 3SPU cache missing!"
    assert CACHE_4HL2.exists(), "Run scripts/phase2_structural_dynamics.py first."
    assert CACHE_4RL0.exists(), "Run scripts/phase2_structural_dynamics.py first."
    assert CACHE_5ZGY.exists(), "Run scripts/phase2_structural_dynamics.py first."
    assert CACHE_5O2E.exists(), "Run scripts/phase2_structural_dynamics.py first."

    with open(MANIFEST_PATH) as f:
        data = json.load(f)
    assert data["phase"] == "PHASE II: MULTI-CONFORMATIONAL STRUCTURAL DYNAMICS ENGINE"
    assert data["canonical_reference"]["pdb_id"] == "3SPU"
    assert data["canonical_reference"]["num_residues"] == 228


def test_canonical_3spu_locked_integrity():
    """Verify that dynamics_3spu.npz was NOT overwritten or corrupted."""
    d3 = np.load(CANONICAL_3SPU)
    assert len(d3["residue_numbers"]) == 228
    assert d3["dci_asym"].shape == (228, 228)
    # Assert exact antisymmetry
    assert np.allclose(d3["dci_asym"], -d3["dci_asym"].T, atol=1e-8)
    assert np.all(np.diag(d3["dci_asym"]) == 0.0)


def test_m154_driver_invariant_universal():
    """Verify that M154 remains an allosteric driver (>0) across all 4 NDM-1 conformations."""
    with open(MANIFEST_PATH) as f:
        data = json.load(f)

    # 1. Check 3SPU reference couplings
    ref_couplings = data["canonical_reference"]["m154_driver_coupling_to_zinc_pocket"]
    for lig, c_val in ref_couplings.items():
        assert c_val > 0.0, f"3SPU M154 coupling to {lig} is not positive ({c_val})"

    # 2. Check all comparative structures
    shifts = data["conformational_shifts"]
    for pdb_id in ["4HL2", "4RL0", "5ZGY", "5O2E"]:
        m154_data = shifts[pdb_id]["m154_driver_coupling_to_zinc_pocket"]
        for lig, c_val in m154_data.items():
            assert c_val > 0.0, f"{pdb_id} M154 coupling to {lig} is not positive ({c_val})"


def test_substrate_induced_loop_clamping():
    """Verify that substrate binding in 4HL2 and 4RL0 rigidifies catalytic loop L8."""
    with open(MANIFEST_PATH) as f:
        data = json.load(f)

    shifts = data["conformational_shifts"]
    # Loop L8 delta MSF must be negative (clamping / rigidification)
    assert shifts["4HL2"]["loop_l8_msf_shift"]["delta"] < 0.0
    assert shifts["4RL0"]["loop_l8_msf_shift"]["delta"] < 0.0
    assert shifts["5O2E"]["loop_l8_msf_shift"]["delta"] < 0.0
