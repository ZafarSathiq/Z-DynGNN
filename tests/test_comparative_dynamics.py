"""Unit and integration tests for Step 11 Multi-Conformational Structural Shift Audit."""
import json
from pathlib import Path

import numpy as np
import pytest

SUMMARY_PATH = Path("results/conformational_shift_audit_summary.json")
CACHE_4HL2 = Path("data/processed/dynamics_4hl2.npz")
CACHE_4RL0 = Path("data/processed/dynamics_4rl0.npz")
CACHE_5ZGY = Path("data/processed/dynamics_5zgy.npz")
CACHE_5O2E = Path("data/processed/dynamics_5o2e.npz")
CANONICAL_3SPU = Path("data/processed/dynamics_3spu.npz")


def test_comparative_caches_exist():
    """Verify that all comparative dynamics caches exist and canonical 3SPU is untouched."""
    assert CANONICAL_3SPU.exists(), "Canonical 3SPU cache missing!"
    assert CACHE_4HL2.exists(), "Run scripts/phase2_structural_dynamics.py first."
    assert CACHE_4RL0.exists(), "Run scripts/phase2_structural_dynamics.py first."
    assert CACHE_5ZGY.exists(), "Run scripts/phase2_structural_dynamics.py first."
    assert CACHE_5O2E.exists(), "Run scripts/phase2_structural_dynamics.py first."
    assert SUMMARY_PATH.exists(), "Run scripts/phase2_structural_dynamics.py first."


def test_canonical_3spu_isolation_invariant():
    """Verify that canonical 3SPU cache was NOT corrupted or overwritten."""
    d3 = np.load(CANONICAL_3SPU)
    assert len(d3["residue_numbers"]) == 228
    assert d3["dci_asym"].shape == (228, 228)
    # Verify strict antisymmetry
    assert np.allclose(d3["dci_asym"], -d3["dci_asym"].T, atol=1e-8)


def test_comparative_dynamics_antisymmetry():
    """Verify that DCI_asym is strictly antisymmetric across all comparative structures."""
    for path in [CACHE_4HL2, CACHE_4RL0, CACHE_5ZGY, CACHE_5O2E]:
        d = np.load(path)
        assert len(d["residue_numbers"]) == 228
        assert d["dci_asym"].shape == (228, 228)
        assert np.allclose(d["dci_asym"], -d["dci_asym"].T, atol=1e-8)
        assert np.all(np.diag(d["dci_asym"]) == 0.0)


def test_m154_allosteric_driver_conservation():
    """Verify that M154 remains an allosteric driver (>0) across all NDM-1 conformations."""
    with open(SUMMARY_PATH) as f:
        data = json.load(f)

    for pdb_id in ["4HL2", "4RL0", "5ZGY", "5O2E"]:
        shifts = data["conformational_shifts"][pdb_id]
        m154_couplings = shifts["m154_driver_coupling_to_zinc_pocket"]
        # All 6 zinc coordinating residues must have positive driver coupling from M154
        for lig, c_val in m154_couplings.items():
            assert c_val > 0.0, f"M154 driver coupling to {lig} in {pdb_id} is not positive ({c_val})"


def test_ca_rmsd_structural_integrity():
    """Verify that C-alpha RMSD relative to 3SPU is within valid sub-angstrom limits."""
    with open(SUMMARY_PATH) as f:
        data = json.load(f)

    shifts = data["conformational_shifts"]
    assert shifts["4HL2"]["ca_rmsd_vs_3spu_angstroms"] < 1.0
    assert shifts["4RL0"]["ca_rmsd_vs_3spu_angstroms"] < 1.0
    assert shifts["5ZGY"]["ca_rmsd_vs_3spu_angstroms"] < 1.0
    assert shifts["5O2E"]["ca_rmsd_vs_3spu_angstroms"] < 1.0
