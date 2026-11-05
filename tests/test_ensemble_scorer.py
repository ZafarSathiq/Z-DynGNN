"""Unit and integration tests for src/zdyngnn/csd/ensemble_scorer.py."""
import numpy as np
import pytest
from rdkit import Chem
import torch

from zdyngnn.csd.ensemble_scorer import (
    GaussianRBF,
    HeterogeneousComplexGraph,
    HeterogeneousEnsembleScorer,
    MultiConformalEnsembleScorer,
    build_complex_graph,
    extract_ligand_features,
)


def test_extract_ligand_features():
    """Verify that 32-dim atomic features and 3D coordinates are cleanly extracted."""
    mol = Chem.MolFromSmiles("c1ccccc1C(=O)NCCSC")
    assert mol is not None

    x_lig, coords_lig = extract_ligand_features(mol)

    num_heavy = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() != "H")
    assert x_lig.shape == (num_heavy, 32)
    assert coords_lig.shape == (num_heavy, 3)

    # Invariants: no NaN or Inf in features or coordinates
    assert not torch.isnan(x_lig).any()
    assert not torch.isinf(x_lig).any()
    assert not torch.isnan(coords_lig).any()
    assert not torch.isinf(coords_lig).any()


def test_gaussian_rbf_expansion():
    """Verify continuous 16-center Gaussian RBF distance expansion."""
    rbf = GaussianRBF(num_centers=16, d_min=0.0, d_max=6.0)

    # Test distance values
    dists = torch.tensor([0.0, 1.5, 3.0, 4.5, 6.0], dtype=torch.float32)
    expanded = rbf(dists)

    assert expanded.shape == (5, 16)
    assert (expanded >= 0.0).all() and (expanded <= 1.0).all()
    # At center 0.0, the first center should have value close to 1.0
    assert expanded[0, 0] > 0.95


def test_heterogeneous_scorer_forward_and_backward():
    """Verify bipartite message passing forward pass and gradient flow."""
    torch.manual_seed(42)
    scorer = HeterogeneousEnsembleScorer(prot_dim=34, lig_dim=32, hidden_dim=32)

    # Synthetic protein (10 residues) and synthetic ligand (5 atoms)
    n_prot = 10
    n_lig = 5
    x_prot = torch.randn(n_prot, 34)
    coords_prot = torch.randn(n_prot, 3) * 5.0
    x_lig = torch.randn(n_lig, 32)
    coords_lig = torch.randn(n_lig, 3) * 5.0

    graph = build_complex_graph(x_prot, coords_prot, x_lig, coords_lig, cutoff_distance=8.0)

    assert graph.edge_index_inter.shape[0] == 2
    assert graph.edge_rbf_inter.shape[1] == 16

    mu, sigma = scorer(graph)

    # Invariants: scalar outputs, positive sigma
    assert mu.dim() == 0
    assert sigma.dim() == 0
    assert sigma > 0.0

    # Gradient check via Gaussian NLL
    loss = 0.5 * torch.log(sigma ** 2) + 0.5 * ((0.0 - mu) ** 2) / (sigma ** 2)
    loss.backward()

    # Ensure gradients propagated to projection and message passing layers
    assert scorer.prot_proj[0].weight.grad is not None
    assert scorer.msg_p2l[0].weight.grad is not None
    assert scorer.mean_head[-1].weight.grad is not None


def test_multi_conformal_ensemble_scorer_end_to_end():
    """Verify multi-conformal scoring against real Apo (3SPU) and Holo (4RL0) crystal states."""
    scorer = MultiConformalEnsembleScorer(
        apo_dynamics_path="data/processed/dynamics_3spu.npz",
        holo_dynamics_path="data/processed/dynamics_4rl0.npz",
        structure_path="data/processed/structure_3spu_ca.npz",
    )

    # Test candidate lead (thiol-functionalized beta-lactamase inhibitor)
    mol = Chem.MolFromSmiles("c1ccccc1C(=O)NC(CS)C(=O)O")
    assert mol is not None

    # Score candidate with active site alignment
    # Active site centroid in NDM-1: approx [39.0, 27.0, 37.0]
    pocket_center = np.array([39.09, 27.05, 37.42], dtype=np.float32)
    scores = scorer.score_molecule(mol, pocket_centroid=pocket_center)

    assert "mu_apo" in scores
    assert "mu_holo" in scores
    assert "mu_ensemble" in scores
    assert "sigma_ensemble" in scores
    assert "conformation_shift" in scores

    assert scores["sigma_ensemble"] > 0.0
    assert np.isfinite(scores["mu_ensemble"])
    assert np.isfinite(scores["conformation_shift"])
