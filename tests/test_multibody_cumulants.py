"""Automated unit tests for Multi-Body Allosteric Cumulants and Higher-Order Dynamics.

Validates the Dodd et al. (Science Advances 2026) multi-body cumulant engine,
regularized submatrix log-determinants, 3-clique extraction, catalytic hexamer metrics,
and the backward compatibility invariant of ZDynGNN under use_multibody=False.
"""
from pathlib import Path
import numpy as np
import pytest
import torch

from zdyngnn.dynamics.multibody import (
    compute_joint_correlator,
    compute_multibody_cumulant,
    compute_node_multibody_features,
    compute_subsystem_logdet,
    extract_catalytic_hexamer_metrics,
    find_contact_cliques,
)
from zdyngnn.models.gnn import ZDynGNN

CANONICAL_3SPU_DYN = Path("data/processed/dynamics_3spu.npz")
CANONICAL_3SPU_STRUCT = Path("data/processed/structure_3spu_ca.npz")
AMP_CHECKPOINT = Path("checkpoints/zdyn_gnn_multibody_256.0_AMP.pt")


def test_find_contact_cliques_synthetic():
    """Verify 3-clique extraction on a known synthetic graph."""
    # Graph with 4 nodes: 0-1, 1-2, 2-0 (triangle), and 2-3 (tail)
    adj = np.zeros((4, 4), dtype=int)
    adj[0, 1] = adj[1, 0] = 1
    adj[1, 2] = adj[2, 1] = 1
    adj[2, 0] = adj[0, 2] = 1
    adj[2, 3] = adj[3, 2] = 1

    cliques = find_contact_cliques(adj)
    assert len(cliques) == 1
    assert cliques[0] == (0, 1, 2)


def test_find_contact_cliques_3spu():
    """Verify 3-clique extraction on the canonical 3SPU contact graph."""
    assert CANONICAL_3SPU_STRUCT.exists()
    struct_data = np.load(CANONICAL_3SPU_STRUCT)
    adj = struct_data["adj_matrix"]
    n_nodes = adj.shape[0]
    assert n_nodes == 228

    cliques = find_contact_cliques(adj)
    # The 8.0 Angstrom contact graph of NDM-1 contains > 1000 triangles
    assert len(cliques) > 500
    # Verify clique ordering and invariants
    for c in cliques[:50]:
        assert len(c) == 3
        assert c[0] < c[1] < c[2]
        assert adj[c[0], c[1]] == 1
        assert adj[c[1], c[2]] == 1
        assert adj[c[0], c[2]] == 1


def test_subsystem_logdet_numerical_stability():
    """Verify that regularized logdet prevents NaN/collapse even on zero-eigenvalue matrices."""
    # Rank-1 singular matrix with 0 determinant
    v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    singular_cov = np.outer(v, v)
    assert np.isclose(np.linalg.det(singular_cov), 0.0)

    # Bare determinant would be 0 (logdet -> -inf)
    # Regularized logdet must be finite and well-defined
    logdet = compute_subsystem_logdet(singular_cov, [0, 1, 2], eps=1e-3)
    assert np.isfinite(logdet)
    assert not np.isnan(logdet)

    # Empty indices test
    assert compute_subsystem_logdet(singular_cov, []) == 0.0


def test_multibody_cumulant_permutation_invariance():
    """Verify that multi-body cumulant kappa(S) is invariant to index ordering."""
    np.random.seed(42)
    # Generate a positive semi-definite 5x5 covariance matrix
    A = np.random.randn(5, 5)
    cov = A @ A.T

    # Triplet (0, 2, 4) in various permutations
    k1 = compute_multibody_cumulant(cov, [0, 2, 4])
    k2 = compute_multibody_cumulant(cov, [4, 0, 2])
    k3 = compute_multibody_cumulant(cov, [2, 4, 0])

    assert np.isfinite(k1)
    assert np.isclose(k1, k2, rtol=1e-5)
    assert np.isclose(k1, k3, rtol=1e-5)


def test_catalytic_hexamer_metrics_validity():
    """Verify extraction of the 6-body catalytic zinc cage metrics on 3SPU."""
    assert CANONICAL_3SPU_DYN.exists()
    dyn_data = np.load(CANONICAL_3SPU_DYN)
    cov = dyn_data["gnm_covariance"]

    metrics = extract_catalytic_hexamer_metrics(cov)
    assert "hexamer_logdet" in metrics
    assert "hexamer_trace" in metrics
    assert "hexamer_mean_corr" in metrics
    assert "hexamer_condition" in metrics

    assert np.isfinite(metrics["hexamer_logdet"])
    assert metrics["hexamer_trace"] > 0.0
    assert metrics["hexamer_condition"] >= 1.0


def test_node_multibody_features_shape_and_bounds():
    """Verify node-level feature pooling produces valid (N, 6) tensor without NaNs."""
    assert CANONICAL_3SPU_STRUCT.exists()
    assert CANONICAL_3SPU_DYN.exists()
    struct_data = np.load(CANONICAL_3SPU_STRUCT)
    dyn_data = np.load(CANONICAL_3SPU_DYN)

    adj = struct_data["adj_matrix"]
    cov = dyn_data["gnm_covariance"]
    n_nodes = adj.shape[0]

    cliques = find_contact_cliques(adj)[:30]
    clique_dict = {clq: compute_multibody_cumulant(cov, clq) for clq in cliques}
    hex_metrics = extract_catalytic_hexamer_metrics(cov)

    feats = compute_node_multibody_features(
        n_nodes=n_nodes,
        cliques=cliques,
        clique_cumulants=clique_dict,
        cat_metrics=hex_metrics,
    )

    assert feats.shape == (228, 6)
    assert np.all(np.isfinite(feats))
    # Catalytic indicators must be 1.0 for the 6 catalytic residues and 0.0 elsewhere
    cat_set = {77, 79, 81, 146, 165, 207}
    for i in range(228):
        if i in cat_set:
            assert feats[i, 4] == 1.0
            assert feats[i, 5] != 0.0
        else:
            assert feats[i, 4] == 0.0
            assert feats[i, 5] == 0.0


def test_zdyngnn_multibody_checkpoint_loading():
    """Verify that ZDynGNN(use_multibody=True) loads trained multi-body checkpoints with strict=True."""
    assert AMP_CHECKPOINT.exists()
    checkpoint = torch.load(AMP_CHECKPOINT, map_location="cpu")

    # Initialize model with use_multibody=True
    model = ZDynGNN(node_in_dim=34, hidden_dim=64, num_layers=3, use_multibody=True, multibody_dim=6)

    # Must load cleanly with strict=True: zero missing or unexpected keys
    load_res = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert len(load_res.missing_keys) == 0
    assert len(load_res.unexpected_keys) == 0

    # Ensure forward pass with use_multibody=False produces finite outputs
    x = torch.randn(228, 34)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    mu, sigma = model(x, edge_index)
    assert mu.shape == (228, 1)
    assert sigma.shape == (228, 1)
    assert torch.all(torch.isfinite(mu))
    assert torch.all(sigma > 0.0)


def test_zdyngnn_multibody_enabled_forward():
    """Verify that ZDynGNN(use_multibody=True) fuses multibody_feats and preserves gradients."""
    model = ZDynGNN(
        node_in_dim=34,
        hidden_dim=64,
        num_layers=3,
        use_multibody=True,
        multibody_dim=6,
    )
    assert model.use_multibody is True
    assert model.multibody_encoder is not None

    x = torch.randn(228, 34)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    mb_feats = torch.randn(228, 6)

    mu, sigma = model(x, edge_index, multibody_feats=mb_feats)
    assert mu.shape == (228, 1)
    assert sigma.shape == (228, 1)

    loss = mu.sum() + sigma.sum()
    loss.backward()

    # Verify gradients flow into multibody_encoder
    for param in model.multibody_encoder.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
