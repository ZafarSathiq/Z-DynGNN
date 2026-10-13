"""Unit and integration tests for Upgrade C: 3D-Equivariant Physics GNN (ZDynEGNN)."""
import torch
import torch.nn as nn
import pytest

from zdyngnn.models.gnn import (
    EquivariantDynamicsMessagePassingLayer,
    ZDynEGNN,
    gaussian_nll_loss,
)


def test_equivariant_layer_e3_equivariance():
    """Verify that EquivariantDynamicsMessagePassingLayer is strictly E(3) equivariant."""
    torch.manual_seed(42)
    n_nodes = 8
    n_edges = 12
    dim = 32

    layer = EquivariantDynamicsMessagePassingLayer(node_dim=dim, edge_dim=16)
    layer.eval()

    coords = torch.randn(n_nodes, 3)
    h = torch.randn(n_nodes, dim)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, 16)
    dci = torch.randn(n_edges, 1)

    # 1. 3D Rotation Matrix R in SO(3)
    theta = 0.785398  # 45 degrees
    rot = torch.tensor([
        [torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta)), 0.0],
        [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta)), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    # 2. 3D Translation Vector t in R^3
    trans = torch.tensor([5.0, -3.0, 2.0], dtype=torch.float32)

    coords_transformed = (coords @ rot.T) + trans

    with torch.no_grad():
        h1, c1 = layer(h, coords, edge_index, edge_attr, dci)
        h2, c2 = layer(h, coords_transformed, edge_index, edge_attr, dci)

    # Invariant: h must be identical
    diff_h = (h1 - h2).abs().max().item()
    assert diff_h < 1e-5, f"h is not invariant to SE(3) transformation! Diff={diff_h}"

    # Equivariant: c2 must equal R @ c1 + t
    c1_transformed = (c1 @ rot.T) + trans
    diff_c = (c2 - c1_transformed).abs().max().item()
    assert diff_c < 1e-5, f"Coordinates are not equivariant to SE(3) transformation! Diff={diff_c}"


def test_zdyn_egnn_forward_and_backward():
    """Verify forward and backward gradients through ZDynEGNN with Gaussian NLL."""
    torch.manual_seed(42)
    n_nodes = 15
    n_edges = 20
    model = ZDynEGNN(node_in_dim=34, hidden_dim=32, num_layers=2)

    x = torch.randn(n_nodes, 34)
    coords = torch.randn(n_nodes, 3)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_dist = torch.rand(n_edges) * 7.5 + 0.5
    dci_edge = torch.randn(n_edges) * 0.5

    # Forward
    mu, sigma = model(x, coords, edge_index, edge_dist=edge_dist, dci_edge=dci_edge)
    assert mu.shape == (n_nodes, 1)
    assert sigma.shape == (n_nodes, 1)
    assert (sigma > 0).all()

    # Backward gradient check
    y_target = torch.randn(n_nodes, 1)
    loss = gaussian_nll_loss(y_target, mu, sigma)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad and "mutation_mean_head" not in name and "mutation_logvar_head" not in name:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient contains NaN!"


def test_zdyn_egnn_mutation_level_forward():
    """Verify mutation-specific forward pass with WT and mutant amino acid indices."""
    torch.manual_seed(42)
    n_nodes = 20
    n_mutations = 5
    model = ZDynEGNN(node_in_dim=34, hidden_dim=32, num_layers=2)

    x = torch.randn(n_nodes, 34)
    coords = torch.randn(n_nodes, 3)
    edge_index = torch.randint(0, n_nodes, (2, 30))

    mut_pos = torch.tensor([0, 2, 5, 10, 19], dtype=torch.long)
    wt_idx = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    mut_idx = torch.tensor([19, 18, 17, 16, 15], dtype=torch.long)

    mu, sigma = model(
        x, coords, edge_index,
        mut_positions=mut_pos, mut_wt_idx=wt_idx, mut_aa_idx=mut_idx,
    )
    assert mu.shape == (n_mutations, 1)
    assert sigma.shape == (n_mutations, 1)
    assert (sigma > 0).all()
