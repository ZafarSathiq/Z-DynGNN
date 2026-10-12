"""Unit and integration tests for Step 5 Z-DynGNN Modeling and Training."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from zdyngnn.models.baselines import RidgeBaseline, SiteMeanBaseline
from zdyngnn.models.dataset import (
    MutationDataset,
    ProteinGraphContext,
    collate_mutation_batch,
)
from zdyngnn.models.gnn import ZDynGNN, gaussian_nll_loss

STRUCT_CACHE = "data/processed/structure_3spu_ca.npz"
DYN_CACHE = "data/processed/dynamics_3spu.npz"


@pytest.fixture
def graph_context():
    assert Path(STRUCT_CACHE).exists() and Path(DYN_CACHE).exists()
    return ProteinGraphContext.from_processed_caches(STRUCT_CACHE, DYN_CACHE)


def test_zdyn_gnn_architecture_and_forward():
    """Verify ZDynGNN message passing layers, RBF expansion, and dual output heads."""
    model = ZDynGNN(
        node_in_dim=54,
        num_rbf=16,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
    )

    n_nodes = 20
    n_edges = 50
    x = torch.randn(n_nodes, 54)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_dist = torch.rand(n_edges) * 8.0
    dci_edge = (torch.rand(n_edges) - 0.5) * 2.0

    mu, sigma = model(x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge)

    assert mu.shape == (n_nodes, 1)
    assert sigma.shape == (n_nodes, 1)
    assert (sigma > 0).all()
    assert not torch.isnan(mu).any()
    assert not torch.isnan(sigma).any()


def test_gaussian_nll_loss_gradients():
    """Verify Gaussian NLL loss computation and non-vanishing gradients."""
    model = ZDynGNN(node_in_dim=20, hidden_dim=16, num_layers=1)
    x = torch.randn(8, 20)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    # 1. Mutation-level forward and backward
    mut_pos = torch.tensor([0, 1, 2], dtype=torch.long)
    mut_wt = torch.tensor([0, 1, 2], dtype=torch.long)
    mut_aa = torch.tensor([3, 4, 5], dtype=torch.long)
    mu_mut, sigma_mut = model(
        x, edge_index, mut_positions=mut_pos, mut_wt_idx=mut_wt, mut_aa_idx=mut_aa
    )
    y_mut = torch.randn(3, 1)
    loss_mut = gaussian_nll_loss(y_mut, mu_mut, sigma_mut)
    loss_mut.backward()

    for name, p in model.named_parameters():
        if "node_mean_head" not in name and "node_logvar_head" not in name:
            assert p.grad is not None, f"Parameter {name} has no gradient in mutation mode."
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient."

    # 2. Node-level forward and backward
    model.zero_grad()
    mu_node, sigma_node = model(x, edge_index)
    y_node = torch.randn(8, 1)
    loss_node = gaussian_nll_loss(y_node, mu_node, sigma_node)
    loss_node.backward()
    assert model.node_mean_head[0].weight.grad is not None
    assert model.node_logvar_head[0].weight.grad is not None

def test_baselines_fit_predict():
    """Verify SiteMeanBaseline and RidgeBaseline on position-disjoint splits."""
    train_df = pd.DataFrame({
        "pos": [43, 43, 44, 44],
        "wt": ["D", "D", "Q", "Q"],
        "mut": ["A", "E", "A", "R"],
        "fitness_score": [-1.0, -0.5, -2.0, -1.8],
    })
    test_df = pd.DataFrame({
        "pos": [45, 45],  # Unseen positions
        "wt": ["R", "R"],
        "mut": ["A", "K"],
        "fitness_score": [-1.2, -1.1],
    })

    # Site-mean baseline: unseen positions should predict global mean
    sm = SiteMeanBaseline().fit(train_df)
    mu_test, sigma_test = sm.predict(test_df)
    assert len(mu_test) == 2
    assert np.allclose(mu_test, train_df["fitness_score"].mean())
    assert (sigma_test > 0).all()

    # Ridge baseline
    ridge = RidgeBaseline(alpha=1.0).fit(train_df)
    r_mu, r_sigma = ridge.predict(test_df)
    assert len(r_mu) == 2
    assert not np.isnan(r_mu).any()


def test_mutation_dataset_and_batch_collator(graph_context):
    """Verify batch collation linking mutations to 3D graph context."""
    mutation_df = pd.DataFrame({
        "pos": [43, 44, 154],
        "wt": ["D", "Q", "M"],
        "mut": ["A", "R", "L"],
        "fitness_score": [-1.2, -0.8, -1.5],
        "replicate_sd": [0.2, 0.3, 0.25],
    })

    dataset = MutationDataset(mutation_df)
    assert len(dataset) == 3

    batch_items = [dataset[i] for i in range(len(dataset))]
    batch = collate_mutation_batch(batch_items, graph_context)

    b = 3
    n = graph_context.num_nodes
    e = graph_context.num_edges

    assert batch["x"].shape == (b * n, 54)
    assert batch["edge_index"].shape == (2, b * e)
    assert batch["edge_dist"].shape == (b * e, 1)
    assert batch["dci_edge"].shape == (b * e, 1)
    assert batch["target_indices"].shape == (b,)
    assert batch["y"].shape == (b, 1)

    # Check target indices mapping
    expected_indices = [
        0 * n + graph_context.pos_to_idx[43],
        1 * n + graph_context.pos_to_idx[44],
        2 * n + graph_context.pos_to_idx[154],
    ]
    assert batch["target_indices"].tolist() == expected_indices
