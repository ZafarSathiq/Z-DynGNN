"""Z-score Calibrated Dynamics-Aware Graph Neural Network (Z-DynGNN).

Architecture pairing biophysical directional perturbation dynamics (DCI_asym)
with continuous radial basis edge distances and a heteroscedastic dual-head
output (predictive mean mu and predictive aleatoric scale sigma) trained via
Gaussian Negative Log-Likelihood (NLL) loss.
"""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianRBF(nn.Module):
    """Radial Basis Function (RBF) expansion of continuous distances."""

    def __init__(self, num_centers: int = 16, d_min: float = 0.0, d_max: float = 8.0):
        super().__init__()
        self.num_centers = num_centers
        centers = torch.linspace(d_min, d_max, num_centers)
        self.register_buffer("centers", centers)
        step = (d_max - d_min) / max(1, num_centers - 1)
        gamma = 0.5 / (step ** 2)
        self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """Expand distance tensor d of shape (E,) or (E, 1) into (E, num_centers)."""
        d = d.view(-1, 1)
        return torch.exp(-self.gamma * ((d - self.centers) ** 2))


class DynamicsMessagePassingLayer(nn.Module):
    """Message passing layer with directional DCI_asym edge modulation and residual update."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Message network: source node + edge features + DCI_asym
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim + 1, node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )

        # Node update network
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )

        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        dci_edge: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            h: Node representations, shape (N, node_dim).
            edge_index: Graph edge indices, shape (2, E). (edge_index[0]=src, edge_index[1]=dst).
            edge_attr: Edge attributes (RBF distances), shape (E, edge_dim).
            dci_edge: Directional DCI_asym values along edges (src -> dst), shape (E, 1).

        Returns:
            Updated node representations, shape (N, node_dim).
        """
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        n_nodes = h.size(0)

        # Defensive ensure 2D shapes
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        if dci_edge.dim() == 1:
            dci_edge = dci_edge.view(-1, 1)

        # Concatenate source node features, edge attributes, and DCI_asym modulation
        msg_input = torch.cat([h[src_nodes], edge_attr, dci_edge], dim=-1)
        messages = self.msg_mlp(msg_input)

        # Aggregate messages into destination nodes via index_add
        aggregated = torch.zeros(n_nodes, self.node_dim, device=h.device, dtype=h.dtype)
        aggregated.index_add_(0, dst_nodes, messages)

        # Update node representations with residual connection
        update_input = torch.cat([h, aggregated], dim=-1)
        h_update = self.update_mlp(update_input)
        h_next = self.norm(h + h_update)

        return h_next


class ZDynGNN(nn.Module):
    """Z-score Calibrated Dynamics-Aware Graph Neural Network (Z-DynGNN).

    Pairs biophysical directional perturbation dynamics (DCI_asym) with
    continuous radial basis edge distances and a heteroscedastic dual-head
    output (predictive mean mu and predictive aleatoric scale sigma).
    """

    def __init__(
        self,
        node_in_dim: int = 34,       # WT AA one-hot (20) + structural/dynamics features (14)
        num_rbf: int = 16,           # RBF distance centers
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        dist_cutoff: float = 8.0,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.num_rbf = num_rbf
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dist_cutoff = dist_cutoff

        # RBF distance expansion
        self.rbf = GaussianRBF(num_centers=num_rbf, d_min=0.0, d_max=dist_cutoff)

        # Encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        # Message passing layers
        self.layers = nn.ModuleList([
            DynamicsMessagePassingLayer(
                node_dim=hidden_dim,
                edge_dim=hidden_dim // 2,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Node-level default heads (for node-level forward pass)
        self.node_mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.node_logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Mutation-level dual heads: takes [h_pos (hidden_dim), wt_onehot (20), mut_onehot (20)] = hidden_dim + 40
        self.mutation_mean_head = nn.Sequential(
            nn.Linear(hidden_dim + 40, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.mutation_logvar_head = nn.Sequential(
            nn.Linear(hidden_dim + 40, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: Optional[torch.Tensor] = None,
        dci_edge: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        mut_positions: Optional[torch.Tensor] = None,
        mut_wt_idx: Optional[torch.Tensor] = None,
        mut_aa_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Node features of shape (N, node_in_dim).
            edge_index: Graph edge indices of shape (2, E).
            edge_dist: Continuous Euclidean edge distances of shape (E,) or (E, 1).
            dci_edge: Directional DCI_asym edge weights of shape (E,) or (E, 1).
            edge_attr: Optional pre-computed edge features.
            mut_positions: Optional tensor of mutated node indices, shape (M,).
            mut_wt_idx: Optional tensor of WT amino acid indices in 0..19, shape (M,).
            mut_aa_idx: Optional tensor of MUT amino acid indices in 0..19, shape (M,).

        Returns:
            mu: Predictive mean fitness effect, shape (M, 1) or (N, 1).
            sigma: Predictive aleatoric standard deviation, shape (M, 1) or (N, 1).
        """
        n_edges = edge_index.size(1)
        device = x.device

        # Handle edge attributes
        if edge_attr is None:
            if edge_dist is None:
                edge_dist = torch.ones(n_edges, 1, device=device)
            rbf_feats = self.rbf(edge_dist)
            edge_feats = self.edge_encoder(rbf_feats)
        else:
            if edge_attr.size(-1) == self.num_rbf:
                edge_feats = self.edge_encoder(edge_attr)
            else:
                edge_feats = edge_attr

        # Handle DCI_asym modulation
        if dci_edge is None:
            dci_edge = torch.zeros(n_edges, 1, device=device)
        else:
            dci_edge = dci_edge.view(-1, 1)

        # Encode nodes
        h = self.node_encoder(x)

        # Apply message passing layers
        for layer in self.layers:
            h = layer(h, edge_index, edge_feats, dci_edge)

        # Mode A: Mutation-specific forward pass
        if mut_positions is not None and mut_wt_idx is not None and mut_aa_idx is not None:
            h_pos = h[mut_positions]
            wt_oh = F.one_hot(mut_wt_idx, num_classes=20).float().to(device)
            mut_oh = F.one_hot(mut_aa_idx, num_classes=20).float().to(device)
            z_mut = torch.cat([h_pos, wt_oh, mut_oh], dim=-1)

            mu = self.mutation_mean_head(z_mut)
            logvar = self.mutation_logvar_head(z_mut)

        # Mode B: Node-level default forward pass
        else:
            mu = self.node_mean_head(h)
            logvar = self.node_logvar_head(h)

        # Clamping log-variance for numerical stability: logvar in [-10.0, 5.0]
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        sigma = torch.exp(0.5 * logvar)

        return mu, sigma


def gaussian_nll_loss(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Gaussian Negative Log-Likelihood (NLL) loss with heteroscedastic scale parameter.

    L_NLL = 0.5 * ( ((y - mu) / sigma)^2 + 2 * ln(sigma) )

    Args:
        y_true: Ground truth target values of shape (N,) or (N, 1).
        mu: Predictive mean values of shape (N,) or (N, 1).
        sigma: Predictive standard deviations of shape (N,) or (N, 1).
        reduction: 'mean' or 'sum'.

    Returns:
        Scalar loss tensor.
    """
    y_true = y_true.view(-1, 1)
    mu = mu.view(-1, 1)
    sigma = sigma.view(-1, 1)

    var = sigma ** 2
    logvar = 2.0 * torch.log(sigma + 1e-8)
    z_squared = ((y_true - mu) ** 2) / (var + 1e-8)

    loss = 0.5 * (z_squared + logvar)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
