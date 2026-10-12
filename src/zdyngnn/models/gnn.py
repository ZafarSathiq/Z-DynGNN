"""Z-score Calibrated Dynamics-Aware Graph Neural Network (Z-DynGNN).

Architecture predicting predictive mean mu(x) and predictive scale sigma(x)
to parameterize studentized Z-score residuals for distribution-free conformal calibration.
"""
import torch
import torch.nn as nn


class ZDynGNN(nn.Module):
    """Z-score Calibrated Dynamics-Aware Graph Neural Network (Z-DynGNN).

    Pairs biophysical directional perturbation dynamics (DCI_asym) with
    continuous radial basis edge distances and a heteroscedastic dual-head
    output (mean mu and aleatoric scale sigma).
    """

    def __init__(
        self,
        node_in_dim: int = 20,
        edge_in_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Node feature projection
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)

        # Dual output heads: predictive mean and predictive log-variance (Z-score scale parameter)
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        """Forward pass.

        Returns:
            mu: Predictive mean fitness effect.
            sigma: Predictive aleatoric standard deviation (Z-score scale parameter).
        """
        h = self.node_encoder(x)
        mu = self.mean_head(h)
        logvar = self.logvar_head(h)
        # Numerical stability clamp for variance
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma
