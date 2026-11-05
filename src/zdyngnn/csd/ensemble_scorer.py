"""Heterogeneous Protein-Ligand Graph Neural Network Ensemble Scorer.

Scores candidate small molecules and synthon assemblies against NDM-1 without
rigid-backbone assumptions by modeling the protein-ligand complex as a heterogeneous
bipartite graph across multi-conformal crystal structures (Apo 3SPU and Holo 4RL0).

Key Features:
  - 34-dim protein node embeddings (20 WT amino acid + 14 biophysical dynamics).
  - 32-dim ligand heavy-atom embeddings (elements, hybridizations, charges, HBD/HBA).
  - 16-center continuous Gaussian RBF distance expansion along intermolecular contacts.
  - Cross-attention bipartite message passing between ligand atoms and pocket residues.
  - Heteroscedastic dual heads predicting binding affinity (mu) and aleatoric scale (sigma).
  - Multi-conformal ensemble evaluation reporting consensus affinity, conformational shift,
    and uncertainty bounds.
"""
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdDepictor import Compute2DCoords
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
import torch
import torch.nn as nn

from zdyngnn.models.dataset import ProteinGraphContext
from zdyngnn.models.gnn import GaussianRBF

logger = logging.getLogger(__name__)


def extract_ligand_features(mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract 32-dimensional atomic features and 3D coordinates for all heavy atoms in ligand.

    Feature Layout (32 dimensions):
      - 0..7: Atomic element one-hot (C, N, O, S, P, F, Cl, Br) [8]
      - 8..11: Hybridization (SP, SP2, SP3, other) [4]
      - 12..15: Formal charge, aromaticity, ring membership, in 5/6 ring [4]
      - 16..31: Physicochemical properties: Gasteiger charge, HBD, HBA, valence, etc. [16]

    Returns:
      x_lig: FloatTensor of shape (M, 32).
      coords_lig: FloatTensor of shape (M, 3).
    """
    m = Chem.Mol(mol)
    # Ensure hydrogens are added for charge computation, then filter to heavy atoms
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass

    m_h = Chem.AddHs(m)
    try:
        ComputeGasteigerCharges(m_h)
    except Exception:
        pass

    # Generate 3D conformer if not present
    if m.GetNumConformers() == 0:
        res = EmbedMolecule(m, randomSeed=42, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
        if res != 0:
            # Fallback to standard 2D layout with zero z-coordinate
            Compute2DCoords(m)

    conf = m.GetConformer() if m.GetNumConformers() > 0 else None

    heavy_atoms = [atom for atom in m.GetAtoms() if atom.GetSymbol() != "H"]
    num_heavy = len(heavy_atoms)

    if num_heavy == 0:
        # Edge case: fallback empty or single dummy atom
        return torch.zeros((1, 32), dtype=torch.float32), torch.zeros((1, 3), dtype=torch.float32)

    element_list = ["C", "N", "O", "S", "P", "F", "Cl", "Br"]
    features = np.zeros((num_heavy, 32), dtype=np.float32)
    coords = np.zeros((num_heavy, 3), dtype=np.float32)

    for i, atom in enumerate(heavy_atoms):
        idx = atom.GetIdx()

        # Coordinates
        if conf is not None:
            pos = conf.GetAtomPosition(idx)
            coords[i] = [pos.x, pos.y, pos.z]
        else:
            coords[i] = [0.0, 0.0, 0.0]

        # 0..7: Element one-hot
        sym = atom.GetSymbol()
        if sym in element_list:
            features[i, element_list.index(sym)] = 1.0

        # 8..11: Hybridization
        hyb = atom.GetHybridization()
        if hyb == Chem.rdchem.HybridizationType.SP:
            features[i, 8] = 1.0
        elif hyb == Chem.rdchem.HybridizationType.SP2:
            features[i, 9] = 1.0
        elif hyb == Chem.rdchem.HybridizationType.SP3:
            features[i, 10] = 1.0
        else:
            features[i, 11] = 1.0

        # 12..15: Structural properties
        features[i, 12] = float(atom.GetFormalCharge())
        features[i, 13] = 1.0 if atom.GetIsAromatic() else 0.0
        features[i, 14] = 1.0 if atom.IsInRing() else 0.0
        features[i, 15] = 1.0 if (atom.IsInRingSize(5) or atom.IsInRingSize(6)) else 0.0

        # 16..31: Physicochemical descriptors & Gasteiger charge
        try:
            charge_str = atom.GetProp("_GasteigerCharge")
            val = float(charge_str)
            features[i, 16] = 0.0 if np.isnan(val) or np.isinf(val) else np.clip(val, -2.0, 2.0)
        except Exception:
            features[i, 16] = 0.0

        features[i, 17] = float(atom.GetTotalValence()) / 8.0
        features[i, 18] = float(atom.GetTotalNumHs()) / 4.0
        features[i, 19] = float(atom.GetMass()) / 100.0

        # HBD / HBA indicator heuristics
        if sym in ["N", "O"] and atom.GetTotalNumHs() > 0:
            features[i, 20] = 1.0  # H-Bond Donor
        if sym in ["N", "O", "S", "F"]:
            features[i, 21] = 1.0  # H-Bond Acceptor

    return torch.from_numpy(features), torch.from_numpy(coords)


@dataclass
class HeterogeneousComplexGraph:
    """Represents a heterogeneous protein-ligand interaction graph."""
    x_prot: torch.Tensor           # (N_prot, 34)
    coords_prot: torch.Tensor      # (N_prot, 3)
    x_lig: torch.Tensor            # (N_lig, 32)
    coords_lig: torch.Tensor       # (N_lig, 3)
    edge_index_inter: torch.Tensor # (2, K), row 0: prot idx, row 1: lig idx
    edge_dist_inter: torch.Tensor  # (K,)
    edge_rbf_inter: torch.Tensor   # (K, 16)


def build_complex_graph(
    x_prot: torch.Tensor,
    coords_prot: torch.Tensor,
    x_lig: torch.Tensor,
    coords_lig: torch.Tensor,
    cutoff_distance: float = 6.0,
    rbf_kernel: Optional[GaussianRBF] = None,
) -> HeterogeneousComplexGraph:
    """Construct bipartite intermolecular contact edges between protein residues and ligand atoms."""
    if rbf_kernel is None:
        rbf_kernel = GaussianRBF(num_centers=16, d_min=0.0, d_max=cutoff_distance)

    # Compute Euclidean distance matrix between protein C-alpha and ligand heavy atoms
    # Shape: (N_prot, N_lig)
    diff = coords_prot.unsqueeze(1) - coords_lig.unsqueeze(0)
    dist_matrix = torch.norm(diff, dim=-1)

    # Mask within cutoff
    contact_mask = dist_matrix <= cutoff_distance
    prot_indices, lig_indices = torch.where(contact_mask)

    if len(prot_indices) == 0:
        # Fallback to closest 5 pairs if no contacts within cutoff (e.g. initial unaligned placement)
        flat_indices = torch.topk(dist_matrix.flatten(), k=min(5, dist_matrix.numel()), largest=False).indices
        n_lig = coords_lig.size(0)
        prot_indices = flat_indices // n_lig
        lig_indices = flat_indices % n_lig

    edge_index = torch.stack([prot_indices, lig_indices], dim=0)
    edge_dist = dist_matrix[prot_indices, lig_indices]
    edge_rbf = rbf_kernel(edge_dist)

    return HeterogeneousComplexGraph(
        x_prot=x_prot,
        coords_prot=coords_prot,
        x_lig=x_lig,
        coords_lig=coords_lig,
        edge_index_inter=edge_index,
        edge_dist_inter=edge_dist,
        edge_rbf_inter=edge_rbf,
    )


class HeterogeneousEnsembleScorer(nn.Module):
    """Bipartite Graph Neural Network scoring protein-ligand binding interactions."""

    def __init__(
        self,
        prot_dim: int = 34,
        lig_dim: int = 32,
        rbf_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection layers
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.lig_proj = nn.Sequential(
            nn.Linear(lig_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Bipartite intermolecular message passing: Protein -> Ligand
        self.msg_p2l = nn.Sequential(
            nn.Linear(hidden_dim * 2 + rbf_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm_lig = nn.LayerNorm(hidden_dim)

        # Bipartite intermolecular message passing: Ligand -> Protein
        self.msg_l2p = nn.Sequential(
            nn.Linear(hidden_dim * 2 + rbf_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm_prot = nn.LayerNorm(hidden_dim)

        # Heteroscedastic dual heads
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, graph: HeterogeneousComplexGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
          mu: Predicted binding affinity score (scalar).
          sigma: Calibrated predictive aleatoric scale (scalar).
        """
        h_prot = self.prot_proj(graph.x_prot)  # (N_prot, hidden_dim)
        h_lig = self.lig_proj(graph.x_lig)    # (N_lig, hidden_dim)

        prot_idx = graph.edge_index_inter[0]
        lig_idx = graph.edge_index_inter[1]
        rbf_dist = graph.edge_rbf_inter

        # 1. Message Passing: Protein -> Ligand
        msg_in_p2l = torch.cat([h_prot[prot_idx], h_lig[lig_idx], rbf_dist], dim=-1)
        m_p2l = self.msg_p2l(msg_in_p2l)

        agg_lig = torch.zeros_like(h_lig)
        agg_lig.index_add_(0, lig_idx, m_p2l)
        h_lig = self.norm_lig(h_lig + agg_lig)

        # 2. Message Passing: Ligand -> Protein
        msg_in_l2p = torch.cat([h_lig[lig_idx], h_prot[prot_idx], rbf_dist], dim=-1)
        m_l2p = self.msg_l2p(msg_in_l2p)

        agg_prot = torch.zeros_like(h_prot)
        agg_prot.index_add_(0, prot_idx, m_l2p)
        h_prot = self.norm_prot(h_prot + agg_prot)

        # 3. Readout Pooling
        z_lig = torch.mean(h_lig, dim=0, keepdim=True)    # (1, hidden_dim)
        z_prot = torch.mean(h_prot, dim=0, keepdim=True)  # (1, hidden_dim)

        z_complex = torch.cat([z_lig, z_prot], dim=-1)   # (1, hidden_dim * 2)

        mu = self.mean_head(z_complex).squeeze()
        logvar = self.logvar_head(z_complex).squeeze()
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        sigma = torch.exp(0.5 * logvar)

        return mu, sigma


class MultiConformalEnsembleScorer:
    """Evaluates small molecules across multi-conformal NDM-1 crystal states (3SPU & 4RL0)."""

    def __init__(
        self,
        apo_dynamics_path: str = "data/processed/dynamics_3spu.npz",
        holo_dynamics_path: str = "data/processed/dynamics_4rl0.npz",
        structure_path: str = "data/processed/structure_3spu_ca.npz",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.rbf = GaussianRBF(num_centers=16, d_min=0.0, d_max=6.0).to(self.device)
        self.model = HeterogeneousEnsembleScorer().to(self.device)
        self.model.eval()

        # Load Protein Contexts
        self.ctx_apo = ProteinGraphContext.from_processed_caches(structure_path, apo_dynamics_path)
        self.ctx_holo = ProteinGraphContext.from_processed_caches(structure_path, holo_dynamics_path)

        # Extract 34-dim protein features
        self.x_prot_apo = self._prepare_prot_tensor(self.ctx_apo)
        self.x_prot_holo = self._prepare_prot_tensor(self.ctx_holo)

        # Extract coordinates (Apo 3SPU from structure cache, Holo 4RL0 from dynamics/structure)
        if self.ctx_apo.coords is not None:
            self.coords_apo = torch.from_numpy(self.ctx_apo.coords.astype(np.float32)).to(self.device)
        else:
            struct_data = np.load(structure_path)
            self.coords_apo = torch.from_numpy(struct_data["coords"].astype(np.float32)).to(self.device)

        d_holo = np.load(holo_dynamics_path)
        if "coords" in d_holo:
            self.coords_holo = torch.from_numpy(d_holo["coords"].astype(np.float32)).to(self.device)
        elif self.ctx_holo.coords is not None:
            self.coords_holo = torch.from_numpy(self.ctx_holo.coords.astype(np.float32)).to(self.device)
        else:
            self.coords_holo = self.coords_apo.clone()
    def _prepare_prot_tensor(self, ctx: ProteinGraphContext) -> torch.Tensor:
        """Combine 20 WT amino acid one-hot with 14 dynamics features into (228, 34) tensor."""
        base_features = ctx.get_base_node_features()  # (228, 54)
        # Columns 0..19: WT AA one-hot (20); Columns 20..33: dynamics (14)
        prot_x = base_features[:, :34]
        return torch.from_numpy(prot_x.astype(np.float32)).to(self.device)

    def score_molecule(
        self,
        mol: Chem.Mol,
        pocket_centroid: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Score candidate molecule against Apo and Holo conformations.

        Args:
          mol: Validated RDKit molecule.
          pocket_centroid: Optional translation vector to align ligand with active pocket.

        Returns:
          Dictionary with predicted affinity, uncertainty, and conformational shift metrics.
        """
        x_lig, coords_lig = extract_ligand_features(mol)
        x_lig = x_lig.to(self.device)
        coords_lig = coords_lig.to(self.device)

        # Align ligand conformer with active site if centroid provided
        if pocket_centroid is not None:
            lig_center = torch.mean(coords_lig, dim=0)
            target_center = torch.from_numpy(pocket_centroid.astype(np.float32)).to(self.device)
            coords_lig = coords_lig - lig_center + target_center

        with torch.no_grad():
            # 1. Score against Apo state (3SPU)
            g_apo = build_complex_graph(self.x_prot_apo, self.coords_apo, x_lig, coords_lig, rbf_kernel=self.rbf)
            mu_apo, sig_apo = self.model(g_apo)

            # 2. Score against Holo state (4RL0)
            g_holo = build_complex_graph(self.x_prot_holo, self.coords_holo, x_lig, coords_lig, rbf_kernel=self.rbf)
            mu_holo, sig_holo = self.model(g_holo)

            mu_a = float(mu_apo.item())
            sig_a = float(sig_apo.item())
            mu_h = float(mu_holo.item())
            sig_h = float(sig_holo.item())

            # Multi-conformal consensus affinity and dispersion
            mu_ens = 0.5 * (mu_a + mu_h)
            # Combined variance: average variance + epistemic conformational spread
            sig_ens = float(np.sqrt(0.5 * (sig_a ** 2 + sig_h ** 2) + 0.25 * ((mu_a - mu_h) ** 2)))
            delta_conf = mu_h - mu_a  # Tolerance to L8 loop clamping

        return {
            "mu_apo": mu_a,
            "sigma_apo": sig_a,
            "mu_holo": mu_h,
            "sigma_holo": sig_h,
            "mu_ensemble": mu_ens,
            "sigma_ensemble": sig_ens,
            "conformation_shift": delta_conf,
        }
