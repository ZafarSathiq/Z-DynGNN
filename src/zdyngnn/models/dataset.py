"""Mutation Graph Dataset and Batching for Z-DynGNN.

Constructs protein residue interaction graphs and mini-batches pairing single
amino acid variants with their 3D structural and directional dynamics context.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from zdyngnn.models.baselines import AA_LIST, AA_TO_IDX


@dataclass
class ProteinGraphContext:
    """Precomputed structural and biophysical graph context for NDM-1."""
    residue_numbers: np.ndarray      # Shape (N,), sequence positions (43..270)
    pos_to_idx: Dict[int, int]       # Mapping from residue number to 0-based index
    wt_sequence: List[str]           # Wild-type amino acid for each residue (N,)
    node_dynamics: np.ndarray        # Shape (N, 14), topology and dynamics features
    edge_index: np.ndarray           # Shape (2, E), source and destination node indices
    edge_dist: np.ndarray            # Shape (E,), Euclidean distance in Angstroms
    dci_edge: np.ndarray             # Shape (E,), directional DCI_asym from src to dst
    num_nodes: int                   # N = 228
    num_edges: int                   # E
    coords: Optional[np.ndarray] = None  # Shape (N, 3), 3D Cartesian coordinates
    @classmethod
    def from_processed_caches(
        cls,
        structure_path: str = "data/processed/structure_3spu_ca.npz",
        dynamics_path: str = "data/processed/dynamics_3spu.npz",
    ) -> "ProteinGraphContext":
        """Load and assemble static protein graph context from cached Step 2 and Step 3 files."""
        struct_data = np.load(structure_path)
        dyn_data = np.load(dynamics_path)

        res_nums = struct_data["residue_numbers"].astype(np.int32)
        sequence = list(struct_data["sequence"])
        adj_matrix = struct_data["adj_matrix"]
        dist_matrix = struct_data["dist_matrix"]
        node_dynamics = dyn_data["node_features"].astype(np.float32)
        dci_asym = dyn_data["dci_asym"].astype(np.float32)

        n_nodes = len(res_nums)
        pos_to_idx = {int(p): i for i, p in enumerate(res_nums)}

        # Extract edges from binary adjacency
        src_list, dst_list = np.where(adj_matrix > 0)
        edge_index = np.vstack([src_list, dst_list]).astype(np.int64)
        n_edges = edge_index.shape[1]

        # Extract pairwise distances and DCI_asym along each directed edge (src -> dst)
        edge_dist = dist_matrix[src_list, dst_list].astype(np.float32)
        # Note: dci_asym(dst, src) is the driving effect of src on dst
        dci_edge = dci_asym[dst_list, src_list].astype(np.float32)

        return cls(
            residue_numbers=res_nums,
            pos_to_idx=pos_to_idx,
            wt_sequence=sequence,
            node_dynamics=node_dynamics,
            edge_index=edge_index,
            edge_dist=edge_dist,
            dci_edge=dci_edge,
            num_nodes=n_nodes,
            num_edges=n_edges,
            coords=struct_data["coords"].astype(np.float32) if "coords" in struct_data else None,
        )

    def get_base_node_features(self) -> np.ndarray:
        """Construct the wild-type base node feature matrix of shape (N, 54).

        Features:
            - Columns 0..19: WT amino acid one-hot (20)
            - Columns 20..39: Default mut amino acid identical to WT (20)
            - Columns 40..53: Node dynamics and topological features (14)
        """
        n = self.num_nodes
        base_x = np.zeros((n, 54), dtype=np.float32)

        for i, aa in enumerate(self.wt_sequence):
            if aa in AA_TO_IDX:
                aa_idx = AA_TO_IDX[aa]
                base_x[i, aa_idx] = 1.0       # WT one-hot
                base_x[i, 20 + aa_idx] = 1.0  # MUT one-hot (default wild-type)

        base_x[:, 40:54] = self.node_dynamics
        return base_x


class MutationDataset(Dataset):
    """PyTorch Dataset yielding individual mutation records."""

    def __init__(
        self,
        mutation_df: pd.DataFrame,
        score_col: str = "fitness_score",
        sd_col: str = "replicate_sd",
    ):
        self.df = mutation_df.reset_index(drop=True)
        self.score_col = score_col
        self.sd_col = sd_col

        self.positions = self.df["pos"].values.astype(np.int32)
        self.wt = self.df["wt"].values
        self.mut = self.df["mut"].values
        self.scores = self.df[score_col].values.astype(np.float32)

        if sd_col in self.df.columns:
            # Handle potential NaNs in replicate SD with median imputation
            raw_sds = pd.to_numeric(self.df[sd_col], errors="coerce").to_numpy(dtype=np.float64)
            median_sd = float(np.nanmedian(raw_sds)) if not np.all(np.isnan(raw_sds)) else 0.35
            clean_sds = np.where(np.isnan(raw_sds), median_sd, raw_sds).astype(np.float32)
            self.sds = clean_sds
        else:
            self.sds = np.ones(len(self.df), dtype=np.float32) * 0.35

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Union[int, str, float]]:
        return {
            "pos": int(self.positions[index]),
            "wt": str(self.wt[index]),
            "mut": str(self.mut[index]),
            "fitness_score": float(self.scores[index]),
            "replicate_sd": float(self.sds[index]),
        }


def collate_mutation_batch(
    batch: List[Dict[str, Union[int, str, float]]],
    graph_context: ProteinGraphContext,
) -> Dict[str, torch.Tensor]:
    """Collate a batch of mutations into a unified disconnected batch graph tensor.

    Args:
        batch: List of mutation items from MutationDataset.
        graph_context: Precomputed structural graph.

    Returns:
        Dictionary of PyTorch tensors ready for ZDynGNN forward pass.
    """
    batch_size = len(batch)
    n = graph_context.num_nodes
    e = graph_context.num_edges

    # Base node features of shape (N, 54)
    base_x = graph_context.get_base_node_features()

    # Replicate for each graph in batch: (B * N, 54)
    batch_x = np.tile(base_x, (batch_size, 1))

    target_node_indices = []
    y_list = []
    sd_list = []
    pos_list = []

    for b, item in enumerate(batch):
        pos = int(item["pos"])
        mut_aa = str(item["mut"])
        y = float(item["fitness_score"])
        sd = float(item["replicate_sd"])

        local_idx = graph_context.pos_to_idx[pos]
        global_idx = b * n + local_idx
        target_node_indices.append(global_idx)

        # Update mutant amino acid one-hot at the mutated node
        # Clear default mut AA (columns 20..39)
        batch_x[global_idx, 20:40] = 0.0
        if mut_aa in AA_TO_IDX:
            batch_x[global_idx, 20 + AA_TO_IDX[mut_aa]] = 1.0

        y_list.append(y)
        sd_list.append(sd)
        pos_list.append(pos)

    # Replicate edges with offset: edge_index + b * N
    base_edge_index = graph_context.edge_index
    edge_index_list = [base_edge_index + b * n for b in range(batch_size)]
    batch_edge_index = np.hstack(edge_index_list)

    batch_edge_dist = np.tile(graph_context.edge_dist, batch_size)
    batch_dci_edge = np.tile(graph_context.dci_edge, batch_size)

    return {
        "x": torch.from_numpy(batch_x),                                              # (B * N, 54)
        "edge_index": torch.from_numpy(batch_edge_index),                            # (2, B * E)
        "edge_dist": torch.from_numpy(batch_edge_dist).view(-1, 1),                  # (B * E, 1)
        "dci_edge": torch.from_numpy(batch_dci_edge).view(-1, 1),                    # (B * E, 1)
        "target_indices": torch.tensor(target_node_indices, dtype=torch.long),       # (B,)
        "y": torch.tensor(y_list, dtype=torch.float32).view(-1, 1),                  # (B, 1)
        "sd": torch.tensor(sd_list, dtype=torch.float32).view(-1, 1),                # (B, 1)
        "positions": torch.tensor(pos_list, dtype=torch.long),                       # (B,)
    }
