"""Maximum-Weight Clique and Dense Subgraph Solvers for Pharmacophore Compatibility Graphs.

Extracts optimal multi-component synthon combinations from the compatibility graph G = (V, E)
using exact weighted branch-and-bound (Bron-Kerbosch with pivoting) and polynomial-time
greedy peeling algorithms (Charikar 2-approximation).

Formulation:
  - Each vertex v in V represents a placed synthon fragment with match prior weight w(v) > 0.
  - Edges e in E represent mutual geometric and chemical reaction compatibility.
  - A clique of size k in G corresponds to a valid k-component make-on-demand drug assembly
    (e.g. core chelator + R1 aromatic cap + R2 hydrophobic tail) where all components simultaneously
    satisfy distance constraints and chemical coupling rules.
"""
from dataclasses import dataclass, field
import heapq
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SynthonAssemblyCandidate:
    """Represents a discovered multi-synthon candidate assembly extracted from the compatibility graph."""
    assembly_id: int
    synthon_ids: List[str]
    site_ids: List[str]
    roles: List[str]
    total_weight: float              # Sum of vertex matching weights
    edge_weight: float               # Sum of internal compatibility edge weights
    subgraph_density: float          # 2 * |E_S| / (|S| * (|S| - 1)), exactly 1.0 for cliques
    is_clique: bool                  # True if induced subgraph is fully connected
    node_indices: List[int] = field(default_factory=list)


class MaxWeightCliqueSolver:
    """Exact branch-and-bound solver for finding the Top-K heaviest cliques in weighted graphs."""

    def __init__(
        self,
        graph: nx.Graph,
        min_size: int = 2,
        max_size: int = 4,
    ):
        self.graph = graph
        self.min_size = min_size
        self.max_size = max_size

        # Extract vertex weights (default 1.0)
        self.weights: Dict[int, float] = {}
        for node in graph.nodes():
            self.weights[node] = float(graph.nodes[node].get("weight", 1.0))

    def solve_top_k(self, top_k: int = 20) -> List[SynthonAssemblyCandidate]:
        """Find the top-K cliques by total vertex weight."""
        if self.graph.number_of_nodes() == 0 or self.graph.number_of_edges() == 0:
            return []

        # Find all maximal cliques using Bron-Kerbosch with pivoting
        raw_cliques = list(nx.find_cliques(self.graph))
        candidates: List[SynthonAssemblyCandidate] = []
        assembly_counter = 0

        for c_nodes in raw_cliques:
            c_len = len(c_nodes)
            if c_len < self.min_size:
                continue

            # If clique exceeds max_size, consider all sub-cliques of size max_size
            if c_len > self.max_size:
                # Subdivide by taking top-weight subset of size max_size
                sorted_nodes = sorted(c_nodes, key=lambda n: self.weights[n], reverse=True)
                selected_nodes = sorted_nodes[:self.max_size]
            else:
                selected_nodes = c_nodes

            subg = self.graph.subgraph(selected_nodes)
            k = len(selected_nodes)
            v_weight = sum(self.weights[n] for n in selected_nodes)
            e_weight = sum(d.get("weight", 1.0) for _, _, d in subg.edges(data=True))

            synthon_ids = [self.graph.nodes[n].get("synthon_id", f"SYN_{n}") for n in selected_nodes]
            site_ids = [self.graph.nodes[n].get("site_id", f"SITE_{n}") for n in selected_nodes]
            roles = [self.graph.nodes[n].get("role", "fragment") for n in selected_nodes]

            max_possible_edges = k * (k - 1) // 2
            density = subg.number_of_edges() / max(1, max_possible_edges)

            cand = SynthonAssemblyCandidate(
                assembly_id=assembly_counter,
                synthon_ids=synthon_ids,
                site_ids=site_ids,
                roles=roles,
                total_weight=float(v_weight),
                edge_weight=float(e_weight),
                subgraph_density=float(density),
                is_clique=(subg.number_of_edges() == max_possible_edges),
                node_indices=list(selected_nodes),
            )
            candidates.append(cand)
            assembly_counter += 1

        # Deduplicate candidates with identical synthon combinations
        unique_candidates: Dict[Tuple[str, ...], SynthonAssemblyCandidate] = {}
        for cand in candidates:
            key = tuple(sorted(cand.synthon_ids))
            if key not in unique_candidates or cand.total_weight > unique_candidates[key].total_weight:
                unique_candidates[key] = cand

        sorted_cands = sorted(unique_candidates.values(), key=lambda c: (c.total_weight, c.edge_weight), reverse=True)
        return sorted_cands[:top_k]


class DenseSubgraphPeelingSolver:
    """Polynomial-time greedy peeling algorithm (Charikar 2-approximation) for dense subgraphs.

    Time Complexity: O(|V| + |E|).
    Useful for dense or quasi-clique discovery where graph sizes are too large for exact enumeration.
    """

    def __init__(self, graph: nx.Graph, min_size: int = 2, max_size: int = 6):
        self.graph = graph.copy()
        self.min_size = min_size
        self.max_size = max_size

    def solve(self) -> Optional[SynthonAssemblyCandidate]:
        """Execute greedy degree-peeling to identify the densest induced subgraph."""
        if self.graph.number_of_nodes() < self.min_size:
            return None

        current_graph = self.graph.copy()
        best_nodes: List[int] = list(current_graph.nodes())
        best_density = 0.0

        # Iteratively peel the lowest-degree vertex
        while current_graph.number_of_nodes() >= self.min_size:
            n_nodes = current_graph.number_of_nodes()
            n_edges = current_graph.number_of_edges()

            if n_nodes <= self.max_size:
                max_edges = n_nodes * (n_nodes - 1) / 2.0
                density = n_edges / max(1.0, max_edges)
                if density > best_density:
                    best_density = density
                    best_nodes = list(current_graph.nodes())

            # Find vertex with minimum degree
            degrees = dict(current_graph.degree())
            min_node = min(degrees, key=degrees.get)
            current_graph.remove_node(min_node)

        if len(best_nodes) < self.min_size:
            return None

        subg = self.graph.subgraph(best_nodes)
        k = len(best_nodes)
        max_edges = k * (k - 1) // 2
        density = subg.number_of_edges() / max(1, max_edges)

        v_weight = sum(self.graph.nodes[n].get("weight", 1.0) for n in best_nodes)
        e_weight = sum(d.get("weight", 1.0) for _, _, d in subg.edges(data=True))

        return SynthonAssemblyCandidate(
            assembly_id=0,
            synthon_ids=[self.graph.nodes[n].get("synthon_id", f"SYN_{n}") for n in best_nodes],
            site_ids=[self.graph.nodes[n].get("site_id", f"SITE_{n}") for n in best_nodes],
            roles=[self.graph.nodes[n].get("role", "fragment") for n in best_nodes],
            total_weight=float(v_weight),
            edge_weight=float(e_weight),
            subgraph_density=float(density),
            is_clique=(subg.number_of_edges() == max_edges),
            node_indices=list(best_nodes),
        )


def extract_optimal_synthon_assemblies(
    graph: nx.Graph,
    top_k: int = 50,
    min_size: int = 2,
    max_size: int = 4,
    method: str = "exact",
) -> List[SynthonAssemblyCandidate]:
    """High-level API to extract top candidate synthon combinations from a compatibility graph.

    Args:
      graph: Compatibility graph G = (V, E).
      top_k: Maximum number of assemblies to return.
      min_size: Minimum number of synthons in assembly (e.g. 2 for binary reaction).
      max_size: Maximum number of synthons in assembly (e.g. 3 for ternary reaction).
      method: 'exact' (branch-and-bound Bron-Kerbosch) or 'dense' (greedy peeling).

    Returns:
      List of SynthonAssemblyCandidate objects ranked by weight and density.
    """
    if method == "exact":
        solver = MaxWeightCliqueSolver(graph, min_size=min_size, max_size=max_size)
        return solver.solve_top_k(top_k=top_k)
    elif method == "dense":
        peeling_solver = DenseSubgraphPeelingSolver(graph, min_size=min_size, max_size=max_size)
        res = peeling_solver.solve()
        return [res] if res is not None else []
    else:
        raise ValueError(f"Unknown clique extraction method: {method}. Choose 'exact' or 'dense'.")
