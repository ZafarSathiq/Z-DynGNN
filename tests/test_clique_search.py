"""Unit and integration tests for src/zdyngnn/csd/clique_search.py."""
import networkx as nx
import numpy as np
import pytest

from zdyngnn.csd.clique_search import (
    DenseSubgraphPeelingSolver,
    MaxWeightCliqueSolver,
    SynthonAssemblyCandidate,
    extract_optimal_synthon_assemblies,
)
from zdyngnn.csd.compatibility_graph import (
    PharmacophoreCompatibilityGraph,
)
from zdyngnn.csd.space_parser import (
    STANDARD_COMBINATORIAL_REACTIONS,
    ChemicalReactionRule,
    Synthon,
)


def test_max_weight_clique_solver_ranking():
    """Verify exact weighted branch-and-bound clique extraction and score ranking."""
    G = nx.Graph()
    # Add 4 vertices
    # Clique 1: {0, 1, 2} with weights 3.5 + 2.0 + 1.5 = 7.0
    # Clique 2: {0, 3} with weights 3.5 + 1.0 = 4.5
    G.add_node(0, weight=3.5, synthon_id="SYN_CORE_ZN", site_id="SITE_ZN", role="core_acid")
    G.add_node(1, weight=2.0, synthon_id="SYN_CAP_L8", site_id="SITE_L8", role="amine_r1")
    G.add_node(2, weight=1.5, synthon_id="SYN_TAIL_L3", site_id="SITE_L3", role="tail_r2")
    G.add_node(3, weight=1.0, synthon_id="SYN_WEAK_L8", site_id="SITE_L8", role="amine_r1")

    G.add_edge(0, 1, weight=5.0)
    G.add_edge(0, 2, weight=4.0)
    G.add_edge(1, 2, weight=3.0)
    G.add_edge(0, 3, weight=2.0)

    solver = MaxWeightCliqueSolver(G, min_size=2, max_size=3)
    results = solver.solve_top_k(top_k=5)

    assert len(results) == 2
    top = results[0]
    assert top.is_clique is True
    assert top.subgraph_density == 1.0
    assert pytest.approx(top.total_weight, 1e-4) == 7.0
    assert set(top.synthon_ids) == {"SYN_CORE_ZN", "SYN_CAP_L8", "SYN_TAIL_L3"}

    second = results[1]
    assert second.is_clique is True
    assert pytest.approx(second.total_weight, 1e-4) == 4.5
    assert set(second.synthon_ids) == {"SYN_CORE_ZN", "SYN_WEAK_L8"}


def test_dense_subgraph_peeling():
    """Verify polynomial greedy degree-peeling algorithm."""
    G = nx.Graph()
    # Create complete graph K4 plus an isolated pendant edge
    for i in range(4):
        G.add_node(i, weight=1.0, synthon_id=f"SYN_{i}")
    for i in range(4):
        for j in range(i + 1, 4):
            G.add_edge(i, j, weight=1.0)

    # Pendant vertex 4 connected only to 0
    G.add_node(4, weight=0.1, synthon_id="SYN_PENDANT")
    G.add_edge(0, 4, weight=0.1)

    peeler = DenseSubgraphPeelingSolver(G, min_size=2, max_size=4)
    res = peeler.solve()

    assert res is not None
    # Dense subgraph should peel vertex 4 and retain the K4 clique {0, 1, 2, 3}
    assert res.subgraph_density == 1.0
    assert "SYN_PENDANT" not in res.synthon_ids
    assert len(res.synthon_ids) == 4


def test_extract_optimal_synthon_assemblies_integration():
    """Verify high-level extraction API coupled with PharmacophoreCompatibilityGraph."""
    graph_engine = PharmacophoreCompatibilityGraph()
    reactions = [ChemicalReactionRule.from_dict(d) for d in STANDARD_COMBINATORIAL_REACTIONS]

    # Setup synthons
    s1 = Synthon.from_smiles("SYN_CHELATOR", "SCC(=O)[1*]", reaction_role="core_acid")
    s2 = Synthon.from_smiles("SYN_AMINE_01", "c1ccccc1CCN[2*]", reaction_role="amine_r1")
    s3 = Synthon.from_smiles("SYN_AMINE_02", "CCCN[2*]", reaction_role="amine_r1")

    graph_engine.add_synthon_placement(s1, "SITE_ZN_CHELATION", match_score=3.5)
    graph_engine.add_synthon_placement(s2, "SITE_HYDROPHOBIC_L8", match_score=2.0)
    graph_engine.add_synthon_placement(s3, "SITE_LOOP_L3_WALL", match_score=1.8)

    g = graph_engine.build(reactions=reactions)

    assemblies = extract_optimal_synthon_assemblies(g, top_k=10, min_size=2, max_size=2)
    assert len(assemblies) == 2

    # Top assembly should be Chelator + Amine 01
    top = assemblies[0]
    assert top.is_clique is True
    assert set(top.synthon_ids) == {"SYN_CHELATOR", "SYN_AMINE_01"}
    assert pytest.approx(top.total_weight, 1e-4) == 5.5
