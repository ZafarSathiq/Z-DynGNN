"""Unit tests for Pharmacophore Compatibility Graph (src/zdyngnn/csd/compatibility_graph.py)."""
import numpy as np
import pytest

from zdyngnn.csd.compatibility_graph import (
    NDM1PocketModel,
    PharmacophoreCompatibilityGraph,
    ReceptorSite,
    SynthonMatch,
)
from zdyngnn.csd.space_parser import (
    STANDARD_COMBINATORIAL_REACTIONS,
    ChemicalReactionRule,
    CombinatorialSpaceParser,
    Synthon,
)


def test_ndm1_pocket_model_canonical_coordinates():
    """Verify that NDM-1 pocket model instantiates canonical active-site interaction sites."""
    pocket = NDM1PocketModel()

    # Verify Dual-Zinc site
    zn_site = pocket.get_site("SITE_ZN_CHELATION")
    assert zn_site.site_type == "dual_zinc"
    # Centroid between Zn1 [39.22, 25.30, 38.19] and Zn2 [38.96, 28.80, 36.65]
    expected_centroid = (np.array([39.221, 25.297, 38.186]) + np.array([38.958, 28.800, 36.646])) / 2.0
    assert np.allclose(zn_site.coord, expected_centroid, atol=1e-3)

    # Verify Loop L8 hydrophobic subpocket
    l8_site = pocket.get_site("SITE_HYDROPHOBIC_L8")
    assert l8_site.site_type == "hydrophobic_subpocket"
    assert 248 in l8_site.target_residues  # Met248

    # Verify all 4 sites are present
    assert len(pocket.all_sites()) == 4


def test_synthon_placement_filtering():
    """Verify that synthon matches are filtered based on pharmacophore and site type."""
    graph_engine = PharmacophoreCompatibilityGraph()

    # 1. Thiol chelator should match dual-zinc site
    s_thiol = Synthon.from_smiles("THIOL_01", "SCC(=O)[1*]", reaction_role="core_acid")
    m_thiol = graph_engine.add_synthon_placement(s_thiol, "SITE_ZN_CHELATION")
    assert m_thiol is not None
    assert m_thiol.score >= 3.0

    # 2. Hydrophobic amine should NOT match dual-zinc site
    s_amine = Synthon.from_smiles("AMINE_01", "CCN[2*]", reaction_role="amine_r1")
    m_amine_zn = graph_engine.add_synthon_placement(s_amine, "SITE_ZN_CHELATION")
    assert m_amine_zn is None  # Rejection by pharmacophore filter

    # 3. Hydrophobic amine should match hydrophobic or polar sites
    m_amine_l8 = graph_engine.add_synthon_placement(s_amine, "SITE_HYDROPHOBIC_L8")
    assert m_amine_l8 is not None


def test_compatibility_graph_construction_and_clique_detection():
    """Verify that compatibility edges and maximal cliques are correctly formed."""
    graph_engine = PharmacophoreCompatibilityGraph()
    reactions = [ChemicalReactionRule.from_dict(d) for d in STANDARD_COMBINATORIAL_REACTIONS]

    # Setup Synthon A: Zinc-chelating core acid placed at dual-zinc site
    s_acid = Synthon.from_smiles("SYN_CHELATOR_ACID", "SCC(=O)[1*]", reaction_role="core_acid")
    m_acid = graph_engine.add_synthon_placement(s_acid, "SITE_ZN_CHELATION", match_score=3.5)

    # Setup Synthon B1: Compatible aromatic amine placed at Loop L8 pocket (reaction: amide coupling)
    s_amine1 = Synthon.from_smiles("SYN_AMINE_AROMATIC", "c1ccccc1CCN[2*]", reaction_role="amine_r1")
    m_amine1 = graph_engine.add_synthon_placement(s_amine1, "SITE_HYDROPHOBIC_L8", match_score=2.0)

    # Setup Synthon B2: Second compatible amine placed at Loop L3 pocket
    s_amine2 = Synthon.from_smiles("SYN_AMINE_ALIPHATIC", "CCCN[2*]", reaction_role="amine_r1")
    m_amine2 = graph_engine.add_synthon_placement(s_amine2, "SITE_LOOP_L3_WALL", match_score=1.8)

    # Build the compatibility graph
    g = graph_engine.build(reactions=reactions)

    assert graph_engine.num_vertices == 3
    # Acid can pair with Amine 1 and Amine 2.
    # However, Amine 1 and Amine 2 cannot pair with each other (both are amine_r1, no amine-amine reaction).
    assert graph_engine.num_edges == 2

    # Check adjacency matrix shape
    adj = graph_engine.get_adjacency_matrix()
    assert adj.shape == (3, 3)
    assert np.all(adj == adj.T)  # Must be symmetric undirected graph

    # Check vertex weights
    weights = graph_engine.get_vertex_weights()
    assert len(weights) == 3
    assert weights[0] == 3.5

    # Extract cliques (should find two 2-cliques: [Acid, Amine1] and [Acid, Amine2])
    cliques = graph_engine.find_all_cliques(min_size=2)
    assert len(cliques) == 2

    # Top clique should be [Acid, Amine1] with score 3.5 + 2.0 = 5.5
    top_clique = cliques[0]
    top_synthon_ids = {m.synthon.synthon_id for m in top_clique}
    assert top_synthon_ids == {"SYN_CHELATOR_ACID", "SYN_AMINE_AROMATIC"}
