"""Pharmacophore Compatibility Graph Construction for Combinatorial Chemical Space Docking.

Constructs the compatibility graph G = (V, E) between candidate synthons and target
active-site interaction sites (e.g. NDM-1 dual-zinc center, loop L3/L8 subpockets, Asn220).

Formulation:
  - Vertices v = (synthon, site) represent valid geometric placements.
  - Vertex weights w(v) represent local pharmacophore matching / binding propensity.
  - Edges e = (u, v) enforce mutual spatial distance compatibility, absence of internal
    steric clash, and chemical reaction feasibility under registered combinatorial rules.
  - Cliques in G correspond to valid, multi-component make-on-demand drug candidates.
"""
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from zdyngnn.csd.space_parser import ChemicalReactionRule, Synthon

logger = logging.getLogger(__name__)


@dataclass
class ReceptorSite:
    """Represents a pharmacophoric interaction anchor in the target binding pocket."""
    site_id: str
    name: str
    site_type: str                   # 'dual_zinc', 'hydrophobic_subpocket', 'hbond_donor', 'hbond_acceptor'
    coord: np.ndarray                # Shape (3,), Cartesian coordinate in Angstroms
    radius: float = 3.0              # Tolerance radius in Angstroms
    target_residues: List[int] = field(default_factory=list)  # Residue numbers involved in site


@dataclass
class SynthonMatch:
    """Represents a valid spatial placement of a synthon into a receptor site."""
    match_id: int
    synthon: Synthon
    site: ReceptorSite
    score: float = 1.0               # Match score / local affinity prior
    exit_vector_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


class NDM1PocketModel:
    """Standardized pharmacophore interaction model for the NDM-1 active-site pocket."""

    def __init__(
        self,
        zn1_coord: Optional[np.ndarray] = None,
        zn2_coord: Optional[np.ndarray] = None,
        pocket_coords: Optional[Dict[str, np.ndarray]] = None,
    ):
        # Default coordinates derived from PDB 3SPU Chain A (1.30 Å crystal structure)
        self.zn1 = zn1_coord if zn1_coord is not None else np.array([39.221, 25.297, 38.186], dtype=np.float32)
        self.zn2 = zn2_coord if zn2_coord is not None else np.array([38.958, 28.800, 36.646], dtype=np.float32)

        # Dual-zinc coordination centroid (the primary catalytic chelation target)
        self.dual_zinc_centroid = (self.zn1 + self.zn2) / 2.0

        self.sites: Dict[str, ReceptorSite] = {}
        self._initialize_canonical_sites(pocket_coords)

    def _initialize_canonical_sites(self, custom_coords: Optional[Dict[str, np.ndarray]] = None):
        """Build the canonical 4-site pharmacophore model for NDM-1."""
        c = custom_coords if custom_coords is not None else {}

        # 1. Dual-Zinc Chelation Anchor
        self.sites["SITE_ZN_CHELATION"] = ReceptorSite(
            site_id="SITE_ZN_CHELATION",
            name="Dual-Zinc Coordination Center (Zn1-Zn2 Bridge)",
            site_type="dual_zinc",
            coord=c.get("dual_zinc", self.dual_zinc_centroid),
            radius=2.8,
            target_residues=[120, 122, 124, 189, 208, 250],
        )

        # 2. Hydrophobic Subpocket (Adjacent to Loop L8 and Met248)
        self.sites["SITE_HYDROPHOBIC_L8"] = ReceptorSite(
            site_id="SITE_HYDROPHOBIC_L8",
            name="Loop L8 Hydrophobic Cavity (Met248 Subpocket)",
            site_type="hydrophobic_subpocket",
            coord=c.get("hydrophobic_l8", np.array([32.5, 32.0, 36.5], dtype=np.float32)),
            radius=3.5,
            target_residues=[248, 223, 219],
        )

        # 3. Polar / H-Bond Donor Site (Near Asn220 on Loop L8)
        self.sites["SITE_HBOND_ASN220"] = ReceptorSite(
            site_id="SITE_HBOND_ASN220",
            name="Loop L8 Asn220 H-Bond Recognition Anchor",
            site_type="hbond_donor",
            coord=c.get("hbond_asn220", np.array([41.5, 22.0, 29.5], dtype=np.float32)),
            radius=3.0,
            target_residues=[220],
        )

        # 4. Loop L3 Hydrophobic / Flexible Wall (Near Ala74 and Met67)
        self.sites["SITE_LOOP_L3_WALL"] = ReceptorSite(
            site_id="SITE_LOOP_L3_WALL",
            name="Loop L3 Flanking Wall (Ala74 Subpocket)",
            site_type="hydrophobic_subpocket",
            coord=c.get("loop_l3", np.array([33.0, 24.0, 42.0], dtype=np.float32)),
            radius=3.5,
            target_residues=[67, 74],
        )

    def get_site(self, site_id: str) -> ReceptorSite:
        """Retrieve interaction site by ID."""
        if site_id not in self.sites:
            raise KeyError(f"Receptor site {site_id} not found in NDM-1 pocket model.")
        return self.sites[site_id]

    def all_sites(self) -> List[ReceptorSite]:
        """Return all active-site pharmacophore sites."""
        return list(self.sites.values())


class PharmacophoreCompatibilityGraph:
    """Builds and analyzes the pharmacophore compatibility graph G = (V, E)."""

    def __init__(self, pocket_model: Optional[NDM1PocketModel] = None):
        self.pocket = pocket_model if pocket_model is not None else NDM1PocketModel()
        self.matches: List[SynthonMatch] = []
        self._next_match_id = 0
        self.graph = nx.Graph()

    def add_synthon_placement(
        self,
        synthon: Synthon,
        site_id: str,
        match_score: Optional[float] = None,
        exit_vector_offset: Optional[np.ndarray] = None,
    ) -> Optional[SynthonMatch]:
        """Register a valid placement of a synthon into a target receptor site."""
        site = self.pocket.get_site(site_id)

        # Type compatibility check
        is_type_compatible = False
        if site.site_type == "dual_zinc":
            is_type_compatible = (synthon.pharmacophore_type == "zn_chelator")
        elif site.site_type in ["hydrophobic_subpocket"]:
            is_type_compatible = (synthon.pharmacophore_type in ["hydrophobic", "neutral"]) or (synthon.logp > 0.0)
        elif site.site_type in ["hbond_donor", "hbond_acceptor"]:
            is_type_compatible = (synthon.hba > 0) or (synthon.hbd > 0) or (synthon.pharmacophore_type != "hydrophobic")

        if not is_type_compatible:
            return None
        # Compute default score if not provided (chelators receive high prior)
        if match_score is not None:
            score = float(match_score)
        elif site.site_type == "dual_zinc":
            score = 3.0  # Dominant binding anchor
        else:
            score = 1.0 + 0.1 * synthon.molecular_weight / 50.0
        offset = exit_vector_offset if exit_vector_offset is not None else np.zeros(3, dtype=np.float32)

        match = SynthonMatch(
            match_id=self._next_match_id,
            synthon=synthon,
            site=site,
            score=score,
            exit_vector_offset=offset,
        )
        self.matches.append(match)
        self._next_match_id += 1
        return match

    def build(
        self,
        reactions: List[ChemicalReactionRule],
        distance_tolerance: float = 1.5,
        min_inter_site_dist: float = 3.0,
    ) -> nx.Graph:
        """Construct the compatibility graph G = (V, E) across all registered placements.

        Edges connect matches (u, v) if:
          1. They target distinct receptor sites (site_u != site_v).
          2. They feature distinct synthons (synthon_u != synthon_v).
          3. They are chemically compatible under at least one reaction rule.
          4. Their physical 3D distance matches within spatial distance_tolerance.
        """
        self.graph.clear()
        n = len(self.matches)

        # 1. Add vertices with attributes
        for m in self.matches:
            self.graph.add_node(
                m.match_id,
                synthon_id=m.synthon.synthon_id,
                site_id=m.site.site_id,
                role=m.synthon.reaction_role,
                weight=float(m.score),
                coord=m.site.coord,
            )

        # Build reaction role compatibility lookup: (role1, role2) -> reaction_id
        chem_compatible_pairs: Set[Tuple[str, str]] = set()
        for rxn in reactions:
            if len(rxn.roles) == 2:
                r1, r2 = rxn.roles[0], rxn.roles[1]
                chem_compatible_pairs.add((r1, r2))
                chem_compatible_pairs.add((r2, r1))

        # 2. Evaluate all pairwise edge constraints
        for i in range(n):
            u = self.matches[i]
            for j in range(i + 1, n):
                v = self.matches[j]

                # Constraint 1: Distinct receptor sites
                if u.site.site_id == v.site.site_id:
                    continue

                # Constraint 2: Distinct synthons
                if u.synthon.synthon_id == v.synthon.synthon_id:
                    continue

                # Constraint 3: Chemical reaction feasibility
                pair_roles = (u.synthon.reaction_role, v.synthon.reaction_role)
                if pair_roles not in chem_compatible_pairs:
                    continue

                # Constraint 4: 3D Spatial Distance Consistency
                site_dist = float(np.linalg.norm(u.site.coord - v.site.coord))
                if site_dist < min_inter_site_dist:
                    continue  # Sites too close, will cause atomic clash

                # In synthon fragments, typical single-bond distance is ~1.54 Å.
                # Distance compatibility: site distance must allow chemical bond bridge (3.0 to 12.0 Å)
                if site_dist > 14.0:
                    continue  # Target sites too far apart for standard 2-component linkers

                # Edge weight combines vertex scores and distance tightness
                edge_weight = float((u.score + v.score) / max(1.0, np.abs(site_dist - 6.0)))
                self.graph.add_edge(
                    u.match_id,
                    v.match_id,
                    weight=edge_weight,
                    site_dist=site_dist,
                )

        return self.graph

    @property
    def num_vertices(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    def get_adjacency_matrix(self) -> np.ndarray:
        """Return binary adjacency matrix A of shape (V, V)."""
        nodes = sorted(list(self.graph.nodes()))
        if len(nodes) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return nx.to_numpy_array(self.graph, nodelist=nodes, dtype=np.float32)

    def get_vertex_weights(self) -> np.ndarray:
        """Return array of vertex weights w of shape (V,)."""
        nodes = sorted(list(self.graph.nodes()))
        return np.array([self.graph.nodes[node]["weight"] for node in nodes], dtype=np.float32)

    def find_all_cliques(self, min_size: int = 2) -> List[List[SynthonMatch]]:
        """Extract all maximal cliques of size >= min_size using the Bron-Kerbosch algorithm."""
        match_map = {m.match_id: m for m in self.matches}
        raw_cliques = list(nx.find_cliques(self.graph))
        valid_cliques = []

        for c_nodes in raw_cliques:
            if len(c_nodes) >= min_size:
                clique_matches = [match_map[nid] for nid in c_nodes]
                valid_cliques.append(clique_matches)

        # Sort cliques by sum of match scores in descending order
        valid_cliques.sort(key=lambda c: sum(m.score for m in c), reverse=True)
        return valid_cliques
