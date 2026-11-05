"""Combinatorial Chemical Space Docking (CSD) module for Z-DynGNN.

Provides native open-source parsing and navigation of combinatorial chemical space
files (such as Enamine REAL Space and WuXi GalaXi) without requiring closed-source
proprietary licenses.
"""
from zdyngnn.csd.clique_search import (
    DenseSubgraphPeelingSolver,
    MaxWeightCliqueSolver,
    SynthonAssemblyCandidate,
    extract_optimal_synthon_assemblies,
)
from zdyngnn.csd.compatibility_graph import (
    NDM1PocketModel,
    PharmacophoreCompatibilityGraph,
    ReceptorSite,
    SynthonMatch,
)
from zdyngnn.csd.ensemble_scorer import (
    GaussianRBF,
    HeterogeneousComplexGraph,
    HeterogeneousEnsembleScorer,
    MultiConformalEnsembleScorer,
    build_complex_graph,
    extract_ligand_features,
)
from zdyngnn.csd.resistance_gate import (
    CandidateScreenResult,
    ConformalResistanceGate,
    HospitalVariant,
)
from zdyngnn.csd.pipeline import (
    populate_curated_enamine_space,
    run_chemical_space_search,
)
from zdyngnn.csd.space_parser import (
    ChemicalReactionRule,
    CombinatorialSpaceParser,
    Synthon,
)

__all__ = [
    "Synthon",
    "ChemicalReactionRule",
    "CombinatorialSpaceParser",
    "ReceptorSite",
    "NDM1PocketModel",
    "SynthonMatch",
    "PharmacophoreCompatibilityGraph",
    "GaussianRBF",
    "HeterogeneousComplexGraph",
    "HeterogeneousEnsembleScorer",
    "MultiConformalEnsembleScorer",
    "extract_ligand_features",
    "build_complex_graph",
    "HospitalVariant",
    "CandidateScreenResult",
    "ConformalResistanceGate",
    "SynthonAssemblyCandidate",
    "MaxWeightCliqueSolver",
    "DenseSubgraphPeelingSolver",
    "extract_optimal_synthon_assemblies",
    "populate_curated_enamine_space",
    "run_chemical_space_search",
]
