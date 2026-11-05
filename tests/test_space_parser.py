"""Unit and integration tests for src/zdyngnn/csd/space_parser.py."""
import pytest
from rdkit import Chem

from zdyngnn.csd.space_parser import (
    ChemicalReactionRule,
    CombinatorialSpaceParser,
    Synthon,
)


def test_synthon_parsing_and_exit_vectors():
    """Verify that synthon SMILES with isotopic attachment vectors ([1*], [2*]) are correctly parsed."""
    # Acid synthon with [1*] attached to carbonyl
    s_acid = Synthon.from_smiles(
        synthon_id="SYN_ACID_01",
        smiles="CC(=O)[1*]",
        reaction_role="core_acid",
    )
    assert s_acid is not None
    assert s_acid.synthon_id == "SYN_ACID_01"
    assert s_acid.molecular_weight > 40.0
    assert 1 in s_acid.exit_points

    # Amine synthon with [2*] attached to nitrogen
    s_amine = Synthon.from_smiles(
        synthon_id="SYN_AMINE_01",
        smiles="NCC[2*]",
        reaction_role="amine_r1",
    )
    assert s_amine is not None
    assert 2 in s_amine.exit_points


def test_zinc_chelator_detection():
    """Verify that dual-zinc chelators (thiols, carboxylates, phosphonates) are automatically classified."""
    # Thiol chelator (e.g. captopril-like anchor)
    s_thiol = Synthon.from_smiles(
        synthon_id="CHELATOR_THIOL",
        smiles="SC[1*]",
        reaction_role="thiol_chelator",
    )
    assert s_thiol is not None
    assert s_thiol.pharmacophore_type == "zn_chelator"

    # Carboxylate chelator
    s_carb = Synthon.from_smiles(
        synthon_id="CHELATOR_CARBOXYL",
        smiles="OC(=O)C[1*]",
        reaction_role="core_acid",
    )
    assert s_carb is not None
    assert s_carb.pharmacophore_type == "zn_chelator"

    # Phosphonate chelator
    s_phos = Synthon.from_smiles(
        synthon_id="CHELATOR_PHOS",
        smiles="OP(=O)(O)C[1*]",
        reaction_role="core_acid",
    )
    assert s_phos is not None
    assert s_phos.pharmacophore_type == "zn_chelator"

    # Neutral aromatic ring (non-chelator)
    s_neutral = Synthon.from_smiles(
        synthon_id="NEUTRAL_BENZENE",
        smiles="c1ccccc1[2*]",
        reaction_role="amine_r1",
    )
    assert s_neutral is not None
    assert s_neutral.pharmacophore_type == "neutral"


def test_reaction_rule_amide_coupling():
    """Verify that chemical reaction rules correctly assemble synthons into sanitized products."""
    parser = CombinatorialSpaceParser()

    s_acid = parser.add_synthon("ACID_BENZOIC", "c1ccccc1C(=O)[1*]", reaction_role="core_acid")
    s_amine = parser.add_synthon("AMINE_METHYL", "CN[2*]", reaction_role="amine_r1")

    assert s_acid is not None
    assert s_amine is not None

    product = parser.assemble_molecule("REAC_01_AMIDE", [s_acid, s_amine])
    assert product is not None

    prod_smiles = Chem.MolToSmiles(product)
    # Product should be N-methylbenzamide or an amide derivative without dummy atoms
    assert "*" not in prod_smiles
    assert "C(=O)N" in prod_smiles or "NC(=O)" in prod_smiles


def test_combinatorial_enumeration_and_space_capacity():
    """Verify combinatorial assembly enumeration and space size calculation."""
    parser = CombinatorialSpaceParser()

    # Add 2 acids
    parser.add_synthon("ACID_1", "CC(=O)[1*]", reaction_role="core_acid")
    core_chelator = parser.add_synthon("ACID_CHELATOR", "SCC(=O)[1*]", reaction_role="core_acid")

    # Add 3 amines
    parser.add_synthon("AMINE_1", "CN[2*]", reaction_role="amine_r1")
    parser.add_synthon("AMINE_2", "CCN[2*]", reaction_role="amine_r1")
    parser.add_synthon("AMINE_3", "c1ccccc1N[2*]", reaction_role="amine_r1")

    # Space size for REAC_01_AMIDE: 2 acids * 3 amines = 6 virtual molecules
    assert parser.calculate_space_size() >= 6

    # Enumerate assemblies using core chelator
    candidates = parser.enumerate_compatible_assemblies("REAC_01_AMIDE", core_chelator, max_candidates=10)
    assert len(candidates) == 3

    for cand in candidates:
        assert "smiles" in cand
        assert "sa_score" in cand
        assert cand["core_synthon_id"] == "ACID_CHELATOR"
        assert cand["sa_score"] <= 3.0
