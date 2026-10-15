"""PDB Coordinate and Contact Graph Parser for NDM-1.

Extracts C_alpha coordinates, distance matrices, and contact degrees from
high-resolution crystal structures (e.g., PDB 3SPU Chain A, residues 43-270).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

# Standard 3-letter to 1-letter amino acid code mapping
AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


@dataclass
class PDBStructureData:
    """Parsed structural representation of a protein chain."""
    pdb_id: str
    chain_id: str
    num_residues: int
    residue_numbers: np.ndarray  # Shape (N,), integer sequence positions (e.g. 43..270)
    sequence: str                # 1-letter amino acid string of length N
    coords: np.ndarray           # Shape (N, 3), C-alpha coordinates in Angstroms
    b_factors: np.ndarray        # Shape (N,), C-alpha B-factors
    dist_matrix: np.ndarray      # Shape (N, N), pairwise Euclidean distances
    adj_matrix: np.ndarray       # Shape (N, N), binary adjacency (dist <= cutoff and dist > 0)
    degrees: np.ndarray          # Shape (N,), contact coordination numbers
    res_to_idx: Dict[int, int]   # Mapping from residue number to 0-based array index


def parse_pdb_structure(
    pdb_path: Union[str, Path],
    chain_id: str = "A",
    res_start: int = 43,
    res_end: int = 270,
    distance_cutoff: float = 8.0,
    model_id: int = 0,
) -> PDBStructureData:
    """Parse C-alpha coordinates and contact topology from a PDB file.

    Args:
        pdb_path: Path to .pdb file.
        chain_id: Target chain identifier (default 'A').
        res_start: Starting residue position inclusive (default 43).
        res_end: Ending residue position inclusive (default 270).
        distance_cutoff: Distance threshold in Angstroms for contact adjacency (default 8.0).
        model_id: Model index in PDB (default 0).

    Returns:
        PDBStructureData containing extracted coordinates, adjacency, and sequence.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found at: {pdb_path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    if structure is None:
        raise ValueError(f"Failed to parse PDB structure from {pdb_path}")
    model = structure[model_id]

    if chain_id not in model:
        raise KeyError(f"Chain '{chain_id}' not found in PDB model {model_id}. Available chains: {list(model.child_dict.keys())}")

    chain = model[chain_id]

    res_nums: List[int] = []
    res_codes: List[str] = []
    coords_list: List[np.ndarray] = []
    b_factors_list: List[float] = []

    for residue in chain:
        # Skip heteroatoms / water (id[0] must be ' ')
        if residue.id[0] != " ":
            continue
        res_num = residue.id[1]
        if res_start <= res_num <= res_end:
            if "CA" in residue:
                ca_atom = residue["CA"]
                res_nums.append(res_num)
                res_3 = residue.resname.upper()
                res_1 = AA_3TO1.get(res_3, seq1(res_3))
                res_codes.append(res_1)
                coords_list.append(ca_atom.get_coord())
                b_factors_list.append(float(ca_atom.get_bfactor()))

    n_res = len(res_nums)
    if n_res == 0:
        raise ValueError(f"No C-alpha residues found in chain {chain_id} in range [{res_start}, {res_end}].")

    residue_numbers = np.array(res_nums, dtype=np.int32)
    coords = np.array(coords_list, dtype=np.float32)
    b_factors = np.array(b_factors_list, dtype=np.float32)
    sequence = "".join(res_codes)

    # Compute pairwise Euclidean distance matrix
    # diff: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1).astype(np.float32)

    # Binary adjacency: within cutoff, self-loops excluded
    adj_matrix = ((dist_matrix <= distance_cutoff) & (dist_matrix > 0.0)).astype(np.int32)
    degrees = adj_matrix.sum(axis=-1).astype(np.int32)

    res_to_idx = {res_num: i for i, res_num in enumerate(res_nums)}

    return PDBStructureData(
        pdb_id=pdb_path.stem,
        chain_id=chain_id,
        num_residues=n_res,
        residue_numbers=residue_numbers,
        sequence=sequence,
        coords=coords,
        b_factors=b_factors,
        dist_matrix=dist_matrix,
        adj_matrix=adj_matrix,
        degrees=degrees,
        res_to_idx=res_to_idx,
    )
