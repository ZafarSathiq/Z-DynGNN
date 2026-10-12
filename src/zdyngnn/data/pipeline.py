"""Data Preprocessing Pipeline and Cache Generation for Z-DynGNN.

Ingests raw crystallographic coordinates and DMS experimental measurements, performs
residue alignment, computes noise ceiling benchmarks, and exports structured caches to data/processed/.
"""
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from zdyngnn.data.dms_loader import (
    DMS_CONDITIONS,
    filter_missense_dms_data,
    load_raw_dms_excel,
    melt_to_long_format,
)
from zdyngnn.data.noise_ceiling import (
    compute_all_noise_ceilings,
    noise_ceilings_to_dataframe,
)
from zdyngnn.data.pdb_parser import PDBStructureData, parse_pdb_structure


def prepare_and_cache_datasets(
    pdb_path: Union[str, Path] = "DMS Data NDM-1/Strutural data/3SPU.pdb",
    dms_path: Union[str, Path] = "DMS Data NDM-1/41467_2024_52614_MOESM6_ESM.xlsx",
    output_dir: Union[str, Path] = "data/processed",
    chain_id: str = "A",
    res_start: int = 43,
    res_end: int = 270,
    distance_cutoff: float = 8.0,
) -> Dict[str, Any]:
    """Execute end-to-end data preprocessing and write caches to output_dir.

    Args:
        pdb_path: Path to crystal structure PDB.
        dms_path: Path to DMS Excel supplement.
        output_dir: Destination directory for processed files.
        chain_id: Chain ID in PDB (default 'A').
        res_start: First resolved residue index (default 43).
        res_end: Last resolved residue index (default 270).
        distance_cutoff: Distance threshold in Angstroms for contact graph.

    Returns:
        Summary dict containing parsed statistics, output file paths, and alignment checks.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse PDB Structure
    struct_data: PDBStructureData = parse_pdb_structure(
        pdb_path=pdb_path,
        chain_id=chain_id,
        res_start=res_start,
        res_end=res_end,
        distance_cutoff=distance_cutoff,
    )

    structure_cache_path = out_dir / "structure_3spu_ca.npz"
    np.savez_compressed(
        structure_cache_path,
        coords=struct_data.coords,
        dist_matrix=struct_data.dist_matrix,
        adj_matrix=struct_data.adj_matrix,
        degrees=struct_data.degrees,
        b_factors=struct_data.b_factors,
        residue_numbers=struct_data.residue_numbers,
        sequence=np.array(list(struct_data.sequence)),
    )

    # 2. Ingest & Filter DMS Data
    raw_dms = load_raw_dms_excel(dms_path)
    missense_df = filter_missense_dms_data(
        raw_dms,
        res_start=res_start,
        res_end=res_end,
        exclude_stop_codons=True,
        exclude_synonymous=True,
    )

    # Sequence alignment verification: check that DMS WT amino acids match PDB sequence
    alignment_mismatches = []
    pdb_seq_map = {res_num: aa for res_num, aa in zip(struct_data.residue_numbers, struct_data.sequence)}
    for _, row in missense_df[["pos", "wt"]].drop_duplicates().iterrows():
        pos = int(row["pos"])
        wt = str(row["wt"])
        pdb_wt = pdb_seq_map.get(pos)
        if pdb_wt is None or pdb_wt != wt:
            alignment_mismatches.append({"pos": pos, "dms_wt": wt, "pdb_wt": pdb_wt})

    if alignment_mismatches:
        raise ValueError(f"Sequence mismatch detected between DMS WT and PDB 3SPU: {alignment_mismatches}")

    wide_cache_path = out_dir / "dms_missense_wide.csv"
    missense_df.to_csv(wide_cache_path, index=False)
    missense_df.to_csv(out_dir / "dms_missense_wide.csv.gz", index=False, compression="gzip")

    # 3. Transform to Long Format
    long_df = melt_to_long_format(missense_df)
    long_cache_path = out_dir / "dms_missense_long.csv"
    long_df.to_csv(long_cache_path, index=False)
    long_df.to_csv(out_dir / "dms_missense_long.csv.gz", index=False, compression="gzip")

    # 4. Compute Noise Ceilings
    ceilings = compute_all_noise_ceilings(missense_df)
    df_ceilings = noise_ceilings_to_dataframe(ceilings)
    ceilings_csv_path = out_dir / "noise_ceilings.csv"
    df_ceilings.to_csv(ceilings_csv_path, index=False)

    ceilings_json_path = out_dir / "noise_ceilings.json"
    with open(ceilings_json_path, "w") as f:
        json.dump({k: asdict(v) for k, v in ceilings.items()}, f, indent=2)

    # Summary metadata manifest
    manifest = {
        "pdb_id": struct_data.pdb_id,
        "chain_id": struct_data.chain_id,
        "num_resolved_residues": int(struct_data.num_residues),
        "residue_range": [int(struct_data.residue_numbers.min()), int(struct_data.residue_numbers.max())],
        "degree_mean": float(struct_data.degrees.mean()),
        "degree_min": int(struct_data.degrees.min()),
        "degree_max": int(struct_data.degrees.max()),
        "num_missense_variants": len(missense_df),
        "num_conditions": len(DMS_CONDITIONS),
        "num_total_measurements": len(long_df),
        "alignment_verified": True,
        "files_generated": [
            str(structure_cache_path.name),
            str(wide_cache_path.name),
            str(long_cache_path.name),
            str(ceilings_csv_path.name),
            str(ceilings_json_path.name),
        ],
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
