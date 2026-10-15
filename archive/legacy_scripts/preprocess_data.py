#!/usr/bin/env python3
"""CLI script to run Step 2 Data Preprocessing Pipeline."""
import argparse
import json
import sys
from pathlib import Path

from zdyngnn.data.pipeline import prepare_and_cache_datasets


def main():
    parser = argparse.ArgumentParser(description="Preprocess 3SPU PDB coordinates and Chen et al. 2024 DMS data.")
    parser.add_argument("--pdb", default="DMS Data NDM-1/Strutural data/3SPU.pdb", help="Path to PDB file.")
    parser.add_argument("--dms", default="DMS Data NDM-1/41467_2024_52614_MOESM6_ESM.xlsx", help="Path to DMS Excel file.")
    parser.add_argument("--out", default="data/processed", help="Path to output directory.")
    parser.add_argument("--cutoff", type=float, default=8.0, help="Distance cutoff in Angstroms.")
    args = parser.parse_args()

    print(f"[*] Running Step 2 Data Pipeline...")
    print(f"    PDB: {args.pdb}")
    print(f"    DMS: {args.dms}")
    print(f"    Output: {args.out}")
    print(f"    Cutoff: {args.cutoff} Å")

    manifest = prepare_and_cache_datasets(
        pdb_path=args.pdb,
        dms_path=args.dms,
        output_dir=args.out,
        distance_cutoff=args.cutoff,
    )

    print("\n[+] Preprocessing Complete!")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
