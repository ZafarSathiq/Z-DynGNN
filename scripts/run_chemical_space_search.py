#!/usr/bin/env python3
"""CLI runner for Z-DynGNN Open-Source Combinatorial Chemical Space Docking (CSD)."""
import argparse
import sys
from pathlib import Path

# Defensive path setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zdyngnn.csd.pipeline import run_chemical_space_search


def main():
    parser = argparse.ArgumentParser(
        description="Z-DynGNN Open-Source Chemical Space Docking (CSD) CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--space-file", type=str, default=None, help="Path to vendor .space archive or synthon directory")
    parser.add_argument("--receptor-apo", type=str, default="data/processed/structure_3spu_ca.npz", help="Path to Apo receptor cache (3SPU)")
    parser.add_argument("--receptor-holo", type=str, default="data/processed/dynamics_4rl0.npz", help="Path to Holo receptor cache (4RL0)")
    parser.add_argument("--catalog-path", type=str, default="data/processed/bldb_full_ndm_catalog.json", help="Path to BLDB hospital catalog")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top drug leads to extract and rank")
    parser.add_argument("--alpha", type=float, default=0.05, help="Conformal miscoverage level (1 - alpha confidence)")
    parser.add_argument("--condition", type=str, default="0.25_MEM", choices=["0.25_MEM", "16.0_CTX", "256.0_AMP"], help="Antibiotic calibration condition")
    parser.add_argument("--output-dir", type=str, default="results/csd_search_ndm1", help="Output directory for CSV and manifest")

    args = parser.parse_args()
    csv_path = run_chemical_space_search(
        space_file=args.space_file,
        receptor_apo_path=args.receptor_apo,
        receptor_holo_path=args.receptor_holo,
        catalog_path=args.catalog_path,
        top_k=args.top_k,
        alpha=args.alpha,
        condition=args.condition,
        output_dir=args.output_dir,
    )
    print(f"\n[+] Output generated at: {csv_path}")


if __name__ == "__main__":
    main()
