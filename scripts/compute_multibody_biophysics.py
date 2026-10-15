#!/usr/bin/env python3
"""Execute Zero-Training Path 1 Multi-Body Biophysical Validation.

Extracts higher-order 3-body allosteric cumulants and 6-body catalytic zinc cage
generalized fluctuation volumes across 4 NDM-1 crystal structures:
  - 3SPU (Apo native, 1.30 Å)
  - 4HL2 (Ampicillin-bound open state, 1.05 Å)
  - 4RL0 (Meropenem-bound clamped intermediate, 1.30 Å)
  - 5ZGY (Cryogenic ultra-high resolution, 0.95 Å)

Saves the node-level multi-body feature tensor to data/processed/multibody_3spu.npz
and audit summary to results/multibody_biophysical_audit.json.
"""
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np

from zdyngnn.dynamics.enm import build_kirchhoff_matrix, compute_gnm_pseudoinverse
from zdyngnn.dynamics.multibody import (
    compute_multibody_cumulant,
    compute_node_multibody_features,
    extract_catalytic_hexamer_metrics,
    find_contact_cliques,
)

OUTPUT_NPZ = Path("data/processed/multibody_3spu.npz")
AUDIT_JSON = Path("results/multibody_biophysical_audit.json")


def run_multibody_biophysical_audit() -> Dict:
    t0 = time.time()
    print("=" * 78)
    print("PATH 1: ZERO-TRAINING MULTI-BODY ALLOSTERY & CATALYTIC CAGE AUDIT")
    print("=" * 78)

    structures = [
        ("3SPU", "Apo Native (1.30 Å)", "data/processed/dynamics_3spu.npz"),
        ("4HL2", "Ampicillin-Bound Open (1.05 Å)", "data/processed/dynamics_4hl2.npz"),
        ("4RL0", "Meropenem-Bound Clamped (1.30 Å)", "data/processed/dynamics_4rl0.npz"),
        ("5ZGY", "Cryogenic Ultra-High Res (0.95 Å)", "data/processed/dynamics_5zgy.npz"),
        ("5O2E", "Cephalosporin-Bound (1.30 Å)", "data/processed/dynamics_5o2e.npz"),
    ]

    results_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "structures_evaluated": {},
    }

    # Catalytic hexamer indices: His120(77), His122(79), Asp124(81), His189(146), Cys208(165), His250(207)
    cat_indices = (77, 79, 81, 146, 165, 207)
    
    # Specific allosteric triplets
    triplets = {
        "M154_L8_H120": (111, 168, 77),   # M154 - K211(L8) - H120(Zn1)
        "M154_L8_D124": (111, 175, 81),   # M154 - L218(L8 apex) - D124(Zn1)
        "M154_L3_H250": (111, 26, 207),   # M154 - G69(L3 loop) - H250(Zn2)
        "Surface_Control": (211, 212, 213), # D254 - A255 - P256
    }

    print("\n[1] Evaluating 6-Body Catalytic Zinc Cage Volume Across Conformations...")
    print(f"{'Structure':<8} {'State Description':<32} {'Log-Det (Volume)':<18} {'Trace (MSF)':<14} {'Mean Corr':<12} {'Cond Num'}")
    print("-" * 92)

    cov_caches = {}
    cliques_3spu = None
    cumulants_3spu = {}

    for pdb_id, desc, path_str in structures:
        p = Path(path_str)
        if not p.exists():
            print(f"Warning: {path_str} missing, skipping {pdb_id}")
            continue

        d = np.load(p)
        if "adj_matrix" in d:
            adj = d["adj_matrix"]
        else:
            adj = np.load("data/processed/structure_3spu_ca.npz")["adj_matrix"]

        gamma = build_kirchhoff_matrix(adj)
        cov = compute_gnm_pseudoinverse(gamma)
        cov_caches[pdb_id] = cov

        # Hexamer metrics
        hex_metrics = extract_catalytic_hexamer_metrics(cov, cat_indices=cat_indices)

        # Allosteric triplets
        triplet_vals = {}
        for name, idxs in triplets.items():
            triplet_vals[name] = compute_multibody_cumulant(cov, idxs)

        results_summary["structures_evaluated"][pdb_id] = {
            "description": desc,
            "hexamer_metrics": hex_metrics,
            "allosteric_triplets": triplet_vals,
        }

        print(
            f"{pdb_id:<8} {desc:<32} "
            f"{hex_metrics['hexamer_logdet']:<18.4f} "
            f"{hex_metrics['hexamer_trace']:<14.4f} "
            f"{hex_metrics['hexamer_mean_corr']:<12.4f} "
            f"{hex_metrics['hexamer_condition']:.2f}"
        )

    print("\n[2] Evaluating 3-Body Allosteric Conduit Cumulants Involving M154...")
    print(f"{'Structure':<8} {'κ(M154,L8,H120)':<18} {'κ(M154,L8,D124)':<18} {'κ(M154,L3,H250)':<18} {'κ(Control)'}")
    print("-" * 80)
    for pdb_id in ["3SPU", "4HL2", "4RL0", "5ZGY", "5O2E"]:
        if pdb_id in results_summary["structures_evaluated"]:
            trip = results_summary["structures_evaluated"][pdb_id]["allosteric_triplets"]
            print(
                f"{pdb_id:<8} "
                f"{trip['M154_L8_H120']:<18.4e} "
                f"{trip['M154_L8_D124']:<18.4e} "
                f"{trip['M154_L3_H250']:<18.4e} "
                f"{trip['Surface_Control']:.4e}"
            )

    print("\n[3] Extracting Structural 3-Cliques for Canonical 3SPU Contact Network...")
    struct_3spu = np.load("data/processed/structure_3spu_ca.npz")
    adj_3spu = struct_3spu["adj_matrix"]
    cliques_3spu = find_contact_cliques(adj_3spu)
    print(f"[+] Total 3-Cliques (Triangles) in 8.0 Å Graph: {len(cliques_3spu)}")

    # Precompute sample cumulants for node pooling
    cov_3spu = cov_caches["3SPU"]
    for clq in cliques_3spu[:200]:
        cumulants_3spu[clq] = compute_multibody_cumulant(cov_3spu, clq)

    hex_3spu = results_summary["structures_evaluated"]["3SPU"]["hexamer_metrics"]
    node_mb_feats = compute_node_multibody_features(
        n_nodes=228,
        cliques=cliques_3spu[:200],
        clique_cumulants=cumulants_3spu,
        cat_metrics=hex_3spu,
        cat_indices=cat_indices,
    )

    # Save feature tensor
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        node_multibody_features=node_mb_feats,
        cliques_count=len(cliques_3spu),
        hexamer_metrics=hex_3spu,
    )
    print(f"[+] Saved node multi-body features tensor of shape {node_mb_feats.shape} to {OUTPUT_NPZ}")
    # Generate and save multi-body features for all evaluated structures
    for pdb_id in ["4HL2", "4RL0", "5ZGY", "5O2E"]:
        if pdb_id in cov_caches:
            cov_s = cov_caches[pdb_id]
            dyn_s = np.load(f"data/processed/dynamics_{pdb_id.lower()}.npz")
            adj_s = dyn_s["adj_matrix"]
            cliques_s = find_contact_cliques(adj_s)
            cumulants_s = {clq: compute_multibody_cumulant(cov_s, clq) for clq in cliques_s[:200]}
            hex_s = results_summary["structures_evaluated"][pdb_id]["hexamer_metrics"]
            node_mb_s = compute_node_multibody_features(
                n_nodes=228,
                cliques=cliques_s[:200],
                clique_cumulants=cumulants_s,
                cat_metrics=hex_s,
                cat_indices=cat_indices,
            )
            out_s = Path(f"data/processed/multibody_{pdb_id.lower()}.npz")
            np.savez_compressed(
                out_s,
                node_multibody_features=node_mb_s,
                cliques_count=len(cliques_s),
                hexamer_metrics=hex_s,
            )
            print(f"[+] Saved {pdb_id} multi-body features tensor of shape {node_mb_s.shape} to {out_s}")

    # Save audit summary
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_JSON, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"[+] Saved audit summary to {AUDIT_JSON}")

    elapsed = time.time() - t0
    print(f"\n[✓] Path 1 Zero-Training Biophysical Validation completed in {elapsed:.2f}s.")
    return results_summary


if __name__ == "__main__":
    run_multibody_biophysical_audit()
