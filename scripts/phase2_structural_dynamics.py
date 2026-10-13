#!/usr/bin/env python3
"""Phase II: Multi-Conformational Structural Dynamics Engine.

Consolidates Engineering Steps 3 and 11 into a single unified biophysical dynamics engine:
  1. Computes canonical GNM/ANM Kirchhoff spectrum, B-factor correlation, and DCI_asym on PDB 3SPU.
  2. Evaluates substrate-induced conformational dynamics on PDB 4HL2 (Amp-bound), 4RL0 (Mer-bound), 5ZGY (0.95 Å), and 5YD7 (VIM-2).
  3. Audits universal conservation of M154 as an allosteric driver across all NDM-1 structures.
  4. Audits active-site loop clamping and conformational flexibility shifts.
  5. Strictly locks data/processed/dynamics_3spu.npz and saves parallel comparative caches.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from zdyngnn.data.pdb_parser import parse_pdb_structure
from zdyngnn.dynamics.enm import (
    build_anm_hessian,
    build_kirchhoff_matrix,
    compute_anm_pseudoinverse,
    compute_gnm_pseudoinverse,
    compute_gnm_spectrum,
    compute_theoretical_msf,
)
from zdyngnn.dynamics.prs import (
    compute_dci_matrices,
    compute_driver_sensor_scores,
    compute_prs_response_matrix,
)
from zdyngnn.dynamics.topology import (
    assign_topology_bins,
    build_residue_feature_matrix,
    compute_clustering_coefficients,
    compute_laplacian_spectral_features,
)

STRUCTURE_REGISTRY = [
    {
        "pdb_id": "3SPU",
        "name": "Native Apo Reference State",
        "file": "3SPU.pdb",
        "chain_id": "A",
        "res_start": 43,
        "res_end": 270,
        "is_ndm1": True,
        "is_canonical": True,
        "resolution": 1.30,
    },
    {
        "pdb_id": "4HL2",
        "name": "Ampicillin-Bound Holo State",
        "file": "4HL2.pdb",
        "chain_id": "A",
        "res_start": 43,
        "res_end": 270,
        "is_ndm1": True,
        "is_canonical": False,
        "resolution": 1.05,
    },
    {
        "pdb_id": "4RL0",
        "name": "Meropenem-Bound Catalytic Intermediate",
        "file": "4RL0.pdb",
        "chain_id": "B",  # Chain B has all 228 residues modeled
        "res_start": 43,
        "res_end": 270,
        "is_ndm1": True,
        "is_canonical": False,
        "resolution": 1.30,
    },
    {
        "pdb_id": "5ZGY",
        "name": "Sub-Angstrom Resolution Structure at pH 7.5",
        "file": "5ZGY.pdb",
        "chain_id": "A",
        "res_start": 43,
        "res_end": 270,
        "is_ndm1": True,
        "is_canonical": False,
        "resolution": 0.95,
    },
    {
        "pdb_id": "5YD7",
        "name": "VIM-2 Metallo-Beta-Lactamase Homolog",
        "file": "5YD7.pdb",
        "chain_id": "A",
        "res_start": 32,
        "res_end": 262,
        "is_ndm1": False,
        "is_canonical": False,
        "resolution": 1.70,
    },
]

ZINC_COORDINATING_LIGANDS = [
    {"pos": 120, "name": "His120", "role": "Zinc Ligand 1"},
    {"pos": 122, "name": "His122", "role": "Zinc Ligand 2"},
    {"pos": 124, "name": "Asp124", "role": "Zinc Ligand 3"},
    {"pos": 189, "name": "His189", "role": "Zinc Ligand 4"},
    {"pos": 208, "name": "Cys208", "role": "Zinc Ligand 5"},
    {"pos": 250, "name": "His250", "role": "Zinc Ligand 6"},
]


def compute_ca_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute optimal superposition C-alpha Root-Mean-Square Deviation (Kabsch algorithm)."""
    p = coords1 - coords1.mean(axis=0)
    q = coords2 - coords2.mean(axis=0)

    h = p.T @ q
    u, s, vt = np.linalg.svd(h)
    d = np.linalg.det(vt.T @ u.T)
    diag = np.diag([1.0, 1.0, d])
    rot = vt.T @ diag @ u.T

    p_rotated = (rot @ p.T).T
    rmsd = float(np.sqrt(np.mean(np.sum((p_rotated - q) ** 2, axis=1))))
    return rmsd


def run_phase2_pipeline(
    pdb_dir: str = "DMS Data NDM-1/Strutural data",
    output_dir: str = "data/processed",
    results_dir: str = "results",
    distance_cutoff: float = 8.0,
) -> dict:
    """Execute Phase II unified multi-conformational structural dynamics pipeline."""
    pdb_p = Path(pdb_dir)
    out_p = Path(output_dir)
    res_p = Path(results_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    res_p.mkdir(parents=True, exist_ok=True)

    print("=====================================================================")
    print("PHASE II: MULTI-CONFORMATIONAL STRUCTURAL DYNAMICS ENGINE")
    print("=====================================================================")

    # 1. Load Canonical 3SPU reference cache to guarantee it remains locked
    canonical_dyn_path = out_p / "dynamics_3spu.npz"
    canonical_struct_path = out_p / "structure_3spu_ca.npz"

    if not canonical_dyn_path.exists() or not canonical_struct_path.exists():
        raise FileNotFoundError("Canonical 3SPU cache missing. Run scripts/compute_dynamics.py first.")

    canonical_dyn = np.load(canonical_dyn_path)
    canonical_struct = np.load(canonical_struct_path)
    ref_coords = canonical_struct["coords"]
    ref_res_nums = canonical_struct["residue_numbers"]
    ref_dci_asym = canonical_dyn["dci_asym"]
    ref_pos_to_idx = {int(p): i for i, p in enumerate(ref_res_nums)}

    ref_fiedler = float(np.linalg.eigvalsh(canonical_dyn["kirchhoff"])[1])
    ref_msf = canonical_dyn["gnm_msf"]
    ref_b_factors = canonical_struct["b_factors"]
    p_corr = float(np.corrcoef(ref_msf, ref_b_factors)[0, 1])

    print(f"[+] Loaded Canonical 3SPU Reference: Chain A, N={len(ref_coords)} residues.")
    print(f"    GNM Fiedler eigenvalue (lambda_2): {ref_fiedler:.4f}")
    print(f"    Theoretical MSF vs. B-factor Pearson r: {p_corr:.4f}")
    print(f"    Notice: dynamics_3spu.npz is strictly locked and will NOT be overwritten.")

    # Reference M154 couplings to zinc pocket
    idx_154_ref = ref_pos_to_idx[154]
    idx_248_ref = ref_pos_to_idx[248]

    ref_m154_couplings = {}
    for lig in ZINC_COORDINATING_LIGANDS:
        idx_lig = ref_pos_to_idx[lig["pos"]]
        ref_m154_couplings[lig["name"]] = float(ref_dci_asym[idx_lig, idx_154_ref])

    phase2_manifest = {
        "phase": "PHASE II: MULTI-CONFORMATIONAL STRUCTURAL DYNAMICS ENGINE",
        "canonical_reference": {
            "pdb_id": "3SPU",
            "name": "Native Apo Reference State",
            "resolution": 1.30,
            "chain_id": "A",
            "num_residues": len(ref_coords),
            "gnm_fiedler": ref_fiedler,
            "msf_bfactor_pearson_r": p_corr,
            "m154_driver_coupling_to_zinc_pocket": ref_m154_couplings,
            "cache_file": "dynamics_3spu.npz",
            "locked_status": "Strictly Locked & Frozen",
        },
        "comparative_structures": {},
        "conformational_shifts": {},
    }

    # 2. Process all comparative crystal structures
    for spec in STRUCTURE_REGISTRY:
        pdb_id = spec["pdb_id"]
        if spec["is_canonical"]:
            continue  # 3SPU already locked

        pdb_file = pdb_p / spec["file"]
        if not pdb_file.exists():
            print(f"[!] Warning: File {pdb_file} not found. Skipping {pdb_id}.")
            continue

        print(f"\n[*] Processing Comparative Structure: PDB {pdb_id} ({spec['name']})...")
        struct = parse_pdb_structure(
            pdb_path=pdb_file,
            chain_id=spec["chain_id"],
            res_start=spec["res_start"],
            res_end=spec["res_end"],
            distance_cutoff=distance_cutoff,
        )

        n_res = struct.num_residues
        print(f"    Residues: {n_res} (positions {struct.residue_numbers[0]} to {struct.residue_numbers[-1]}).")
        print(f"    Contact degree mean: {struct.degrees.mean():.2f}")

        # Compute GNM
        gamma = build_kirchhoff_matrix(struct.adj_matrix)
        eigvals, eigvecs = compute_gnm_spectrum(gamma)
        fiedler = float(eigvals[1])
        gamma_pinv = compute_gnm_pseudoinverse(gamma, num_zero_modes=1)
        msf = compute_theoretical_msf(gamma_pinv)

        # Compute ANM & PRS
        h_matrix = build_anm_hessian(struct.coords, struct.adj_matrix, struct.dist_matrix)
        s_matrix = compute_prs_response_matrix(struct.coords, struct.adj_matrix, struct.dist_matrix, mode="anm")
        dci, dci_asym = compute_dci_matrices(s_matrix)
        driver_scores, sensor_scores = compute_driver_sensor_scores(dci_asym)

        # Compute clustering and laplacian coordinates
        clustering = compute_clustering_coefficients(struct.adj_matrix)
        lap_eigvals, lap_eigvecs = compute_laplacian_spectral_features(struct.adj_matrix, num_eigenvectors=8)

        # Assemble node features and strata
        node_features = build_residue_feature_matrix(
            degrees=struct.degrees,
            clustering=clustering,
            fluctuations=msf,
            b_factors=struct.b_factors,
            driver_scores=driver_scores,
            sensor_scores=sensor_scores,
            spectral_coords=lap_eigvecs,
        )
        strata_bins = assign_topology_bins(struct.degrees, driver_scores)

        # Save to isolated comparative cache
        out_cache_path = out_p / f"dynamics_{pdb_id.lower()}.npz"
        np.savez_compressed(
            out_cache_path,
            pdb_id=pdb_id,
            chain_id=spec["chain_id"],
            residue_numbers=struct.residue_numbers,
            sequence=struct.sequence,
            coords=struct.coords,
            dist_matrix=struct.dist_matrix,
            adj_matrix=struct.adj_matrix,
            degrees=struct.degrees,
            gnm_fiedler=fiedler,
            gnm_eigvals=eigvals,
            msf=msf,
            anm_response_matrix=s_matrix,
            dci=dci,
            dci_asym=dci_asym,
            driver_scores=driver_scores,
            node_features=node_features,
            strata_bins=strata_bins,
            dci_asym_antisymmetric=bool(np.allclose(dci_asym, -dci_asym.T, atol=1e-8)),
        )
        print(f"[+] Saved isolated comparative cache to {out_cache_path} ({out_cache_path.stat().st_size / 1024:.1f} KB)")

        phase2_manifest["comparative_structures"][pdb_id] = {
            "name": spec["name"],
            "resolution": spec["resolution"],
            "chain_id": spec["chain_id"],
            "num_residues": n_res,
            "mean_degree": float(struct.degrees.mean()),
            "gnm_fiedler": fiedler,
            "cache_file": out_cache_path.name,
        }

        # If NDM-1 structure, evaluate conformational shifts relative to 3SPU
        if spec["is_ndm1"] and n_res == len(ref_coords):
            rmsd = compute_ca_rmsd(ref_coords, struct.coords)
            pos_to_idx_curr = {int(p): i for i, p in enumerate(struct.residue_numbers)}
            idx_154_curr = pos_to_idx_curr[154]
            idx_248_curr = pos_to_idx_curr[248]

            # M154 couplings to zinc pocket
            m154_couplings_curr = {}
            delta_m154_couplings = {}
            for lig in ZINC_COORDINATING_LIGANDS:
                idx_lig = pos_to_idx_curr[lig["pos"]]
                c_val = float(dci_asym[idx_lig, idx_154_curr])
                m154_couplings_curr[lig["name"]] = c_val
                delta_m154_couplings[lig["name"]] = c_val - ref_m154_couplings[lig["name"]]

            # Active-site loop mobility shift: Loop L3 (65-73) and Loop L8 (211-226)
            l3_indices = [pos_to_idx_curr[p] for p in range(65, 74)]
            l8_indices = [pos_to_idx_curr[p] for p in range(211, 227)]

            l3_mean_msf_ref = float(np.mean(canonical_dyn["gnm_msf"][l3_indices]))
            l3_mean_msf_curr = float(np.mean(msf[l3_indices]))
            l8_mean_msf_ref = float(np.mean(canonical_dyn["gnm_msf"][l8_indices]))
            l8_mean_msf_curr = float(np.mean(msf[l8_indices]))

            # Met248 coupling to active site (mean coupling to zinc pocket)
            zinc_indices_curr = [pos_to_idx_curr[lig["pos"]] for lig in ZINC_COORDINATING_LIGANDS]
            zinc_indices_ref = [ref_pos_to_idx[lig["pos"]] for lig in ZINC_COORDINATING_LIGANDS]

            m248_zinc_coupling_ref = float(np.mean([ref_dci_asym[i, idx_248_ref] for i in zinc_indices_ref]))
            m248_zinc_coupling_curr = float(np.mean([dci_asym[i, idx_248_curr] for i in zinc_indices_curr]))

            phase2_manifest["conformational_shifts"][pdb_id] = {
                "ca_rmsd_vs_3spu_angstroms": float(rmsd),
                "loop_l3_msf_shift": {
                    "3spu_apo": l3_mean_msf_ref,
                    "current": l3_mean_msf_curr,
                    "delta": l3_mean_msf_curr - l3_mean_msf_ref,
                },
                "loop_l8_msf_shift": {
                    "3spu_apo": l8_mean_msf_ref,
                    "current": l8_mean_msf_curr,
                    "delta": l8_mean_msf_curr - l8_mean_msf_ref,
                },
                "m154_driver_coupling_to_zinc_pocket": m154_couplings_curr,
                "delta_m154_driver_coupling_vs_3spu": delta_m154_couplings,
                "m248_zinc_pocket_coupling": {
                    "3spu_apo": m248_zinc_coupling_ref,
                    "current": m248_zinc_coupling_curr,
                    "delta": m248_zinc_coupling_curr - m248_zinc_coupling_ref,
                },
            }

            print(f"    C-alpha RMSD vs 3SPU Apo: {rmsd:.3f} Å")
            print(f"    Loop L8 (211-226) Mean MSF: {l8_mean_msf_ref:.4f} (Apo) -> {l8_mean_msf_curr:.4f} ({pdb_id}) [Clamping: {((l8_mean_msf_curr - l8_mean_msf_ref)/l8_mean_msf_ref)*100:+.1f}%]")
            print(f"    M154 Driver Coupling to His122: {ref_m154_couplings['His122']:+.4f} (Apo) -> {m154_couplings_curr['His122']:+.4f} ({pdb_id})")

    # 3. Save Master Phase II Manifest
    manifest_path = res_p / "phase2_structural_dynamics_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(phase2_manifest, f, indent=2)
    print(f"\n[+] Saved Phase II master manifest to {manifest_path}")

    return phase2_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase II Multi-Conformational Structural Dynamics.")
    parser.add_argument("--distance_cutoff", type=float, default=8.0, help="Distance cutoff in Angstroms.")
    args = parser.parse_args()

    run_phase2_pipeline(distance_cutoff=args.distance_cutoff)
