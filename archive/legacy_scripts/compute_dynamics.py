#!/usr/bin/env python3
"""CLI script to compute biophysical dynamics (GNM, ANM, PRS, DCI_asym) and topology descriptors."""
import argparse
import json
from pathlib import Path

import numpy as np

from zdyngnn.dynamics.enm import (
    build_kirchhoff_matrix,
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


def compute_and_cache_dynamics(
    structure_cache_path: str = "data/processed/structure_3spu_ca.npz",
    output_dir: str = "data/processed",
) -> dict:
    """Compute all dynamics matrices and export to dynamics_3spu.npz."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading crystal structure from {structure_cache_path}...")
    struct = np.load(structure_cache_path)
    coords = struct["coords"]
    adj = struct["adj_matrix"]
    dist = struct["dist_matrix"]
    degrees = struct["degrees"]
    b_factors = struct["b_factors"]
    res_nums = struct["residue_numbers"]
    n_res = len(coords)

    print(f"    Residues: {n_res} (range {res_nums.min()} to {res_nums.max()})")

    # 1. GNM Kirchhoff & Fluctuation Covariance
    print("[*] Computing Gaussian Network Model (GNM) Kirchhoff spectrum & covariance...")
    gamma = build_kirchhoff_matrix(adj)
    eigvals_gnm, _ = compute_gnm_spectrum(gamma)
    gamma_pinv = compute_gnm_pseudoinverse(gamma, num_zero_modes=1)
    msf = compute_theoretical_msf(gamma_pinv)

    b_corr = float(np.corrcoef(msf, b_factors)[0, 1])
    print(f"    GNM Fiedler eigenvalue (lambda_2): {eigvals_gnm[1]:.4f}")
    print(f"    Pearson correlation (MSF vs B-factors): {b_corr:.4f}")

    # 2. ANM Perturbation Response Scanning & DCI_asym
    print("[*] Computing Anisotropic Network Model (ANM) & Perturbation Response Scanning...")
    prs_response = compute_prs_response_matrix(coords, adj, dist, mode="anm")
    dci, dci_asym = compute_dci_matrices(prs_response)
    driver_scores, sensor_scores = compute_driver_sensor_scores(dci_asym)

    print(f"    DCI_asym matrix shape: {dci_asym.shape}")
    print(f"    Antisymmetry verified: {bool(np.allclose(dci_asym, -dci_asym.T))}")
    print(f"    Driver scores: min={driver_scores.min():.4f}, max={driver_scores.max():.4f}, std={driver_scores.std():.4f}")

    # 3. Graph Topology & Spectral Embeddings
    print("[*] Computing graph clustering and Laplacian spectral coordinates...")
    clustering = compute_clustering_coefficients(adj)
    _, spectral_vecs = compute_laplacian_spectral_features(adj, num_eigenvectors=8)

    print(f"    Mean clustering coefficient: {clustering.mean():.4f}")

    # 4. Standardized Node Features & Conformal Stratification Bins
    print("[*] Assembling residue feature matrix and conformal stratification strata...")
    node_features = build_residue_feature_matrix(
        degrees=degrees,
        clustering=clustering,
        fluctuations=msf,
        b_factors=b_factors,
        driver_scores=driver_scores,
        sensor_scores=sensor_scores,
        spectral_coords=spectral_vecs,
    )

    strata_bins = assign_topology_bins(
        degrees=degrees,
        driver_scores=driver_scores,
        num_degree_bins=4,
        num_dynamics_bins=3,
    )

    # 5. Export to NPZ Cache
    npz_out = out_path / "dynamics_3spu.npz"
    np.savez_compressed(
        npz_out,
        kirchhoff=gamma,
        gnm_covariance=gamma_pinv,
        gnm_msf=msf,
        prs_response=prs_response,
        dci=dci,
        dci_asym=dci_asym,
        driver_scores=driver_scores,
        sensor_scores=sensor_scores,
        clustering=clustering,
        node_features=node_features,
        strata_bins=strata_bins,
        residue_numbers=res_nums,
    )
    print(f"[+] Saved dynamics cache to {npz_out} ({npz_out.stat().st_size / 1024:.1f} KB)")

    summary = {
        "num_residues": int(n_res),
        "residue_range": [int(res_nums.min()), int(res_nums.max())],
        "gnm_fiedler_val": float(eigvals_gnm[1]),
        "msf_bfactor_correlation": float(b_corr),
        "mean_clustering": float(clustering.mean()),
        "dci_asym_antisymmetric": bool(np.allclose(dci_asym, -dci_asym.T)),
        "driver_score_std": float(driver_scores.std()),
        "node_features_dim": list(node_features.shape),
        "num_conformal_strata": int(len(np.unique(strata_bins))),
    }

    summary_json = out_path / "dynamics_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compute biophysical dynamics and topology for NDM-1.")
    parser.add_argument("--struct", default="data/processed/structure_3spu_ca.npz", help="Path to parsed structure npz.")
    parser.add_argument("--out", default="data/processed", help="Output directory.")
    args = parser.parse_args()

    summary = compute_and_cache_dynamics(args.struct, args.out)
    print("\n[+] Step 3 Dynamics Computation Complete!")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
