"""Perturbation Response Scanning (PRS) and Dynamic Coupling Index (DCI) for NDM-1.

Computes linear response displacement matrices under isotropic force perturbations,
yielding the row-normalized Dynamic Coupling Index (DCI) and the directional
Asymmetric Dynamic Coupling Index (DCI_asym).
"""
from typing import Optional, Tuple

import numpy as np

from zdyngnn.dynamics.enm import (
    build_anm_hessian,
    build_kirchhoff_matrix,
    compute_anm_pseudoinverse,
    compute_gnm_pseudoinverse,
)


def compute_prs_response_matrix(
    coords: np.ndarray,
    adj_matrix: np.ndarray,
    dist_matrix: Optional[np.ndarray] = None,
    mode: str = "anm",
) -> np.ndarray:
    """Compute the N x N displacement response amplitude matrix S_ij under unit force perturbations.

    S_ij represents the root-mean-square displacement of residue i when residue j is perturbed.

    Args:
        coords: C-alpha coordinates of shape (N, 3).
        adj_matrix: Contact adjacency matrix of shape (N, N).
        dist_matrix: Pairwise distance matrix of shape (N, N).
        mode: Elastic network model used: 'anm' (3D directional) or 'gnm' (isotropic scalar).

    Returns:
        Response matrix S of shape (N, N), float64.
    """
    coords_f64 = np.asarray(coords, dtype=np.float64)
    n = len(coords_f64)

    if mode == "anm":
        h_matrix = build_anm_hessian(coords_f64, adj_matrix, dist_matrix)
        h_pinv = compute_anm_pseudoinverse(h_matrix, num_zero_modes=6)

        s_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                block = h_pinv[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
                # S_ij = sqrt(Tr(block @ block^T) / 3)
                trace_val = float(np.trace(block @ block.T))
                s_matrix[i, j] = np.sqrt(max(0.0, trace_val / 3.0))

    elif mode == "gnm":
        gamma = build_kirchhoff_matrix(adj_matrix)
        gamma_pinv = compute_gnm_pseudoinverse(gamma, num_zero_modes=1)
        s_matrix = np.abs(gamma_pinv).copy()

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'anm' or 'gnm'.")

    return s_matrix


def compute_dci_matrices(response_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the normalized Dynamic Coupling Index (DCI) and asymmetric coupling (DCI_asym).

    Mathematical definition:
        DCI_ij = S_ij / ( (1 / N) * sum_k S_ik )
        DCI_asym(i, j) = DCI_ij - DCI_ji

    Properties:
        - DCI_asym is strictly antisymmetric: DCI_asym^T = -DCI_asym
        - Diagonals are identically zero: DCI_asym(i, i) = 0
        - DCI_asym(i, j) > 0 indicates residue j is an allosteric driver affecting residue i.

    Args:
        response_matrix: PRS displacement response matrix S of shape (N, N).

    Returns:
        dci: Row-normalized coupling matrix of shape (N, N).
        dci_asym: Antisymmetric directional dynamic coupling matrix of shape (N, N).
    """
    s_mat = np.asarray(response_matrix, dtype=np.float64)
    n = len(s_mat)

    # Average displacement response of residue i over all perturbations
    row_means = s_mat.mean(axis=1, keepdims=True)
    # Avoid division by zero
    row_means = np.where(row_means == 0, 1.0, row_means)

    dci = s_mat / row_means
    dci_asym = dci - dci.T

    # Exact zero diagonal enforcement
    np.fill_diagonal(dci_asym, 0.0)

    return dci, dci_asym


def compute_driver_sensor_scores(dci_asym: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract net allosteric driver and sensor propensity scores per residue.

    Driver(j) = (1 / (N - 1)) * sum_{i != j} DCI_asym(i, j)
    Sensor(i) = (1 / (N - 1)) * sum_{j != i} DCI_asym(j, i) = -Driver(i)

    Args:
        dci_asym: Antisymmetric DCI_asym matrix of shape (N, N).

    Returns:
        driver_scores: Array of shape (N,), positive values indicate driving source.
        sensor_scores: Array of shape (N,), positive values indicate receptive sensor.
    """
    dci_asym_f64 = np.asarray(dci_asym, dtype=np.float64)
    n = len(dci_asym_f64)
    scale = 1.0 / max(1, n - 1)

    # DCI_asym(i, j) is the effect of j on i minus i on j.
    # Sum over i gives how much residue j drives all others.
    driver_scores = (dci_asym_f64.sum(axis=0) * scale).astype(np.float64)
    sensor_scores = -driver_scores

    return driver_scores, sensor_scores
