"""Elastic Network Models (GNM and ANM) for NDM-1.

Implements Kirchhoff connectivity matrix, Anisotropic Network Model Hessian,
spectral decomposition, and fluctuation covariance calculation.
"""
from typing import Optional, Tuple

import numpy as np


def build_kirchhoff_matrix(adj_matrix: np.ndarray) -> np.ndarray:
    """Build the N x N Kirchhoff (Laplacian) matrix for the Gaussian Network Model.

    Gamma_ij = -1 if i != j and edge exists
    Gamma_ij = 0 if i != j and no edge
    Gamma_ii = degree(i)

    Args:
        adj_matrix: Binary contact adjacency matrix of shape (N, N) with zero diagonal.

    Returns:
        Kirchhoff matrix Gamma of shape (N, N), float64.
    """
    adj = np.asarray(adj_matrix, dtype=np.float64)
    degrees = adj.sum(axis=1)
    gamma = np.diag(degrees) - adj
    return gamma


def compute_gnm_spectrum(gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors of Kirchhoff matrix.

    Args:
        gamma: Symmetric Kirchhoff matrix (N, N).

    Returns:
        eigvals: Eigenvalues in ascending order, shape (N,).
        eigvecs: Eigenvectors as columns, shape (N, N).
    """
    gamma_f64 = np.asarray(gamma, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(gamma_f64)
    return eigvals, eigvecs


def compute_gnm_pseudoinverse(
    gamma: np.ndarray,
    num_zero_modes: int = 1,
) -> np.ndarray:
    """Compute Moore-Penrose pseudo-inverse covariance matrix Gamma_pinv = Gamma^dagger.

    Omits the rigid-body translational zero mode(s).

    Args:
        gamma: Kirchhoff matrix of shape (N, N).
        num_zero_modes: Number of translational modes to exclude (default 1 for connected graph).

    Returns:
        Pseudo-inverse covariance matrix of shape (N, N), float64.
    """
    eigvals, eigvecs = compute_gnm_spectrum(gamma)
    n = len(eigvals)
    gamma_pinv = np.zeros((n, n), dtype=np.float64)

    for k in range(num_zero_modes, n):
        if eigvals[k] > 1e-10:
            gamma_pinv += (1.0 / eigvals[k]) * np.outer(eigvecs[:, k], eigvecs[:, k])

    return gamma_pinv


def compute_theoretical_msf(gamma_pinv: np.ndarray) -> np.ndarray:
    """Extract theoretical mean-square fluctuations (MSF) per residue.

    < (Delta r_i)^2 > proportional to Gamma_pinv_ii.

    Args:
        gamma_pinv: GNM pseudo-inverse matrix (N, N).

    Returns:
        Array of theoretical fluctuations, shape (N,).
    """
    return np.diag(gamma_pinv).copy()


def build_anm_hessian(
    coords: np.ndarray,
    adj_matrix: np.ndarray,
    dist_matrix: Optional[np.ndarray] = None,
    spring_constant: float = 1.0,
) -> np.ndarray:
    """Build the 3N x 3N Hessian matrix for the Anisotropic Network Model (ANM).

    For i != j with edge (i, j):
        H_ij = - (gamma / D_ij^2) * (r_i - r_j) (r_i - r_j)^T  (3x3 block)
    For i == j:
        H_ii = - sum_{j != i} H_ij  (3x3 block)

    Args:
        coords: C-alpha Cartesian coordinates, shape (N, 3).
        adj_matrix: Contact adjacency matrix, shape (N, N).
        dist_matrix: Pairwise distance matrix, shape (N, N). If None, computed from coords.
        spring_constant: Uniform spring constant gamma (default 1.0).

    Returns:
        ANM Hessian matrix H of shape (3N, 3N), float64.
    """
    coords_f64 = np.asarray(coords, dtype=np.float64)
    n = len(coords_f64)

    if dist_matrix is None:
        diff = coords_f64[:, None, :] - coords_f64[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)

    h_matrix = np.zeros((3 * n, 3 * n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j]:
                r_ij = coords_f64[i] - coords_f64[j]
                d2 = float(dist_matrix[i, j] ** 2)
                if d2 > 1e-12:
                    sub_block = - (spring_constant / d2) * np.outer(r_ij, r_ij)
                    h_matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = sub_block
                    h_matrix[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] = sub_block

    # Compute diagonal 3x3 blocks: H_ii = - sum_{j != i} H_ij
    for i in range(n):
        h_ii = np.zeros((3, 3), dtype=np.float64)
        for j in range(n):
            if i != j:
                h_ii -= h_matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
        h_matrix[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = h_ii

    return h_matrix


def compute_anm_spectrum(h_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors of the ANM Hessian matrix.

    Args:
        h_matrix: ANM Hessian matrix, shape (3N, 3N).

    Returns:
        eigvals: Eigenvalues in ascending order, shape (3N,).
        eigvecs: Eigenvectors as columns, shape (3N, 3N).
    """
    h_f64 = np.asarray(h_matrix, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(h_f64)
    return eigvals, eigvecs


def compute_anm_pseudoinverse(
    h_matrix: np.ndarray,
    num_zero_modes: int = 6,
) -> np.ndarray:
    """Compute Moore-Penrose pseudo-inverse of ANM Hessian H^dagger.

    Omits the 6 rigid-body translational and rotational zero modes.

    Args:
        h_matrix: ANM Hessian matrix of shape (3N, 3N).
        num_zero_modes: Number of rigid-body zero modes to exclude (default 6).

    Returns:
        Pseudo-inverse matrix of shape (3N, 3N), float64.
    """
    eigvals, eigvecs = compute_anm_spectrum(h_matrix)
    total_dim = len(eigvals)
    h_pinv = np.zeros((total_dim, total_dim), dtype=np.float64)

    for k in range(num_zero_modes, total_dim):
        if eigvals[k] > 1e-10:
            h_pinv += (1.0 / eigvals[k]) * np.outer(eigvecs[:, k], eigvecs[:, k])

    return h_pinv
