"""Residue Graph Topology Descriptors and Conformal Stratification for NDM-1.

Extracts local clustering coefficients, normalized Laplacian spectra, node feature
matrices, and structural stratification bins S(i) for topology-conditioned conformal prediction.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def compute_clustering_coefficients(adj_matrix: np.ndarray) -> np.ndarray:
    """Compute local clustering coefficients C_i from the contact adjacency matrix.

    C_i = 2 * T_i / (d_i * (d_i - 1)) where T_i = 0.5 * (A^3)_ii is the triangle count.

    Args:
        adj_matrix: Binary contact adjacency matrix of shape (N, N).

    Returns:
        clustering: Local clustering coefficients, shape (N,), values in [0, 1].
    """
    adj = np.asarray(adj_matrix, dtype=np.float64)
    degrees = adj.sum(axis=1)

    # A^2 and A^3
    a2 = adj @ adj
    triangles = 0.5 * np.diag(a2 @ adj)

    clustering = np.zeros_like(degrees, dtype=np.float64)
    mask = degrees >= 2
    clustering[mask] = (2.0 * triangles[mask]) / (degrees[mask] * (degrees[mask] - 1.0))

    return np.clip(clustering, 0.0, 1.0)


def compute_laplacian_spectral_features(
    adj_matrix: np.ndarray,
    num_eigenvectors: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and diffusion eigenvectors of normalized Laplacian L_sym.

    L_sym = I - D^(-1/2) A D^(-1/2)

    Args:
        adj_matrix: Binary contact adjacency matrix, shape (N, N).
        num_eigenvectors: Number of smallest non-trivial eigenvectors to return.

    Returns:
        eigvals: Sorted Laplacian eigenvalues, shape (N,).
        eigvecs: Selected eigenvectors of shape (N, num_eigenvectors).
    """
    adj = np.asarray(adj_matrix, dtype=np.float64)
    degrees = adj.sum(axis=1)
    degrees_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    d_inv_sqrt = np.diag(degrees_inv_sqrt)

    # L_sym = I - D^(-1/2) A D^(-1/2)
    l_sym = np.eye(len(adj)) - d_inv_sqrt @ adj @ d_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(l_sym)
    eigvals = np.clip(eigvals, 0.0, 2.0)

    k = min(num_eigenvectors, len(eigvals))
    selected_vecs = eigvecs[:, :k]

    return eigvals, selected_vecs


def build_residue_feature_matrix(
    degrees: np.ndarray,
    clustering: np.ndarray,
    fluctuations: np.ndarray,
    b_factors: np.ndarray,
    driver_scores: np.ndarray,
    sensor_scores: np.ndarray,
    spectral_coords: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Assemble standardized node-level physical and topological feature matrix.

    Features included per residue:
        1. Normalized degree: d_i / max(d)
        2. Clustering coefficient: C_i
        3. Normalized theoretical fluctuation: MSF_i / max(MSF)
        4. Normalized B-factor: B_i / max(B)
        5. PRS driver score: Driver(i)
        6. PRS sensor score: Sensor(i)
        7..7+k. Positional spectral coordinates (if provided)

    Args:
        degrees: Coordination numbers, shape (N,).
        clustering: Clustering coefficients, shape (N,).
        fluctuations: Theoretical MSF fluctuations from GNM, shape (N,).
        b_factors: Experimental B-factors, shape (N,).
        driver_scores: PRS driver propensity, shape (N,).
        sensor_scores: PRS sensor propensity, shape (N,).
        spectral_coords: Laplacian eigenvectors, shape (N, k).

    Returns:
        Feature matrix X of shape (N, F), float32.
    """
    n = len(degrees)
    feat_list = []

    # 1. Degree
    d_max = float(degrees.max()) if degrees.max() > 0 else 1.0
    feat_list.append((degrees / d_max).reshape(n, 1))

    # 2. Clustering
    feat_list.append(clustering.reshape(n, 1))

    # 3. Fluctuation MSF
    fl_max = float(fluctuations.max()) if fluctuations.max() > 0 else 1.0
    feat_list.append((fluctuations / fl_max).reshape(n, 1))

    # 4. B-factor
    bf_max = float(b_factors.max()) if b_factors.max() > 0 else 1.0
    feat_list.append((b_factors / bf_max).reshape(n, 1))

    # 5. Driver score
    feat_list.append(driver_scores.reshape(n, 1))

    # 6. Sensor score
    feat_list.append(sensor_scores.reshape(n, 1))

    # 7. Positional spectral coords
    if spectral_coords is not None:
        feat_list.append(spectral_coords)

    x_features = np.hstack(feat_list).astype(np.float32)
    return x_features


def assign_topology_bins(
    degrees: np.ndarray,
    driver_scores: np.ndarray,
    num_degree_bins: int = 4,
    num_dynamics_bins: int = 3,
) -> np.ndarray:
    """Discretize residues into structural stratification bins S(i) for conformal calibration.

    Binning strategy:
        - Dimension 1 (Packing): Quantile bins of contact degree d_i (e.g., surface/loop, intermediate, core, dense core).
        - Dimension 2 (Dynamics): Quantile bins of PRS driver score (sensor, neutral, driver).
        Total bins = num_degree_bins * num_dynamics_bins.

    Args:
        degrees: Coordination degree array, shape (N,).
        driver_scores: PRS allosteric driver score array, shape (N,).
        num_degree_bins: Number of quantiles for contact degree (default 4).
        num_dynamics_bins: Number of quantiles for dynamics (default 3).

    Returns:
        bin_indices: Integer array of shape (N,), values in [0, num_degree_bins * num_dynamics_bins - 1].
    """
    # Degree binning
    deg_quantiles = np.linspace(0, 100, num_degree_bins + 1)[1:-1]
    deg_thresholds = np.percentile(degrees, deg_quantiles)
    deg_bins = np.digitize(degrees, deg_thresholds)  # Values in 0..num_degree_bins-1

    # Dynamics binning
    dyn_quantiles = np.linspace(0, 100, num_dynamics_bins + 1)[1:-1]
    dyn_thresholds = np.percentile(driver_scores, dyn_quantiles)
    dyn_bins = np.digitize(driver_scores, dyn_thresholds)  # Values in 0..num_dynamics_bins-1

    # 2D flat bin index: S(i) = deg_bin * num_dynamics_bins + dyn_bin
    strata = deg_bins * num_dynamics_bins + dyn_bins
    return strata.astype(np.int32)
