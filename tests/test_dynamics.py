"""Unit and integration tests for Step 3 Graph Topology & Dynamics."""
from pathlib import Path

import numpy as np
import pytest

from zdyngnn.dynamics.enm import (
    build_anm_hessian,
    build_kirchhoff_matrix,
    compute_anm_pseudoinverse,
    compute_anm_spectrum,
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

STRUCT_CACHE = "data/processed/structure_3spu_ca.npz"


@pytest.fixture
def structure_data():
    """Load cached 3SPU C-alpha coordinates and adjacency."""
    assert Path(STRUCT_CACHE).exists(), "Run scripts/preprocess_data.py first."
    return np.load(STRUCT_CACHE)


def test_enm_kirchhoff_properties(structure_data):
    """Verify mathematical invariants of the GNM Kirchhoff matrix."""
    adj = structure_data["adj_matrix"]
    b_factors = structure_data["b_factors"]
    n = len(adj)

    gamma = build_kirchhoff_matrix(adj)
    assert gamma.shape == (n, n)

    # Invariant 1: Symmetry
    assert np.allclose(gamma, gamma.T)

    # Invariant 2: Row & Column sums are zero (null space contains 1)
    assert np.allclose(gamma.sum(axis=1), 0.0, atol=1e-10)
    assert np.allclose(gamma.sum(axis=0), 0.0, atol=1e-10)

    # Invariant 3: Single zero eigenvalue (graph is connected)
    eigvals, eigvecs = compute_gnm_spectrum(gamma)
    assert np.isclose(eigvals[0], 0.0, atol=1e-10)
    assert eigvals[1] > 0.1  # Fiedler eigenvalue is strictly positive

    # Invariant 4: Pseudo-inverse projection identity Gamma_pinv @ Gamma = I - J / N
    gamma_pinv = compute_gnm_pseudoinverse(gamma, num_zero_modes=1)
    identity_approx = gamma_pinv @ gamma
    expected_proj = np.eye(n) - (np.ones((n, n)) / n)
    assert np.allclose(identity_approx, expected_proj, atol=1e-8)

    # Invariant 5: Theoretical MSF correlation with crystallographic B-factors > 0.60
    msf = compute_theoretical_msf(gamma_pinv)
    assert (msf > 0).all()
    corr = np.corrcoef(msf, b_factors)[0, 1]
    assert corr > 0.60


def test_anm_hessian_and_prs(structure_data):
    """Verify ANM Hessian degrees of freedom, PRS response, and DCI_asym antisymmetry."""
    coords = structure_data["coords"]
    adj = structure_data["adj_matrix"]
    dist = structure_data["dist_matrix"]
    n = len(coords)

    # ANM Hessian structure
    h_matrix = build_anm_hessian(coords, adj, dist)
    assert h_matrix.shape == (3 * n, 3 * n)
    assert np.allclose(h_matrix, h_matrix.T)

    # Check 6 rigid-body zero modes
    eigvals, _ = compute_anm_spectrum(h_matrix)
    for k in range(6):
        assert np.isclose(eigvals[k], 0.0, atol=1e-8)
    assert eigvals[6] > 1e-4

    # PRS response matrix
    prs_s = compute_prs_response_matrix(coords, adj, dist, mode="anm")
    assert prs_s.shape == (n, n)
    assert (prs_s >= 0).all()

    # Dynamic coupling index & asymmetry
    dci, dci_asym = compute_dci_matrices(prs_s)
    assert dci.shape == (n, n)
    assert dci_asym.shape == (n, n)

    # Invariant: DCI_asym is strictly antisymmetric
    assert np.allclose(dci_asym, -dci_asym.T, atol=1e-12)
    assert np.allclose(np.diag(dci_asym), 0.0)

    # Driver and sensor score duality
    driver_scores, sensor_scores = compute_driver_sensor_scores(dci_asym)
    assert np.allclose(sensor_scores, -driver_scores)
    assert np.isclose(driver_scores.mean(), 0.0, atol=1e-10)


def test_topology_descriptors_and_strata(structure_data):
    """Verify local clustering coefficients, spectral coordinates, and conformal strata."""
    adj = structure_data["adj_matrix"]
    degrees = structure_data["degrees"]
    b_factors = structure_data["b_factors"]
    n = len(adj)

    # Clustering coefficient
    clustering = compute_clustering_coefficients(adj)
    assert clustering.shape == (n,)
    assert (clustering >= 0.0).all()
    assert (clustering <= 1.0).all()
    assert 0.40 <= clustering.mean() <= 0.70

    # Spectral coordinates
    eigvals, eigvecs = compute_laplacian_spectral_features(adj, num_eigenvectors=6)
    assert eigvecs.shape == (n, 6)
    assert (eigvals >= 0.0).all()
    assert (eigvals <= 2.0).all()

    # Synthetic mock driver scores
    mock_driver = np.random.randn(n)
    mock_sensor = -mock_driver
    mock_msf = np.ones(n)

    node_features = build_residue_feature_matrix(
        degrees=degrees,
        clustering=clustering,
        fluctuations=mock_msf,
        b_factors=b_factors,
        driver_scores=mock_driver,
        sensor_scores=mock_sensor,
        spectral_coords=eigvecs,
    )
    assert node_features.shape == (n, 6 + 6)
    assert not np.isnan(node_features).any()

    # Conformal strata binning
    strata = assign_topology_bins(degrees, mock_driver, num_degree_bins=4, num_dynamics_bins=3)
    assert strata.shape == (n,)
    assert (strata >= 0).all()
    assert (strata < 12).all()
    assert len(np.unique(strata)) >= 8  # Sufficient structural diversity across bins


def test_dynamics_cache_integrity():
    """Verify that the generated dynamics cache data/processed/dynamics_3spu.npz is complete."""
    cache_path = Path("data/processed/dynamics_3spu.npz")
    assert cache_path.exists(), "Cache file must exist after scripts/compute_dynamics.py."

    dyn = np.load(cache_path)
    required_keys = [
        "kirchhoff",
        "gnm_covariance",
        "gnm_msf",
        "prs_response",
        "dci",
        "dci_asym",
        "driver_scores",
        "sensor_scores",
        "clustering",
        "node_features",
        "strata_bins",
        "residue_numbers",
    ]
    for key in required_keys:
        assert key in dyn, f"Key '{key}' missing from dynamics cache."

    assert dyn["kirchhoff"].shape == (228, 228)
    assert dyn["dci_asym"].shape == (228, 228)
    assert dyn["node_features"].shape == (228, 14)
    assert len(dyn["strata_bins"]) == 228
