"""Unit and integration tests for Step 6 Topology-Conditioned Conformal Calibration."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zdyngnn.conformal.calibration import (
    ConformalCertificate,
    calibrate_cluster_conformal,
    calibrate_split_conformal,
    compute_conformal_quantile,
    compute_studentized_residuals,
    compute_winkler_score,
)
from zdyngnn.conformal.stratified import (
    StratifiedConformalResult,
    TopologyConditionedConformalCalibrator,
)


def test_studentized_residuals_and_quantiles():
    """Verify non-conformity score calculation and exact finite-sample quantile formula."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    mu = np.array([1.0, 1.5, 3.5, 2.0])
    sigma = np.array([1.0, 0.5, 0.5, 1.0])

    r = compute_studentized_residuals(y, mu, sigma)
    # R = [0/1=0, 0.5/0.5=1, 0.5/0.5=1, 2/1=2]
    assert np.allclose(r, [0.0, 1.0, 1.0, 2.0])

    # Quantile test: m=4, alpha=0.25 -> (4+1)*0.75 / 4 = 3.75 -> ceil = 4 -> level = 4/4 = 1.0
    q = compute_conformal_quantile(r, alpha=0.25)
    assert q == 2.0


def test_conformal_coverage_guarantee_synthetic():
    """Verify finite-sample coverage guarantee on exchangeable synthetic distributions."""
    np.random.seed(42)
    m_calib = 500
    m_test = 1000

    # Non-Gaussian, heavy-tailed Student-t residuals
    from scipy.stats import t
    calib_res = np.abs(t.rvs(df=3, size=m_calib))
    test_res = np.abs(t.rvs(df=3, size=m_test))

    alpha = 0.05
    q_hat = compute_conformal_quantile(calib_res, alpha=alpha)

    covered = test_res <= q_hat
    empirical_cov = np.mean(covered)

    # In exchangeable finite samples, empirical coverage must be close to or above 1 - alpha (>= 94%)
    assert empirical_cov >= (1.0 - alpha - 0.02)


def test_split_and_cluster_conformal_calibration():
    """Verify split and cluster-adjusted calibration routines and interval formatting."""
    calib_df = pd.DataFrame({
        "pos": [43, 43, 44, 44, 45, 45],
        "fitness_score": [-1.0, -1.2, -2.0, -1.8, 0.5, 0.6],
        "pred_mu": [-1.1, -1.1, -1.9, -1.9, 0.4, 0.4],
        "pred_sigma": [0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
    })
    test_df = pd.DataFrame({
        "pos": [46, 46, 47, 47],
        "fitness_score": [-1.5, -1.4, 0.2, 0.3],
        "pred_mu": [-1.3, -1.3, 0.1, 0.1],
        "pred_sigma": [0.5, 0.5, 0.5, 0.5],
    })

    # Standard split conformal
    cert_split, bounds_split = calibrate_split_conformal(
        calib_y=calib_df["fitness_score"].values,
        calib_mu=calib_df["pred_mu"].values,
        calib_sigma=calib_df["pred_sigma"].values,
        test_y=test_df["fitness_score"].values,
        test_mu=test_df["pred_mu"].values,
        test_sigma=test_df["pred_sigma"].values,
        alpha=0.10,
    )

    assert cert_split.quantile > 0.0
    assert len(bounds_split) == 4
    assert (bounds_split["upper_bound"] >= bounds_split["lower_bound"]).all()
    assert (bounds_split["interval_width"] > 0).all()

    # Cluster-adjusted conformal
    cert_clust, bounds_clust = calibrate_cluster_conformal(
        calib_df=calib_df,
        test_df=test_df,
        alpha=0.10,
        agg_quantile=0.80,
    )
    assert cert_clust.num_calib_samples == 3  # 3 distinct positions


def test_topology_conditioned_calibrator_stratification():
    """Verify stratum-specific quantiles and hierarchical fallback."""
    calib_df = pd.DataFrame({
        "strata": [1] * 20 + [2] * 5,  # Stratum 1 has 20 samples; stratum 2 has 5 (below min)
        "fitness_score": np.random.randn(25),
        "pred_mu": np.random.randn(25),
        "pred_sigma": np.ones(25),
    })
    test_df = pd.DataFrame({
        "strata": [1, 1, 2, 2],
        "fitness_score": [0.1, -0.2, 0.5, -0.4],
        "pred_mu": [0.0, 0.0, 0.0, 0.0],
        "pred_sigma": [1.0, 1.0, 1.0, 1.0],
    })

    calibrator = TopologyConditionedConformalCalibrator(alpha=0.05, min_stratum_samples=10)
    calibrator.fit(calib_df)

    # Stratum 1 should have its own quantile; Stratum 2 should fallback to global
    assert 1 in calibrator.stratum_quantiles
    assert 2 in calibrator.stratum_quantiles
    assert calibrator.stratum_quantiles[2] == calibrator.global_quantile

    res = calibrator.predict_intervals(test_df)
    assert len(res.bounds_df) == 4
    assert 1 in res.stratum_coverage
    assert 2 in res.stratum_coverage


def test_conformal_artifacts_on_disk():
    """Verify generated conformal certificate files in results/."""
    bench_path = Path("results/conformal_benchmarks.csv")
    cert_amp = Path("results/conformal_certificates_256.0_AMP.csv")

    assert bench_path.exists(), "conformal_benchmarks.csv must exist."
    assert cert_amp.exists(), "conformal_certificates_256.0_AMP.csv must exist."

    df_cert = pd.read_csv(cert_amp)
    assert len(df_cert) == 612  # All 612 held-out test mutations
    assert "conformal_quantile" in df_cert.columns
    assert "certified_bound" in df_cert.columns
    assert "lower_bound" in df_cert.columns
    assert "upper_bound" in df_cert.columns
    assert "covered" in df_cert.columns

    # Verify no NaN intervals
    assert not df_cert["lower_bound"].isna().any()
    assert not df_cert["upper_bound"].isna().any()
    assert (df_cert["upper_bound"] >= df_cert["lower_bound"]).all()
