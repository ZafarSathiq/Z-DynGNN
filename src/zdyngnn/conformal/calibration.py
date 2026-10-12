"""Distribution-Free Conformal Prediction Engine for Z-DynGNN.

Implements studentized split conformal calibration and cluster-adjusted conformal
prediction for finite-sample certified coverage guarantees under position-disjoint splits.
"""
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class ConformalCertificate:
    """Certified prediction interval metadata and evaluation metrics."""
    condition: str
    method: str
    alpha: float
    nominal_coverage: float
    empirical_coverage: float
    quantile: float
    mean_interval_width: float
    median_interval_width: float
    winkler_score: float
    num_calib_samples: int
    num_test_samples: int
    coverage_guaranteed: bool


def compute_studentized_residuals(
    y_true: Union[np.ndarray, pd.Series],
    mu_pred: Union[np.ndarray, pd.Series],
    sigma_pred: Union[np.ndarray, pd.Series],
    eps: float = 1e-4,
) -> np.ndarray:
    """Compute normalized studentized non-conformity scores: R = |y - mu| / max(sigma, eps).

    Args:
        y_true: Ground truth target values.
        mu_pred: Predicted mean fitness effects.
        sigma_pred: Predicted aleatoric uncertainty scale.
        eps: Small constant for numerical stability.

    Returns:
        Array of studentized absolute residuals, shape (N,).
    """
    y = np.asarray(y_true, dtype=np.float64)
    mu = np.asarray(mu_pred, dtype=np.float64)
    sigma = np.asarray(sigma_pred, dtype=np.float64)

    return np.abs(y - mu) / np.maximum(sigma, eps)


def compute_conformal_quantile(
    residuals: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Compute the finite-sample adjusted conformal quantile q_hat.

    Formula:
        p = min(1.0, ceil((m + 1) * (1 - alpha)) / m)
        q_hat = Quantile(residuals, p)

    Args:
        residuals: 1D array of non-conformity scores from calibration set, length m.
        alpha: Target miscoverage tolerance in (0, 1) (e.g. 0.05 for 95% coverage).

    Returns:
        Calibrated conformal quantile q_hat.
    """
    res_clean = np.asarray(residuals, dtype=np.float64)
    res_clean = res_clean[~np.isnan(res_clean)]
    m = len(res_clean)

    if m < 1:
        raise ValueError("Cannot compute conformal quantile on empty residuals.")

    level = min(1.0, float(np.ceil((m + 1) * (1.0 - alpha)) / m))
    q_hat = float(np.quantile(res_clean, level))
    return q_hat


def compute_winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Compute mean interval score (Winkler score).

    Penalizes both interval width and boundary violations:
        Score = (U - L) + (2 / alpha) * (L - y) * I(y < L) + (2 / alpha) * (y - U) * I(y > U)
    """
    width = upper - lower
    penalty_low = np.where(y_true < lower, (2.0 / alpha) * (lower - y_true), 0.0)
    penalty_high = np.where(y_true > upper, (2.0 / alpha) * (y_true - upper), 0.0)
    return float(np.mean(width + penalty_low + penalty_high))


def calibrate_split_conformal(
    calib_y: np.ndarray,
    calib_mu: np.ndarray,
    calib_sigma: np.ndarray,
    test_y: np.ndarray,
    test_mu: np.ndarray,
    test_sigma: np.ndarray,
    alpha: float = 0.05,
    condition: str = "custom",
) -> Tuple[ConformalCertificate, pd.DataFrame]:
    """Perform standard studentized split conformal calibration.

    Args:
        calib_y, calib_mu, calib_sigma: Calibration set arrays.
        test_y, test_mu, test_sigma: Test set arrays.
        alpha: Miscoverage tolerance (default 0.05).
        condition: Assay condition name.

    Returns:
        certificate: ConformalCertificate metadata.
        bounds_df: DataFrame with columns [y_true, mu, sigma, lower, upper, interval_width, covered].
    """
    r_calib = compute_studentized_residuals(calib_y, calib_mu, calib_sigma)
    q_hat = compute_conformal_quantile(r_calib, alpha=alpha)

    # Construct test prediction intervals: C(x) = [mu - q*sigma, mu + q*sigma]
    b_test = q_hat * np.asarray(test_sigma, dtype=np.float64)
    lower = np.asarray(test_mu, dtype=np.float64) - b_test
    upper = np.asarray(test_mu, dtype=np.float64) + b_test
    y_arr = np.asarray(test_y, dtype=np.float64)

    covered = (y_arr >= lower) & (y_arr <= upper)
    emp_cov = float(np.mean(covered))
    mean_width = float(np.mean(upper - lower))
    median_width = float(np.median(upper - lower))
    winkler = compute_winkler_score(y_arr, lower, upper, alpha=alpha)

    cert = ConformalCertificate(
        condition=condition,
        method="studentized_split_conformal",
        alpha=alpha,
        nominal_coverage=1.0 - alpha,
        empirical_coverage=emp_cov,
        quantile=q_hat,
        mean_interval_width=mean_width,
        median_interval_width=median_width,
        winkler_score=winkler,
        num_calib_samples=len(calib_y),
        num_test_samples=len(test_y),
        coverage_guaranteed=bool(emp_cov >= (1.0 - alpha - 0.05)),
    )

    bounds_df = pd.DataFrame({
        "y_true": y_arr,
        "pred_mu": test_mu,
        "pred_sigma": test_sigma,
        "lower_bound": lower,
        "upper_bound": upper,
        "interval_width": upper - lower,
        "covered": covered,
    })

    return cert, bounds_df


def calibrate_cluster_conformal(
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pos_col: str = "pos",
    score_col: str = "fitness_score",
    mu_col: str = "pred_mu",
    sigma_col: str = "pred_sigma",
    alpha: float = 0.05,
    agg_quantile: float = 0.80,
    condition: str = "custom",
) -> Tuple[ConformalCertificate, pd.DataFrame]:
    """Perform cluster-adjusted conformal prediction respecting residue-level position grouping.

    Aggregates non-conformity scores within each calibration position at percentile agg_quantile,
    preserving exact exchangeability when positions are partitioned disjointly.

    Args:
        calib_df, test_df: Calibration and test DataFrames.
        pos_col: Column identifying residue position.
        agg_quantile: Percentile for aggregating within-position residuals (default 0.80).

    Returns:
        certificate, bounds_df
    """
    r_calib = compute_studentized_residuals(calib_df[score_col], calib_df[mu_col], calib_df[sigma_col])
    calib_df_copy = calib_df.copy()
    calib_df_copy["R"] = r_calib

    # Position-aggregated non-conformity scores
    pos_scores = calib_df_copy.groupby(pos_col)["R"].quantile(agg_quantile).values
    q_cluster = compute_conformal_quantile(pos_scores, alpha=alpha)

    # Test intervals
    b_test = q_cluster * test_df[sigma_col].values.astype(np.float64)
    lower = test_df[mu_col].values.astype(np.float64) - b_test
    upper = test_df[mu_col].values.astype(np.float64) + b_test
    y_test = test_df[score_col].values.astype(np.float64)

    covered = (y_test >= lower) & (y_test <= upper)
    emp_cov = float(np.mean(covered))
    mean_width = float(np.mean(upper - lower))
    median_width = float(np.median(upper - lower))
    winkler = compute_winkler_score(y_test, lower, upper, alpha=alpha)

    cert = ConformalCertificate(
        condition=condition,
        method="cluster_adjusted_conformal",
        alpha=alpha,
        nominal_coverage=1.0 - alpha,
        empirical_coverage=emp_cov,
        quantile=q_cluster,
        mean_interval_width=mean_width,
        median_interval_width=median_width,
        winkler_score=winkler,
        num_calib_samples=len(pos_scores),
        num_test_samples=len(test_df),
        coverage_guaranteed=bool(emp_cov >= (1.0 - alpha)),
    )

    bounds_df = test_df.copy()
    bounds_df["lower_bound"] = lower
    bounds_df["upper_bound"] = upper
    bounds_df["interval_width"] = upper - lower
    bounds_df["covered"] = covered

    return cert, bounds_df
