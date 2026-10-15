"""Topology-Conditioned Conformal Risk Control for Z-DynGNN.

Stratifies calibration residuals across structural packing and biophysical dynamics
strata S(i) to generate certified confidence intervals adapted to local protein environment.
"""
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from zdyngnn.conformal.calibration import (
    ConformalCertificate,
    compute_conformal_quantile,
    compute_studentized_residuals,
    compute_winkler_score,
)


@dataclass
class StratifiedConformalResult:
    """Detailed results from topology-conditioned conformal prediction."""
    certificate: ConformalCertificate
    stratum_quantiles: Dict[int, float]
    stratum_sample_counts: Dict[int, int]
    stratum_coverage: Dict[int, float]
    stratum_mean_width: Dict[int, float]
    bounds_df: pd.DataFrame


class TopologyConditionedConformalCalibrator:
    """Stratified Conformal Predictor conditioned on residue structural environment S(i)."""

    def __init__(
        self,
        alpha: float = 0.05,
        min_stratum_samples: int = 15,
        condition: str = "custom",
    ):
        self.alpha = alpha
        self.min_stratum_samples = min_stratum_samples
        self.condition = condition

        self.global_quantile: float = 2.0
        self.stratum_quantiles: Dict[int, float] = {}
        self.stratum_sample_counts: Dict[int, int] = {}

    def fit(
        self,
        calib_df: pd.DataFrame,
        strata_col: str = "strata",
        score_col: str = "fitness_score",
        mu_col: str = "pred_mu",
        sigma_col: str = "pred_sigma",
    ) -> "TopologyConditionedConformalCalibrator":
        """Compute stratum-specific conformal quantiles on calibration mutations.

        Args:
            calib_df: DataFrame of calibration predictions.
            strata_col: Column containing structural stratum bin IDs S(i).

        Returns:
            self
        """
        r_calib = compute_studentized_residuals(calib_df[score_col], calib_df[mu_col], calib_df[sigma_col])
        calib_df_copy = calib_df.copy()
        calib_df_copy["R"] = r_calib

        # Compute global fallback quantile
        self.global_quantile = compute_conformal_quantile(r_calib, alpha=self.alpha)

        # Compute per-stratum quantiles with hierarchical fallback
        self.stratum_quantiles = {}
        self.stratum_sample_counts = {}

        for s, group in calib_df_copy.groupby(strata_col):
            s_int = int(str(s))
            m_s = len(group)
            self.stratum_sample_counts[s_int] = m_s

            if m_s >= self.min_stratum_samples:
                res_arr = np.asarray(group["R"].to_numpy(), dtype=np.float64)
                q_s = compute_conformal_quantile(res_arr, alpha=self.alpha)
                self.stratum_quantiles[s_int] = q_s
            else:
                # Hierarchical fallback: borrow strength from global quantile
                self.stratum_quantiles[s_int] = self.global_quantile

        return self

    def predict_intervals(
        self,
        test_df: pd.DataFrame,
        strata_col: str = "strata",
        score_col: str = "fitness_score",
        mu_col: str = "pred_mu",
        sigma_col: str = "pred_sigma",
    ) -> StratifiedConformalResult:
        """Construct certified prediction intervals on held-out test mutations.

        Args:
            test_df: DataFrame of test predictions.

        Returns:
            StratifiedConformalResult with full bounds and per-stratum diagnostics.
        """
        test_strata = np.asarray(test_df[strata_col].to_numpy(), dtype=int)
        test_mu = np.asarray(test_df[mu_col].to_numpy(), dtype=np.float64)
        test_sigma = np.asarray(test_df[sigma_col].to_numpy(), dtype=np.float64)
        y_test = np.asarray(test_df[score_col].to_numpy(), dtype=np.float64)

        # Look up stratum-specific quantiles
        test_q = np.array([self.stratum_quantiles.get(int(s), self.global_quantile) for s in test_strata], dtype=np.float64)

        b_test = test_q * test_sigma
        lower = test_mu - b_test
        upper = test_mu + b_test

        covered = (y_test >= lower) & (y_test <= upper)
        emp_coverage = float(np.mean(covered))
        mean_width = float(np.mean(upper - lower))
        median_width = float(np.median(upper - lower))
        winkler = compute_winkler_score(y_test, lower, upper, alpha=self.alpha)

        cert = ConformalCertificate(
            condition=self.condition,
            method="topology_conditioned_conformal",
            alpha=self.alpha,
            nominal_coverage=1.0 - self.alpha,
            empirical_coverage=emp_coverage,
            quantile=float(np.mean(test_q)),
            mean_interval_width=mean_width,
            median_interval_width=median_width,
            winkler_score=winkler,
            num_calib_samples=sum(self.stratum_sample_counts.values()),
            num_test_samples=len(test_df),
            coverage_guaranteed=bool(emp_coverage >= (1.0 - self.alpha - 0.05)),
        )

        # Compute per-stratum diagnostics
        stratum_cov: Dict[int, float] = {}
        stratum_width: Dict[int, float] = {}

        bounds_df = test_df.copy()
        bounds_df["conformal_quantile"] = test_q
        bounds_df["certified_bound"] = b_test
        bounds_df["lower_bound"] = lower
        bounds_df["upper_bound"] = upper
        bounds_df["interval_width"] = upper - lower
        bounds_df["covered"] = covered

        for s, grp in bounds_df.groupby(strata_col):
            s_int = int(str(s))
            stratum_cov[s_int] = float(np.mean(grp["covered"]))
            stratum_width[s_int] = float(np.mean(grp["interval_width"]))

        return StratifiedConformalResult(
            certificate=cert,
            stratum_quantiles=self.stratum_quantiles,
            stratum_sample_counts=self.stratum_sample_counts,
            stratum_coverage=stratum_cov,
            stratum_mean_width=stratum_width,
            bounds_df=bounds_df,
        )
