"""Topology-conditioned conformal prediction and finite-sample calibration."""
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

__all__ = [
    "ConformalCertificate",
    "compute_studentized_residuals",
    "compute_conformal_quantile",
    "compute_winkler_score",
    "calibrate_split_conformal",
    "calibrate_cluster_conformal",
    "StratifiedConformalResult",
    "TopologyConditionedConformalCalibrator",
]
