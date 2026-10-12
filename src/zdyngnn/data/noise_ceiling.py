"""Empirical Noise Ceiling Calculations for NDM-1 DMS Fitness Measurements.

Computes theoretical maximum explainable variance (R^2_max) and upper correlation bound
(r_max) from biological replicate variances in Chen et al. 2024.
"""
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from zdyngnn.data.dms_loader import DMS_CONDITIONS


@dataclass
class NoiseCeilingResult:
    """Benchmark noise ceiling statistics for a single assay condition."""
    condition: str
    total_variance: float      # Var(y): total observed measurement variance
    error_variance: float      # E[sigma_error^2]: mean replicate error variance
    signal_variance: float     # Var(true signal) = max(0, Var(y) - E[sigma_error^2])
    r2_max: float              # Theoretical maximum R^2 = signal_variance / total_variance
    r_max: float               # Theoretical maximum Pearson r = sqrt(R^2_max)
    snr: float                 # Signal-to-Noise Ratio = signal_variance / error_variance
    num_samples: int           # Number of valid measurement points evaluated


def compute_condition_noise_ceiling(
    scores: Union[np.ndarray, pd.Series],
    sds: Union[np.ndarray, pd.Series],
    condition_name: str = "custom",
) -> NoiseCeilingResult:
    """Compute empirical noise ceiling from fitness scores and replicate standard deviations.

    Mathematical formulation:
        Var(y) = Var(signal) + Var(noise)
        Var(noise) = mean(sd^2)
        Var(signal) = max(0, Var(y) - Var(noise))
        R^2_max = Var(signal) / Var(y)
        r_max = sqrt(R^2_max)
        SNR = Var(signal) / Var(noise)

    Args:
        scores: Observed fitness scores y.
        sds: Standard deviations across experimental replicates.
        condition_name: Name of condition for reporting.

    Returns:
        NoiseCeilingResult containing ceiling parameters.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    sds_arr = np.asarray(sds, dtype=np.float64)

    # Drop NaNs jointly
    valid_mask = (~np.isnan(scores_arr)) & (~np.isnan(sds_arr))
    valid_scores = scores_arr[valid_mask]
    valid_sds = sds_arr[valid_mask]

    n_samples = len(valid_scores)
    if n_samples < 2:
        raise ValueError(f"Insufficient valid points (n={n_samples}) to compute noise ceiling.")

    total_var = float(np.var(valid_scores, ddof=1))
    # Mean replicate error variance
    error_var = float(np.mean(valid_sds ** 2))

    signal_var = float(max(0.0, total_var - error_var))
    r2_max = float(signal_var / total_var) if total_var > 0 else 0.0
    r_max = float(np.sqrt(r2_max))
    snr = float(signal_var / error_var) if error_var > 0 else float("inf")

    return NoiseCeilingResult(
        condition=condition_name,
        total_variance=total_var,
        error_variance=error_var,
        signal_variance=signal_var,
        r2_max=r2_max,
        r_max=r_max,
        snr=snr,
        num_samples=n_samples,
    )


def compute_all_noise_ceilings(df: pd.DataFrame) -> Dict[str, NoiseCeilingResult]:
    """Compute noise ceilings across all 9 Chen et al. drug conditions.

    Args:
        df: Filtered or raw DMS DataFrame containing score and sd columns.

    Returns:
        Dictionary mapping condition name to NoiseCeilingResult.
    """
    results: Dict[str, NoiseCeilingResult] = {}
    for cond in DMS_CONDITIONS:
        name = cond["condition"]
        score_col = cond["score_col"]
        sd_col = cond["sd_col"]
        if score_col in df.columns and sd_col in df.columns:
            res = compute_condition_noise_ceiling(
                scores=df[score_col],
                sds=df[sd_col],
                condition_name=name,
            )
            results[name] = res
    return results


def noise_ceilings_to_dataframe(ceilings: Dict[str, NoiseCeilingResult]) -> pd.DataFrame:
    """Format dictionary of NoiseCeilingResult into a summary DataFrame."""
    records = [asdict(res) for res in ceilings.values()]
    return pd.DataFrame(records)


def compute_normalized_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    r2_max: float,
    r_max: float,
) -> Dict[str, float]:
    """Compute standard and ceiling-normalized model evaluation metrics.

    Args:
        y_true: Ground truth experimental fitness scores.
        y_pred: Model predictions.
        r2_max: Theoretical maximum R^2 from noise ceiling.
        r_max: Theoretical maximum Pearson r from noise ceiling.

    Returns:
        Dictionary of raw and normalized evaluation metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    # Standard metrics
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    r2_raw = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    p_corr, _ = pearsonr(y_true_arr, y_pred_arr)
    s_corr, _ = spearmanr(y_true_arr, y_pred_arr)

    norm_r2 = float(r2_raw / r2_max) if r2_max > 0 else 0.0
    norm_pearson = float(p_corr / r_max) if r_max > 0 else 0.0

    return {
        "r2_raw": float(r2_raw),
        "r2_normalized": norm_r2,
        "pearson_raw": float(p_corr),
        "pearson_normalized": norm_pearson,
        "spearman": float(s_corr),
        "r2_max_ceiling": float(r2_max),
        "r_max_ceiling": float(r_max),
    }
