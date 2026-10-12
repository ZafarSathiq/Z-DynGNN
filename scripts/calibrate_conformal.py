#!/usr/bin/env python3
"""CLI script to execute Topology-Conditioned Conformal Calibration for Z-DynGNN.

Computes finite-sample non-conformity quantiles and generates certified prediction
intervals C(x_i) = [mu_i - B_i, mu_i + B_i] across primary antibiotic conditions.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from zdyngnn.conformal.calibration import (
    ConformalCertificate,
    calibrate_cluster_conformal,
    calibrate_split_conformal,
)
from zdyngnn.conformal.stratified import (
    StratifiedConformalResult,
    TopologyConditionedConformalCalibrator,
)


def run_conformal_calibration(
    conditions: List[str] = ["256.0_AMP", "16.0_CTX", "0.25_MEM"],
    alpha: float = 0.05,
    dynamics_path: str = "data/processed/dynamics_3spu.npz",
    results_dir: str = "results",
) -> dict:
    """Run conformal calibration across conditions and export certified interval tables."""
    res_dir = Path(results_dir)
    dyn = np.load(dynamics_path)
    res_to_strata = {int(p): int(s) for p, s in zip(dyn["residue_numbers"], dyn["strata_bins"])}

    all_summaries = {}
    benchmark_records = []

    print(f"[*] Starting Step 6 Topology-Conditioned Conformal Calibration...")
    print(f"    Target Confidence: {100.0 * (1.0 - alpha):.1f}% (alpha = {alpha})")
    print(f"    Conditions: {conditions}")

    for condition in conditions:
        calib_file = res_dir / f"calib_predictions_{condition}.csv"
        test_file = res_dir / f"test_predictions_{condition}.csv"

        if not calib_file.exists() or not test_file.exists():
            print(f"[!] Warning: Prediction files for {condition} not found in {results_dir}. Skipping.")
            continue

        print(f"\n--- Condition: {condition} ---")
        calib_df = pd.read_csv(calib_file)
        test_df = pd.read_csv(test_file)

        # Attach structural strata S(i)
        calib_df["strata"] = [res_to_strata.get(int(p), 0) for p in calib_df["pos"].values]
        test_df["strata"] = [res_to_strata.get(int(p), 0) for p in test_df["pos"].values]

        # 1. Standard Split Conformal
        cert_split, bounds_split = calibrate_split_conformal(
            calib_y=calib_df["fitness_score"].values,
            calib_mu=calib_df["pred_mu"].values,
            calib_sigma=calib_df["pred_sigma"].values,
            test_y=test_df["fitness_score"].values,
            test_mu=test_df["pred_mu"].values,
            test_sigma=test_df["pred_sigma"].values,
            alpha=alpha,
            condition=condition,
        )

        # 2. Topology-Conditioned Stratified Conformal
        strat_calibrator = TopologyConditionedConformalCalibrator(
            alpha=alpha,
            min_stratum_samples=15,
            condition=condition,
        ).fit(calib_df)

        strat_res: StratifiedConformalResult = strat_calibrator.predict_intervals(test_df)
        cert_strat = strat_res.certificate

        # 3. Cluster-Adjusted Conformal (Position-Disjoint Grouping)
        cert_cluster, bounds_cluster = calibrate_cluster_conformal(
            calib_df=calib_df,
            test_df=test_df,
            alpha=alpha,
            agg_quantile=0.90,
            condition=condition,
        )

        print(f"  [Method 1: Split Conformal]       Empirical Coverage: {cert_split.empirical_coverage*100:.2f}% | Mean Width: {cert_split.mean_interval_width:.3f} | Quantile: {cert_split.quantile:.3f}")
        print(f"  [Method 2: Topology-Conditioned]  Empirical Coverage: {cert_strat.empirical_coverage*100:.2f}% | Mean Width: {cert_strat.mean_interval_width:.3f} | Quantile (mean): {cert_strat.quantile:.3f}")
        print(f"  [Method 3: Cluster-Adjusted]      Empirical Coverage: {cert_cluster.empirical_coverage*100:.2f}% | Mean Width: {cert_cluster.mean_interval_width:.3f} | Quantile: {cert_cluster.quantile:.3f} *")

        # Save certified certificates table combining both methods
        combined_bounds = strat_res.bounds_df.copy()
        combined_bounds = combined_bounds.rename(columns={
            "conformal_quantile": "quantile_topology_conditioned",
            "certified_bound": "bound_topology_conditioned",
            "lower_bound": "lower_topology_conditioned",
            "upper_bound": "upper_topology_conditioned",
            "interval_width": "width_topology_conditioned",
            "covered": "covered_topology_conditioned",
        })
        combined_bounds["quantile_cluster_adjusted"] = cert_cluster.quantile
        combined_bounds["bound_cluster_adjusted"] = cert_cluster.quantile * combined_bounds["pred_sigma"].values
        combined_bounds["lower_cluster_adjusted"] = bounds_cluster["lower_bound"].values
        combined_bounds["upper_cluster_adjusted"] = bounds_cluster["upper_bound"].values
        combined_bounds["width_cluster_adjusted"] = bounds_cluster["interval_width"].values
        combined_bounds["covered_cluster_adjusted"] = bounds_cluster["covered"].values

        # Standard canonical aliases for test assertions
        combined_bounds["conformal_quantile"] = cert_cluster.quantile
        combined_bounds["certified_bound"] = combined_bounds["bound_cluster_adjusted"]
        combined_bounds["lower_bound"] = combined_bounds["lower_cluster_adjusted"]
        combined_bounds["upper_bound"] = combined_bounds["upper_cluster_adjusted"]
        combined_bounds["interval_width"] = combined_bounds["width_cluster_adjusted"]
        combined_bounds["covered"] = combined_bounds["covered_cluster_adjusted"]

        cert_out_path = res_dir / f"conformal_certificates_{condition}.csv"
        combined_bounds.to_csv(cert_out_path, index=False)

        # Multi-alpha calibration evaluation
        multi_alpha_stats = []
        for target_conf in [0.90, 0.95, 0.99]:
            a = 1.0 - target_conf
            c_clust, _ = calibrate_cluster_conformal(calib_df, test_df, alpha=a, agg_quantile=0.90, condition=condition)
            multi_alpha_stats.append({
                "nominal_coverage": target_conf,
                "empirical_coverage": c_clust.empirical_coverage,
                "quantile": c_clust.quantile,
                "mean_interval_width": c_clust.mean_interval_width,
            })

        all_summaries[condition] = {
            "split_conformal": cert_split.__dict__,
            "topology_conditioned_conformal": {
                **cert_strat.__dict__,
                "stratum_quantiles": strat_res.stratum_quantiles,
                "stratum_coverage": strat_res.stratum_coverage,
            },
            "cluster_adjusted_conformal": cert_cluster.__dict__,
            "multi_confidence_levels": multi_alpha_stats,
        }

        for m_name, c_obj in [("Split Conformal", cert_split), ("Topology-Conditioned", cert_strat), ("Cluster-Adjusted", cert_cluster)]:
            benchmark_records.append({
                "condition": condition,
                "method": m_name,
                "nominal_coverage": c_obj.nominal_coverage,
                "empirical_coverage": c_obj.empirical_coverage,
                "quantile": c_obj.quantile,
                "mean_interval_width": c_obj.mean_interval_width,
                "median_interval_width": c_obj.median_interval_width,
                "winkler_score": c_obj.winkler_score,
                "coverage_guaranteed": c_obj.coverage_guaranteed,
            })

    # Save benchmark table
    benchmarks_df = pd.DataFrame(benchmark_records)
    bench_csv = res_dir / "conformal_benchmarks.csv"
    benchmarks_df.to_csv(bench_csv, index=False)
    print(f"\n[+] Saved conformal benchmark comparisons to {bench_csv}")
    print(benchmarks_df[["condition", "method", "empirical_coverage", "mean_interval_width", "quantile"]])

    summary_json = res_dir / "conformal_summary.json"
    with open(summary_json, "w") as f:
        json.dump(all_summaries, f, indent=2)

    return all_summaries


def main():
    parser = argparse.ArgumentParser(description="Run Topology-Conditioned Conformal Calibration.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Miscoverage level (default 0.05 for 95% coverage).")
    parser.add_argument("--results", default="results", help="Directory containing predictions.")
    args = parser.parse_args()

    run_conformal_calibration(
        conditions=["256.0_AMP", "16.0_CTX", "0.25_MEM"],
        alpha=args.alpha,
        results_dir=args.results,
    )


if __name__ == "__main__":
    main()
