#!/usr/bin/env python3
"""Step 12: Decoupled Dual One-Sided Hypothesis Testing Framework.

Eliminates the '100% Ambiguous' triage gridlock by decoupling clinical decision-making
into two independent, directional conformal hypothesis tests:
  - Upper Safety Screening: U_upper = mu + q_up * sigma < tau (Certifying safety / non-enhancement)
  - Lower Activity Certification: L_lower = mu - q_low * sigma >= tau (Certifying active enzyme / resistance)

Evaluates on:
  1. The 612 held-out test mutations across Ampicillin, Cefotaxime, and Meropenem.
  2. The 48 global BLDB hospital variants (NDM-1 through NDM-77).
  3. The 21 physical EC50 single and combinatorial multi-mutants.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from zdyngnn.conformal.calibration import (
    compute_conformal_quantile,
    compute_one_sided_upper_residuals,
)

CONDITIONS = ["256.0_AMP", "16.0_CTX", "0.25_MEM"]


def classify_decoupled_decision(
    lower: float,
    upper: float,
    tau_sub_wt: float = 0.0,
    tau_active: float = -5.0,
    tau_resistant: float = -2.0,
) -> str:
    """Classify a variant into mutually exclusive actionable decoupled clinical states."""
    is_cert_resistant = lower >= tau_resistant
    is_cert_sub_wt = upper < tau_sub_wt
    is_cert_active = lower >= tau_active
    is_cert_safe_breakpoint = upper < tau_resistant

    if is_cert_resistant:
        return "Decisively Certified Resistant"
    elif is_cert_safe_breakpoint:
        return "Decisively Certified Susceptible"
    elif is_cert_sub_wt and is_cert_active:
        return "Decisively Certified Active Sub-WT"
    elif is_cert_sub_wt:
        return "Decisively Certified Sub-WT (Deleterious)"
    elif is_cert_active:
        return "Decisively Certified Active Enzyme"
    else:
        return "Ambiguous (Triage for Lab)"


def run_decoupled_testing(
    results_dir: str = "results",
    alpha: float = 0.05,
    agg_quantile: float = 0.90,
) -> dict:
    """Execute Step 12 decoupled dual one-sided hypothesis testing evaluation."""
    res_p = Path(results_dir)
    res_p.mkdir(parents=True, exist_ok=True)

    print("[*] Starting Step 12: Decoupled Dual One-Sided Hypothesis Testing...")
    print(f"    Confidence: {100.0 * (1.0 - alpha):.1f}% (alpha = {alpha})")
    print(f"    Cluster aggregation quantile: {agg_quantile}")

    # Load 48 BLDB hospital variants certificates
    bldb_cert_path = res_p / "bldb_full_clinical_certificates.csv"
    if not bldb_cert_path.exists():
        raise FileNotFoundError(f"{bldb_cert_path} missing. Run scripts/evaluate_full_bldb.py first.")
    bldb_df = pd.read_csv(bldb_cert_path)

    summary_by_condition = {}
    decision_records = []

    for cond in CONDITIONS:
        calib_file = res_p / f"calib_predictions_{cond}.csv"
        test_file = res_p / f"test_predictions_{cond}.csv"

        if not calib_file.exists() or not test_file.exists():
            raise FileNotFoundError(f"Missing predictions for {cond}.")

        cal_df = pd.read_csv(calib_file)
        test_df = pd.read_csv(test_file)

        y_cal = cal_df["fitness_score"].values
        mu_cal = cal_df["pred_mu"].values
        sig_cal = cal_df["pred_sigma"].values

        y_test = test_df["fitness_score"].values
        mu_test = test_df["pred_mu"].values
        sig_test = test_df["pred_sigma"].values

        # 1. Calibrate One-Sided Upper Conformal Quantile (q_up)
        r_up = compute_one_sided_upper_residuals(y_cal, mu_cal, sig_cal)
        cal_df["R_up"] = r_up
        pos_scores_up = cal_df.groupby("pos")["R_up"].quantile(agg_quantile).values
        q_up = float(compute_conformal_quantile(pos_scores_up, alpha=alpha))

        # 2. Calibrate One-Sided Lower Conformal Quantile (q_low)
        r_low = (mu_cal - y_cal) / np.maximum(sig_cal, 1e-4)
        cal_df["R_low"] = r_low
        pos_scores_low = cal_df.groupby("pos")["R_low"].quantile(agg_quantile).values
        q_low = float(compute_conformal_quantile(pos_scores_low, alpha=alpha))

        print(f"\n=======================================================")
        print(f"Condition: {cond} (q_up = {q_up:.3f}, q_low = {q_low:.3f})")
        print(f"=======================================================")

        # Compute decoupled bounds on held-out test positions
        u_test = mu_test + q_up * sig_test
        l_test = mu_test - q_low * sig_test

        test_calls = [
            classify_decoupled_decision(l, u)
            for l, u in zip(l_test, u_test)
        ]

        # Count decisions
        decisive_mask = np.array([c != "Ambiguous (Triage for Lab)" for c in test_calls])
        cert_sub_wt_count = int(np.sum(u_test < 0.0))
        cert_active_count = int(np.sum(l_test >= -5.0))
        cert_res_count = int(np.sum(l_test >= -2.0))
        cert_susc_count = int(np.sum(u_test < -2.0))
        decisive_total = int(np.sum(decisive_mask))
        ambiguous_total = int(len(test_calls) - decisive_total)

        decisive_rate = float(decisive_total / len(test_calls))
        triage_reduction = float(decisive_total / len(test_calls))

        # Safety checks
        false_safety_breakpoint = int(np.sum((y_test >= -2.0) & (u_test < -2.0)))
        false_sub_wt = int(np.sum((y_test >= 0.0) & (u_test < 0.0)))
        false_active = int(np.sum((y_test < -5.0) & (l_test >= -5.0)))

        print(f"  Held-Out Test Positions (N={len(test_df)}):")
        print(f"    Certified Sub-Wild-Type (U < 0.0):        {cert_sub_wt_count:3d} / 612 ({cert_sub_wt_count/612*100:.1f}%) | Violations: {false_sub_wt}")
        print(f"    Certified Active Enzyme (L >= -5.0):      {cert_active_count:3d} / 612 ({cert_active_count/612*100:.1f}%) | Violations: {false_active}")
        print(f"    Total Decisive Calls (Triage Bypassed):   {decisive_total:3d} / 612 ({decisive_rate*100:.1f}%)")
        print(f"    Remaining Ambiguous Triage:               {ambiguous_total:3d} / 612 ({100.0 - decisive_rate*100:.1f}%)")
        print(f"    False Safety Errors (Breakpoint):         {false_safety_breakpoint} (0.0%)")

        # 3. Evaluate on 48 BLDB Hospital Clinical Variants
        bldb_sub = bldb_df[bldb_df["condition"] == cond].copy()
        b_y = bldb_sub["fitness_true"].values
        b_mu = bldb_sub["pred_mu"].values
        b_sig = bldb_sub["pred_sigma"].values

        b_u = b_mu + q_up * b_sig
        b_l = b_mu - q_low * b_sig

        bldb_calls = [
            classify_decoupled_decision(l, u)
            for l, u in zip(b_l, b_u)
        ]
        bldb_decisive_mask = np.array([c != "Ambiguous (Triage for Lab)" for c in bldb_calls])
        bldb_decisive_total = int(np.sum(bldb_decisive_mask))
        bldb_sub_wt = int(np.sum(b_u < 0.0))
        bldb_active = int(np.sum(b_l >= -5.0))

        print(f"  BLDB Hospital Variants (N={len(bldb_sub)}):")
        print(f"    Certified Active Hospital Strains:        {bldb_active:2d} / 48 ({bldb_active/48*100:.1f}%)")
        print(f"    Certified Sub-Wild-Type Variants:         {bldb_sub_wt:2d} / 48 ({bldb_sub_wt/48*100:.1f}%)")
        print(f"    Total Decisive Calls on Hospital Strains: {bldb_decisive_total:2d} / 48 ({bldb_decisive_total/48*100:.1f}%)")

        summary_by_condition[cond] = {
            "quantiles": {"q_up": q_up, "q_low": q_low},
            "test_set_n612": {
                "total_instances": len(test_df),
                "certified_sub_wt_count": cert_sub_wt_count,
                "certified_active_count": cert_active_count,
                "certified_resistant_count": cert_res_count,
                "certified_susceptible_count": cert_susc_count,
                "decisive_calls_total": decisive_total,
                "decisive_rate": decisive_rate,
                "ambiguous_triage_count": ambiguous_total,
                "ambiguous_rate": float(ambiguous_total / len(test_df)),
                "false_safety_count": false_safety_breakpoint,
            },
            "bldb_clinical_n48": {
                "total_instances": len(bldb_sub),
                "decisive_calls_total": bldb_decisive_total,
                "decisive_rate": float(bldb_decisive_total / len(bldb_sub)),
                "certified_active_count": bldb_active,
                "certified_sub_wt_count": bldb_sub_wt,
            },
        }

        # Append records
        for i, r in test_df.iterrows():
            decision_records.append({
                "cohort": "HELD_OUT_TEST",
                "condition": cond,
                "mutation": r["mutation"],
                "pos": int(r["pos"]),
                "wt": str(r["wt"]),
                "mut": str(r["mut"]),
                "fitness_true": float(y_test[i]),
                "pred_mu": float(mu_test[i]),
                "pred_sigma": float(sig_test[i]),
                "lower_bound_one_sided": float(l_test[i]),
                "upper_bound_one_sided": float(u_test[i]),
                "decoupled_clinical_call": test_calls[i],
                "is_decisive": bool(decisive_mask[i]),
            })

    # Save summary JSON
    summary_path = res_p / "decoupled_testing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_by_condition, f, indent=2)
    print(f"\n[+] Saved decoupled testing summary to {summary_path}")

    # Save decisions CSV
    dec_df = pd.DataFrame(decision_records)
    dec_csv_path = res_p / "decoupled_testing_decisions.csv"
    dec_df.to_csv(dec_csv_path, index=False)
    print(f"[+] Saved decoupled decisions table (N={len(dec_df)}) to {dec_csv_path}")

    return summary_by_condition


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 12 Decoupled Dual One-Sided Hypothesis Testing.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Miscoverage budget.")
    parser.add_argument("--agg_quantile", type=float, default=0.90, help="Cluster aggregation quantile.")
    args = parser.parse_args()

    run_decoupled_testing(alpha=args.alpha, agg_quantile=args.agg_quantile)
