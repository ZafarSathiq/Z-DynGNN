#!/usr/bin/env python3
"""Step 8: Multi-Concentration Pharmacodynamic Expansion & Dose-Response Monotonicity.

Evaluates Z-DynGNN across all 9 drug concentrations in Chen et al. 2024 DMS:
  - Ampicillin: 32.0, 256.0, 512.0 µg/mL
  - Cefotaxime: 2.0, 16.0, 32.0 µg/mL
  - Meropenem: 0.0625, 0.25, 0.5 µg/mL

Performs:
  1. Biological & predicted dose-response monotonicity auditing on held-out test positions.
  2. Cluster-adjusted and one-sided conformal calibration across all 9 conditions.
  3. Pharmacodynamic trajectory tracking of the 22 unexposed clinical variants (including M154L & M248L).
  4. Multi-dose clinical decision utility evaluation under zero retraining.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch

from zdyngnn.conformal.calibration import (
    compute_conformal_quantile,
    compute_one_sided_upper_residuals,
    compute_studentized_residuals,
)
from zdyngnn.eval.clinical import UNEXPOSED_VALIDATION_VARIANTS, classify_clinical_decision
from zdyngnn.models.baselines import AA_TO_IDX
from zdyngnn.models.dataset import ProteinGraphContext
from zdyngnn.models.gnn import ZDynGNN

DRUG_SERIES = {
    "Ampicillin": {
        "mid_condition": "256.0_AMP",
        "conditions": ["32.0_AMP", "256.0_AMP", "512.0_AMP"],
        "concentrations_ug_ml": [32.0, 256.0, 512.0],
    },
    "Cefotaxime": {
        "mid_condition": "16.0_CTX",
        "conditions": ["2.0_CTX", "16.0_CTX", "32.0_CTX"],
        "concentrations_ug_ml": [2.0, 16.0, 32.0],
    },
    "Meropenem": {
        "mid_condition": "0.25_MEM",
        "conditions": ["0.0625_MEM", "0.25_MEM", "0.5_MEM"],
        "concentrations_ug_ml": [0.0625, 0.25, 0.5],
    },
}


def run_dose_response_evaluation(
    data_dir: str = "data/processed",
    results_dir: str = "results",
    checkpoints_dir: str = "checkpoints",
    alpha: float = 0.05,
    agg_quantile: float = 0.90,
) -> dict:
    """Execute Step 8 multi-concentration pharmacodynamic evaluation."""
    data_p = Path(data_dir)
    res_p = Path(results_dir)
    ckpt_p = Path(checkpoints_dir)
    res_p.mkdir(parents=True, exist_ok=True)

    print("[*] Starting Step 8: Multi-Concentration Pharmacodynamic Expansion...")
    print(f"    Confidence: {100.0 * (1.0 - alpha):.1f}% (alpha = {alpha})")
    print(f"    Cluster aggregation quantile: {agg_quantile}")

    # Load shared protein graph context and dynamics
    ctx = ProteinGraphContext.from_processed_caches(
        structure_path=str(data_p / "structure_3spu_ca.npz"),
        dynamics_path=str(data_p / "dynamics_3spu.npz"),
    )
    dyn = np.load(data_p / "dynamics_3spu.npz")
    res_nums = dyn["residue_numbers"]
    pos_to_idx = {int(p): i for i, p in enumerate(res_nums)}

    with open(data_p / "splits_position_disjoint.json") as f:
        splits = json.load(f)
    calib_positions = set(splits["calib_positions"])
    test_positions = set(splits["test_positions"])

    dms_df = pd.read_csv(data_p / "dms_missense_wide.csv")
    calib_dms = dms_df[dms_df["pos"].isin(calib_positions)].copy().reset_index(drop=True)
    test_dms = dms_df[dms_df["pos"].isin(test_positions)].copy().reset_index(drop=True)

    # 22 strictly unexposed clinical variants
    catalog = UNEXPOSED_VALIDATION_VARIANTS
    bldb_map = {item["mutation"]: item for item in catalog}
    bldb_dms = dms_df[dms_df["mutation"].isin(bldb_map.keys())].copy().reset_index(drop=True)

    # Build node representation (20 WT one-hot + 14 dynamics)
    node_x = torch.zeros(ctx.num_nodes, 34, dtype=torch.float32)
    for i, aa in enumerate(ctx.wt_sequence):
        node_x[i, AA_TO_IDX[aa]] = 1.0
    node_x[:, 20:34] = torch.from_numpy(ctx.node_dynamics)

    edge_index = torch.from_numpy(ctx.edge_index)
    edge_dist = torch.from_numpy(ctx.edge_dist).view(-1, 1)
    dci_edge = torch.from_numpy(ctx.dci_edge).view(-1, 1)

    def to_tensors(sub_df: pd.DataFrame):
        p = torch.tensor([pos_to_idx[int(pos)] for pos in sub_df["pos"].values], dtype=torch.long)
        w = torch.tensor([AA_TO_IDX[str(aa)] for aa in sub_df["wt"].values], dtype=torch.long)
        m = torch.tensor([AA_TO_IDX[str(aa)] for aa in sub_df["mut"].values], dtype=torch.long)
        return p, w, m

    calib_p, calib_w, calib_m = to_tensors(calib_dms)
    test_p, test_w, test_m = to_tensors(test_dms)
    bldb_p, bldb_w, bldb_m = to_tensors(bldb_dms)

    # Output storage
    dose_summary = {}
    certificate_records = []
    clinical_trajectory_records = []

    # Experimental noise floor constant (approx 1 SD of replicate noise)
    noise_tol = 0.51

    for drug_name, meta in DRUG_SERIES.items():
        cond_mid = meta["mid_condition"]
        conditions = meta["conditions"]
        ckpt_path = ckpt_p / f"zdyn_gnn_{cond_mid}.pt"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found. Cannot evaluate {drug_name}.")

        print(f"\n=======================================================")
        print(f"Drug Series: {drug_name} (Canonical Model: {cond_mid})")
        print(f"Conditions: {conditions}")
        print(f"=======================================================")

        # Load frozen model
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = ZDynGNN(node_in_dim=34, num_rbf=16, hidden_dim=64, num_layers=3)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Run forward inference with zero retraining
        with torch.no_grad():
            c_mu, c_sig = model(
                node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
                mut_positions=calib_p, mut_wt_idx=calib_w, mut_aa_idx=calib_m,
            )
            t_mu, t_sig = model(
                node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
                mut_positions=test_p, mut_wt_idx=test_w, mut_aa_idx=test_m,
            )
            b_mu, b_sig = model(
                node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
                mut_positions=bldb_p, mut_wt_idx=bldb_w, mut_aa_idx=bldb_m,
            )

        c_mu_np = c_mu.numpy().flatten()
        c_sig_np = c_sig.numpy().flatten()
        t_mu_np = t_mu.numpy().flatten()
        t_sig_np = t_sig.numpy().flatten()
        b_mu_np = b_mu.numpy().flatten()
        b_sig_np = b_sig.numpy().flatten()

        # 1. Biological Monotonicity on Test Set: high <= mid <= low
        y_low = test_dms[f"{conditions[0]}_score"].values
        y_mid = test_dms[f"{conditions[1]}_score"].values
        y_high = test_dms[f"{conditions[2]}_score"].values

        strict_bio_mono = np.mean((y_high <= y_mid) & (y_mid <= y_low))
        noise_bio_mono = np.mean((y_high <= y_mid + noise_tol) & (y_mid <= y_low + noise_tol))
        step1_bio = np.mean(y_mid <= y_low + noise_tol)
        step2_bio = np.mean(y_high <= y_mid + noise_tol)

        print(f"[*] Biological Ground-Truth Monotonicity on Test Set (N={len(test_dms)}):")
        print(f"    Strict (y_high <= y_mid <= y_low):           {strict_bio_mono * 100:.2f}%")
        print(f"    With Replicate Noise Tolerance (eps=0.51):  {noise_bio_mono * 100:.2f}%")
        print(f"    Low -> Mid Dose Monotonicity:               {step1_bio * 100:.2f}%")
        print(f"    Mid -> High Dose Monotonicity:              {step2_bio * 100:.2f}%")

        drug_res = {
            "canonical_mid_condition": cond_mid,
            "conditions": conditions,
            "test_biological_monotonicity_strict": float(strict_bio_mono),
            "test_biological_monotonicity_with_noise_tol": float(noise_bio_mono),
            "step1_low_to_mid_monotonicity": float(step1_bio),
            "step2_mid_to_high_monotonicity": float(step2_bio),
            "conditions_evaluation": {},
        }

        # 2. Condition-by-Condition Conformal Calibration across the 3 concentrations
        for cond_idx, cond in enumerate(conditions):
            y_cal = calib_dms[f"{cond}_score"].values
            y_tst = test_dms[f"{cond}_score"].values
            y_bldb = bldb_dms[f"{cond}_score"].values

            # Performance on test positions
            p_r, _ = pearsonr(y_tst, t_mu_np)
            s_rho, _ = spearmanr(y_tst, t_mu_np)
            rmse = float(np.sqrt(np.mean((y_tst - t_mu_np) ** 2)))

            # Two-Sided Cluster-Adjusted Conformal Calibration
            r_two = compute_studentized_residuals(y_cal, c_mu_np, c_sig_np)
            df_cal = pd.DataFrame({"pos": calib_dms["pos"], "R": r_two})
            pos_scores_two = df_cal.groupby("pos")["R"].quantile(agg_quantile).values
            q_two = float(compute_conformal_quantile(pos_scores_two, alpha=alpha))

            b_test_two = q_two * t_sig_np
            lower_test_two = t_mu_np - b_test_two
            upper_test_two = t_mu_np + b_test_two
            covered_two = (y_tst >= lower_test_two) & (y_tst <= upper_test_two)
            cov_two_rate = float(np.mean(covered_two))

            # One-Sided Upper Cluster-Adjusted Conformal Calibration
            r_up = compute_one_sided_upper_residuals(y_cal, c_mu_np, c_sig_np)
            df_cal["R_up"] = r_up
            pos_scores_up = df_cal.groupby("pos")["R_up"].quantile(agg_quantile).values
            q_up = float(compute_conformal_quantile(pos_scores_up, alpha=alpha))

            margin_test_up = q_up * t_sig_np
            upper_test_one_sided = t_mu_np + margin_test_up
            covered_up = y_tst <= upper_test_one_sided
            cov_up_rate = float(np.mean(covered_up))

            # Decisive triage calls on test set
            test_safe_calls = int(np.sum(upper_test_one_sided < -2.0))
            test_risk_calls = int(np.sum(upper_test_one_sided >= -2.0))
            false_safety = int(np.sum((y_tst >= -2.0) & (upper_test_one_sided < -2.0)))

            # Evaluation on 22 Unexposed Clinical Variants
            b_bldb_two = q_two * b_sig_np
            lower_bldb_two = b_mu_np - b_bldb_two
            upper_bldb_two = b_mu_np + b_bldb_two
            bldb_cov_two = (y_bldb >= lower_bldb_two) & (y_bldb <= upper_bldb_two)
            bldb_cov_two_rate = float(np.mean(bldb_cov_two))

            margin_bldb_up = q_up * b_sig_np
            upper_bldb_up = b_mu_np + margin_bldb_up
            bldb_cov_up = y_bldb <= upper_bldb_up
            bldb_cov_up_rate = float(np.mean(bldb_cov_up))

            # Pure TEST variants (n=9)
            test_mask = bldb_dms["pos"].isin(test_positions).values
            pure_test_cov_two = float(np.mean(bldb_cov_two[test_mask]))
            pure_test_cov_up = float(np.mean(bldb_cov_up[test_mask]))

            print(f"\n  Condition: {cond}")
            print(f"    Test Pearson r:        {p_r:.4f} | Spearman rho: {s_rho:.4f} | RMSE: {rmse:.4f}")
            print(f"    Two-Sided Conformal:   q = {q_two:.3f} | Test Cov: {cov_two_rate * 100:.2f}% | Mean Width 2B: {float(np.mean(2 * b_test_two)):.2f}")
            print(f"    One-Sided Upper:       q = {q_up:.3f} | Test Cov: {cov_up_rate * 100:.2f}% | Mean Margin B: {float(np.mean(margin_test_up)):.2f}")
            print(f"    Clinical Containment:  All 22 (2-sided): {bldb_cov_two_rate * 100:.1f}%, (1-sided): {bldb_cov_up_rate * 100:.1f}%")
            print(f"    Pure TEST Only (n=9):  2-sided: {pure_test_cov_two * 100:.1f}%, 1-sided: {pure_test_cov_up * 100:.1f}%")
            print(f"    False Safety Errors:   {false_safety} out of {int(np.sum(y_tst >= -2.0))} resistant test mutants")

            drug_res["conditions_evaluation"][cond] = {
                "concentration_ug_ml": meta["concentrations_ug_ml"][cond_idx],
                "pearson_r": float(p_r),
                "spearman_rho": float(s_rho),
                "rmse": float(rmse),
                "quantile_two_sided": q_two,
                "empirical_coverage_two_sided": cov_two_rate,
                "mean_interval_width_two_sided": float(np.mean(2 * b_test_two)),
                "quantile_one_sided_upper": q_up,
                "empirical_coverage_one_sided_upper": cov_up_rate,
                "mean_margin_one_sided_upper": float(np.mean(margin_test_up)),
                "clinical_containment_all_22_two_sided": bldb_cov_two_rate,
                "clinical_containment_all_22_one_sided": bldb_cov_up_rate,
                "pure_test_containment_n9_two_sided": pure_test_cov_two,
                "pure_test_containment_n9_one_sided": pure_test_cov_up,
                "false_safety_count": false_safety,
            }

            # Append to 9-condition certificates dataframe
            for i in range(len(test_dms)):
                row_test = test_dms.iloc[i]
                certificate_records.append({
                    "drug": drug_name,
                    "condition": cond,
                    "concentration_ug_ml": meta["concentrations_ug_ml"][cond_idx],
                    "pos": int(row_test["pos"]),
                    "wt": str(row_test["wt"]),
                    "mut": str(row_test["mut"]),
                    "mutation": str(row_test["mutation"]),
                    "fitness_true": float(y_tst[i]),
                    "pred_mu": float(t_mu_np[i]),
                    "pred_sigma": float(t_sig_np[i]),
                    "quantile_two_sided": q_two,
                    "lower_two_sided": float(lower_test_two[i]),
                    "upper_two_sided": float(upper_test_two[i]),
                    "width_two_sided": float(2 * b_test_two[i]),
                    "covered_two_sided": bool(covered_two[i]),
                    "quantile_one_sided_upper": q_up,
                    "upper_one_sided": float(upper_test_one_sided[i]),
                    "margin_one_sided": float(margin_test_up[i]),
                    "covered_one_sided": bool(covered_up[i]),
                })

            # Append to clinical trajectories dataframe
            for i in range(len(bldb_dms)):
                row_bldb = bldb_dms.iloc[i]
                mut_id = str(row_bldb["mutation"])
                clinical_trajectory_records.append({
                    "drug": drug_name,
                    "condition": cond,
                    "concentration_ug_ml": meta["concentrations_ug_ml"][cond_idx],
                    "mutation": mut_id,
                    "pos": int(row_bldb["pos"]),
                    "partition": bldb_map[mut_id]["partition"],
                    "fitness_true": float(y_bldb[i]),
                    "pred_mu": float(b_mu_np[i]),
                    "pred_sigma": float(b_sig_np[i]),
                    "lower_two_sided": float(lower_bldb_two[i]),
                    "upper_two_sided": float(upper_bldb_two[i]),
                    "covered_two_sided": bool(bldb_cov_two[i]),
                    "upper_one_sided": float(upper_bldb_up[i]),
                    "covered_one_sided": bool(bldb_cov_up[i]),
                })

        dose_summary[drug_name] = drug_res

    # Save summary JSON
    summary_path = res_p / "dose_response_monotonicity_summary.json"
    with open(summary_path, "w") as f:
        json.dump(dose_summary, f, indent=2)
    print(f"\n[+] Saved dose-response summary to {summary_path}")

    # Save 9-condition certificates CSV
    certs_df = pd.DataFrame(certificate_records)
    certs_csv_path = res_p / "dose_response_certificates_9cond.csv"
    certs_df.to_csv(certs_csv_path, index=False)
    print(f"[+] Saved 9-condition test certificates (N={len(certs_df)}) to {certs_csv_path}")

    # Save clinical variant trajectories CSV
    trajectories_df = pd.DataFrame(clinical_trajectory_records)
    traj_csv_path = res_p / "dose_response_clinical_trajectories.csv"
    trajectories_df.to_csv(traj_csv_path, index=False)
    print(f"[+] Saved clinical variant trajectories (N={len(trajectories_df)}) to {traj_csv_path}")

    return dose_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 8 Dose-Response Evaluation.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Miscoverage budget.")
    parser.add_argument("--agg_quantile", type=float, default=0.90, help="Cluster aggregation quantile.")
    args = parser.parse_args()

    run_dose_response_evaluation(alpha=args.alpha, agg_quantile=args.agg_quantile)
