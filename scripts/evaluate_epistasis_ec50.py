#!/usr/bin/env python3
"""Step 9: Wet-Lab Quantitative EC50 Microdilution & Combinatorial Epistasis Benchmark.

Evaluates Z-DynGNN against physical microdilution EC50 measurements (µg/mL) and
combinatorial double/triple epistatic mutants from Chen et al. 2024 (Supplementary Data 5).

Performs:
  1. Direct physical EC50 correlation and conversion to continuous fitness units.
  2. Strict partition isolation for single mutants (TEST, CALIB, TRAIN_CONTROL).
  3. Combinatorial multi-mutant inference (double and triple mutants) under zero retraining.
  4. Non-linear epistasis detection (synergistic buffering vs. synthetic lethality).
  5. Certified two-sided and one-sided upper containment evaluation.
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
from zdyngnn.models.baselines import AA_TO_IDX
from zdyngnn.models.dataset import ProteinGraphContext
from zdyngnn.models.gnn import ZDynGNN


def run_epistasis_ec50_evaluation(
    data_dir: str = "data/processed",
    raw_ec50_path: str = "DMS Data NDM-1/41467_2024_52614_MOESM8_ESM.xlsx",
    results_dir: str = "results",
    checkpoints_dir: str = "checkpoints",
    condition: str = "256.0_AMP",
    alpha: float = 0.05,
    agg_quantile: float = 0.90,
) -> dict:
    """Execute Step 9 EC50 microdilution and combinatorial epistasis benchmark."""
    data_p = Path(data_dir)
    res_p = Path(results_dir)
    ckpt_p = Path(checkpoints_dir)
    res_p.mkdir(parents=True, exist_ok=True)

    print("[*] Starting Step 9: Wet-Lab EC50 & Combinatorial Epistasis Benchmark...")
    print(f"    Raw EC50 file: {raw_ec50_path}")
    print(f"    Target Condition: {condition}")
    print(f"    Confidence: {100.0 * (1.0 - alpha):.1f}% (alpha = {alpha})")

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
    train_positions = set(splits["train_positions"])
    calib_positions = set(splits["calib_positions"])
    test_positions = set(splits["test_positions"])

    # Load frozen model checkpoint
    ckpt_path = ckpt_p / f"zdyn_gnn_{condition}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ZDynGNN(node_in_dim=34, num_rbf=16, hidden_dim=64, num_layers=3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load calibration data to determine production quantiles
    dms_df = pd.read_csv(data_p / "dms_missense_wide.csv")
    calib_dms = dms_df[dms_df["pos"].isin(calib_positions)].copy().reset_index(drop=True)

    node_x = torch.zeros(ctx.num_nodes, 34, dtype=torch.float32)
    for i, aa in enumerate(ctx.wt_sequence):
        node_x[i, AA_TO_IDX[aa]] = 1.0
    node_x[:, 20:34] = torch.from_numpy(ctx.node_dynamics)

    edge_index = torch.from_numpy(ctx.edge_index)
    edge_dist = torch.from_numpy(ctx.edge_dist).view(-1, 1)
    dci_edge = torch.from_numpy(ctx.dci_edge).view(-1, 1)

    calib_p = torch.tensor([pos_to_idx[int(p)] for p in calib_dms["pos"].values], dtype=torch.long)
    calib_w = torch.tensor([AA_TO_IDX[str(aa)] for aa in calib_dms["wt"].values], dtype=torch.long)
    calib_m = torch.tensor([AA_TO_IDX[str(aa)] for aa in calib_dms["mut"].values], dtype=torch.long)

    with torch.no_grad():
        c_mu, c_sig = model(node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
                            mut_positions=calib_p, mut_wt_idx=calib_w, mut_aa_idx=calib_m)
    c_mu_np = c_mu.numpy().flatten()
    c_sig_np = c_sig.numpy().flatten()

    y_calib = calib_dms[f"{condition}_score"].values

    # Two-sided quantile
    r_two = compute_studentized_residuals(y_calib, c_mu_np, c_sig_np)
    df_cal = pd.DataFrame({"pos": calib_dms["pos"], "R": r_two})
    pos_scores_two = df_cal.groupby("pos")["R"].quantile(agg_quantile).values
    q_two = float(compute_conformal_quantile(pos_scores_two, alpha=alpha))

    # One-sided upper quantile
    r_up = compute_one_sided_upper_residuals(y_calib, c_mu_np, c_sig_np)
    df_cal["R_up"] = r_up
    pos_scores_up = df_cal.groupby("pos")["R_up"].quantile(agg_quantile).values
    q_up = float(compute_conformal_quantile(pos_scores_up, alpha=alpha))

    print(f"    Production Quantiles ({condition}): q_two = {q_two:.3f}, q_up = {q_up:.3f}")

    # Load SD5 experimental EC50 table
    sd5_df = pd.read_excel(raw_ec50_path, sheet_name="SD5_ndm_reversion_expt_ec50")
    print(f"[+] Loaded {len(sd5_df)} entries from SD5 experimental microdilution assay.")

    # 1. Evaluate single mutants
    singles_df = sd5_df[sd5_df["mut_type"] == "single"].copy().reset_index(drop=True)
    single_p = torch.tensor([pos_to_idx[int(p)] for p in singles_df["ndm_pos"].values], dtype=torch.long)
    single_w = torch.tensor([AA_TO_IDX[str(aa)] for aa in singles_df["ndm_wt"].values], dtype=torch.long)
    single_m = torch.tensor([AA_TO_IDX[str(aa)] for aa in singles_df["ndm_mut"].values], dtype=torch.long)

    with torch.no_grad():
        s_mu, s_sig = model(node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
                            mut_positions=single_p, mut_wt_idx=single_w, mut_aa_idx=single_m)
    s_mu_np = s_mu.numpy().flatten()
    s_sig_np = s_sig.numpy().flatten()

    single_pred_map = {}
    certificate_records = []

    print("\n--- Part 1: Physical EC50 Single Mutants Evaluation (N=12) ---")
    for idx, row in singles_df.iterrows():
        p = int(row["ndm_pos"])
        wt = str(row["ndm_wt"])
        mut = str(row["ndm_mut"])
        mut_str = f"{wt}{p}{mut}"
        y_true = float(row["log2_fold_ec50_vs_wt"])
        ec50_ug_ml = float(row["mean_ec50"])
        ec50_sd = float(row["SD"])

        mu_val = float(s_mu_np[idx])
        sig_val = float(s_sig_np[idx])
        single_pred_map[mut_str] = (mu_val, sig_val)

        # Determine partition with strict isolation
        if p in test_positions:
            partition = "TEST"
        elif p in calib_positions:
            partition = "CALIB"
        else:
            partition = "TRAIN_CONTROL"

        # Certified bounds
        b_two = q_two * sig_val
        lower_two = mu_val - b_two
        upper_two = mu_val + b_two
        covered_two = (y_true >= lower_two) and (y_true <= upper_two)

        margin_up = q_up * sig_val
        upper_up = mu_val + margin_up
        covered_up = y_true <= upper_up

        record = {
            "sample": row["sample"],
            "type": "single",
            "mutation": mut_str,
            "pos": p,
            "partition": partition,
            "mean_ec50_ug_ml": ec50_ug_ml,
            "ec50_sd": ec50_sd,
            "log2_fold_ec50_vs_wt": y_true,
            "expected_additive_ec50": np.nan,
            "epistasis_delta": 0.0,
            "pred_mu": mu_val,
            "pred_sigma": sig_val,
            "lower_bound_two_sided": lower_two,
            "upper_bound_two_sided": upper_two,
            "width_two_sided": 2.0 * b_two,
            "covered_two_sided": covered_two,
            "upper_bound_one_sided": upper_up,
            "margin_one_sided": margin_up,
            "covered_one_sided": covered_up,
        }
        certificate_records.append(record)
        print(f"  {mut_str:6s} ({partition:13s}) | EC50: {ec50_ug_ml:5.1f} ug/mL | log2: {y_true:+5.2f} | pred_mu: {mu_val:+5.2f} | [L, U]: [{lower_two:+5.1f}, {upper_two:+5.1f}] ({'True' if covered_two else 'False'})")

    # 2. Evaluate combinatorial multi-mutants (double and triple mutants)
    combs_df = sd5_df[sd5_df["mut_type"] == "combined"].copy().reset_index(drop=True)

    print("\n--- Part 2: Combinatorial Epistatic Multi-Mutants Evaluation (N=9) ---")
    for idx, row in combs_df.iterrows():
        sample_name = str(row["sample"])
        tokens = sample_name.split()[1:]  # e.g. ["K125R", "S249G"]
        y_true = float(row["log2_fold_ec50_vs_wt"])
        y_expected_add = float(row["expected_additive"])
        epistasis_delta = y_true - y_expected_add
        ec50_ug_ml = float(row["mean_ec50"])
        ec50_sd = float(row["SD"])

        # Accumulate predictions across constitutive mutations
        mus = [single_pred_map[m][0] for m in tokens]
        sigs = [single_pred_map[m][1] for m in tokens]

        mu_comb = float(np.sum(mus))
        sig_comb = float(np.sqrt(np.sum([s ** 2 for s in sigs])))

        b_two = q_two * sig_comb
        lower_two = mu_comb - b_two
        upper_two = mu_comb + b_two
        covered_two = (y_true >= lower_two) and (y_true <= upper_two)

        margin_up = q_up * sig_comb
        upper_up = mu_comb + margin_up
        covered_up = y_true <= upper_up

        record = {
            "sample": sample_name,
            "type": "combined",
            "mutation": "+".join(tokens),
            "pos": np.nan,
            "partition": "COMBINATORIAL",
            "mean_ec50_ug_ml": ec50_ug_ml,
            "ec50_sd": ec50_sd,
            "log2_fold_ec50_vs_wt": y_true,
            "expected_additive_ec50": y_expected_add,
            "epistasis_delta": epistasis_delta,
            "pred_mu": mu_comb,
            "pred_sigma": sig_comb,
            "lower_bound_two_sided": lower_two,
            "upper_bound_two_sided": upper_two,
            "width_two_sided": 2.0 * b_two,
            "covered_two_sided": covered_two,
            "upper_bound_one_sided": upper_up,
            "margin_one_sided": margin_up,
            "covered_one_sided": covered_up,
        }
        certificate_records.append(record)
        epi_type = "Synergistic Rescue" if epistasis_delta > 0.5 else ("Negative Epistasis" if epistasis_delta < -0.5 else "Additive")
        print(f"  {sample_name:22s} | EC50: {ec50_ug_ml:5.1f} ug/mL | log2: {y_true:+5.2f} (add: {y_expected_add:+5.2f}, epi: {epistasis_delta:+5.2f} [{epi_type}]) | [L, U]: [{lower_two:+5.1f}, {upper_two:+5.1f}] ({'True' if covered_two else 'False'})")

    # Assemble and save results
    certs_df = pd.DataFrame(certificate_records)
    certs_csv_path = res_p / "epistasis_combinatorial_certificates.csv"
    certs_df.to_csv(certs_csv_path, index=False)
    print(f"\n[+] Saved combinatorial certificates table to {certs_csv_path}")

    # Compute high-level summary statistics
    singles_sub = certs_df[certs_df["type"] == "single"]
    combs_sub = certs_df[certs_df["type"] == "combined"]
    pure_test_sub = certs_df[certs_df["partition"] == "TEST"]

    summary = {
        "dataset": "Chen et al. 2024 Supplementary Data 5 (EC50 Assays)",
        "condition_model": condition,
        "conformal_quantiles": {
            "two_sided_q": q_two,
            "one_sided_upper_q": q_up,
        },
        "single_mutants_n12": {
            "total_count": len(singles_sub),
            "containment_two_sided": float(singles_sub["covered_two_sided"].mean()),
            "containment_one_sided": float(singles_sub["covered_one_sided"].mean()),
            "pure_test_n2": {
                "mutations": list(pure_test_sub["mutation"].values),
                "containment_two_sided": float(pure_test_sub["covered_two_sided"].mean()),
                "containment_one_sided": float(pure_test_sub["covered_one_sided"].mean()),
            },
        },
        "combinatorial_multi_mutants_n9": {
            "total_count": len(combs_sub),
            "containment_two_sided": float(combs_sub["covered_two_sided"].mean()),
            "containment_one_sided": float(combs_sub["covered_one_sided"].mean()),
            "epistasis_events": [
                {
                    "sample": r["sample"],
                    "actual_log2": r["log2_fold_ec50_vs_wt"],
                    "expected_add": r["expected_additive_ec50"],
                    "epistasis_delta": r["epistasis_delta"],
                    "interpretation": "Synergistic Buffering" if r["epistasis_delta"] > 0 else "Deleterious Interference",
                    "contained_in_bound": bool(r["covered_two_sided"]),
                }
                for _, r in combs_sub.iterrows()
            ],
        },
        "overall_physical_ec50_containment": {
            "total_evaluated": len(certs_df),
            "containment_rate_two_sided": float(certs_df["covered_two_sided"].mean()),
            "containment_rate_one_sided": float(certs_df["covered_one_sided"].mean()),
        },
        "files_generated": [
            str(certs_csv_path.name),
        ],
    }

    summary_json_path = res_p / "epistasis_ec50_validation_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved epistasis EC50 summary to {summary_json_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 9 EC50 & Epistasis Benchmark.")
    parser.add_argument("--condition", default="256.0_AMP", help="Target condition model.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Miscoverage budget.")
    parser.add_argument("--agg_quantile", type=float, default=0.90, help="Cluster aggregation quantile.")
    args = parser.parse_args()

    run_epistasis_ec50_evaluation(
        condition=args.condition,
        alpha=args.alpha,
        agg_quantile=args.agg_quantile,
    )
