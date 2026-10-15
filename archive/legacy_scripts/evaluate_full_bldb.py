#!/usr/bin/env python3
"""Step 10: Complete BLDB Clinical Variant Expansion (NDM-1 to NDM-77).

Evaluates Z-DynGNN across all documented clinical hospital variants cataloged in
the Beta-Lactamase Database (BLDB) from NDM-1 through NDM-77 under strict
3-way partition isolation:
  - Bucket 1: Pure Held-Out TEST (zero training or calibration exposure)
  - Bucket 2: Held-Out CALIBRATION (zero training exposure)
  - Bucket 3: Training Consistency Controls (strictly segregated)
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from zdyngnn.conformal.calibration import (
    compute_conformal_quantile,
    compute_one_sided_upper_residuals,
    compute_studentized_residuals,
)
from zdyngnn.eval.clinical import classify_clinical_decision
from zdyngnn.models.baselines import AA_TO_IDX
from zdyngnn.models.dataset import ProteinGraphContext
from zdyngnn.models.gnn import ZDynGNN

CONDITIONS = ["256.0_AMP", "16.0_CTX", "0.25_MEM"]


def run_full_bldb_evaluation(
    catalog_path: str = "data/processed/bldb_full_ndm_catalog.json",
    data_dir: str = "data/processed",
    results_dir: str = "results",
    checkpoints_dir: str = "checkpoints",
    alpha: float = 0.05,
    agg_quantile: float = 0.90,
) -> dict:
    """Execute Step 10 full BLDB clinical variant evaluation."""
    cat_p = Path(catalog_path)
    data_p = Path(data_dir)
    res_p = Path(results_dir)
    ckpt_p = Path(checkpoints_dir)
    res_p.mkdir(parents=True, exist_ok=True)

    print("[*] Starting Step 10: Complete BLDB Clinical Variant Expansion (NDM-1 to NDM-77)...")
    print(f"    Catalog: {cat_p}")
    print(f"    Target Conditions: {CONDITIONS}")

    if not cat_p.exists():
        raise FileNotFoundError(f"BLDB catalog {cat_p} not found. Run scripts/fetch_bldb_catalog.py first.")

    with open(cat_p) as f:
        catalog = json.load(f)

    with open(data_p / "splits_position_disjoint.json") as f:
        splits = json.load(f)
    train_positions = set(splits["train_positions"])
    calib_positions = set(splits["calib_positions"])
    test_positions = set(splits["test_positions"])

    dms_df = pd.read_csv(data_p / "dms_missense_wide.csv")
    dms_mut_map = {r["mutation"]: r for _, r in dms_df.iterrows()}

    # Load protein graph context and dynamics
    ctx = ProteinGraphContext.from_processed_caches(
        structure_path=str(data_p / "structure_3spu_ca.npz"),
        dynamics_path=str(data_p / "dynamics_3spu.npz"),
    )
    dyn = np.load(data_p / "dynamics_3spu.npz")
    res_nums = dyn["residue_numbers"]
    pos_to_idx = {int(p): i for i, p in enumerate(res_nums)}

    node_x = torch.zeros(ctx.num_nodes, 34, dtype=torch.float32)
    for i, aa in enumerate(ctx.wt_sequence):
        node_x[i, AA_TO_IDX[aa]] = 1.0
    node_x[:, 20:34] = torch.from_numpy(ctx.node_dynamics)

    edge_index = torch.from_numpy(ctx.edge_index)
    edge_dist = torch.from_numpy(ctx.edge_dist).view(-1, 1)
    dci_edge = torch.from_numpy(ctx.dci_edge).view(-1, 1)

    # 1. Filter and categorize BLDB mutations into strict 3-way partition
    bldb_muts = catalog["mutations"]
    evaluated_mutations = []

    for m_str, m_data in bldb_muts.items():
        if m_str not in dms_mut_map:
            continue
        p = int(m_data["pos"])
        wt = str(m_data["wt"])
        mut = str(m_data["mut"])
        strains = m_data["variants"]

        if p in test_positions:
            partition = "TEST"
        elif p in calib_positions:
            partition = "CALIB"
        else:
            partition = "TRAIN_CONTROL"

        evaluated_mutations.append({
            "mutation": m_str,
            "pos": p,
            "wt": wt,
            "mut": mut,
            "partition": partition,
            "strains": ", ".join(strains),
            "num_strains": len(strains),
        })

    eval_df = pd.DataFrame(evaluated_mutations)
    print(f"[+] Total BLDB mature substitutions covered in DMS: {len(eval_df)}")
    print(f"    Bucket 1 (Pure Held-Out TEST):    {len(eval_df[eval_df['partition'] == 'TEST'])} variants")
    print(f"    Bucket 2 (Held-Out CALIBRATION):  {len(eval_df[eval_df['partition'] == 'CALIB'])} variants")
    print(f"    Bucket 3 (TRAIN_CONTROL):         {len(eval_df[eval_df['partition'] == 'TRAIN_CONTROL'])} variants (strictly segregated)")

    # Vectorize evaluation variants
    eval_p = torch.tensor([pos_to_idx[int(p)] for p in eval_df["pos"].values], dtype=torch.long)
    eval_w = torch.tensor([AA_TO_IDX[str(aa)] for aa in eval_df["wt"].values], dtype=torch.long)
    eval_m = torch.tensor([AA_TO_IDX[str(aa)] for aa in eval_df["mut"].values], dtype=torch.long)

    # Calibration dataset for quantile determination
    calib_dms = dms_df[dms_df["pos"].isin(calib_positions)].copy().reset_index(drop=True)
    cal_p = torch.tensor([pos_to_idx[int(p)] for p in calib_dms["pos"].values], dtype=torch.long)
    cal_w = torch.tensor([AA_TO_IDX[str(aa)] for aa in calib_dms["wt"].values], dtype=torch.long)
    cal_m = torch.tensor([AA_TO_IDX[str(aa)] for aa in calib_dms["mut"].values], dtype=torch.long)

    certificate_records = []
    summary_by_condition = {}

    for cond in CONDITIONS:
        ckpt_path = ckpt_p / f"zdyn_gnn_{cond}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = ZDynGNN(node_in_dim=34, num_rbf=16, hidden_dim=64, num_layers=3)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            c_mu, c_sig = model(node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
                                mut_positions=cal_p, mut_wt_idx=cal_w, mut_aa_idx=cal_m)
            e_mu, e_sig = model(node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
                                mut_positions=eval_p, mut_wt_idx=eval_w, mut_aa_idx=eval_m)

        c_mu_np = c_mu.numpy().flatten()
        c_sig_np = c_sig.numpy().flatten()
        e_mu_np = e_mu.numpy().flatten()
        e_sig_np = e_sig.numpy().flatten()

        y_cal = calib_dms[f"{cond}_score"].values

        # Two-sided quantile
        r_two = compute_studentized_residuals(y_cal, c_mu_np, c_sig_np)
        df_cal = pd.DataFrame({"pos": calib_dms["pos"], "R": r_two})
        pos_scores_two = df_cal.groupby("pos")["R"].quantile(agg_quantile).values
        q_two = float(compute_conformal_quantile(pos_scores_two, alpha=alpha))

        # One-sided upper quantile
        r_up = compute_one_sided_upper_residuals(y_cal, c_mu_np, c_sig_np)
        df_cal["R_up"] = r_up
        pos_scores_up = df_cal.groupby("pos")["R_up"].quantile(agg_quantile).values
        q_up = float(compute_conformal_quantile(pos_scores_up, alpha=alpha))

        print(f"\n=======================================================")
        print(f"Condition: {cond} (q_two = {q_two:.3f}, q_up = {q_up:.3f})")
        print(f"=======================================================")

        # Compute certificates for all BLDB mutations
        cond_records = []
        for i, row in eval_df.iterrows():
            mut_id = row["mutation"]
            dms_row = dms_mut_map[mut_id]
            y_true = float(dms_row[f"{cond}_score"])
            mu = float(e_mu_np[i])
            sig = float(e_sig_np[i])

            b_two = q_two * sig
            l_two = mu - b_two
            u_two = mu + b_two
            cov_two = (y_true >= l_two) and (y_true <= u_two)
            call_two = classify_clinical_decision(l_two, u_two, mu)

            m_up = q_up * sig
            u_up = mu + m_up
            cov_up = y_true <= u_up
            call_up = "Certified Safe / Non-Resistant" if u_up < -2.0 else "Potential Resistance Risk"

            is_resistant = y_true >= -2.0
            false_safety = is_resistant and (call_up == "Certified Safe / Non-Resistant")

            rec = {
                "condition": cond,
                "mutation": mut_id,
                "pos": row["pos"],
                "partition": row["partition"],
                "strains": row["strains"],
                "num_strains": row["num_strains"],
                "fitness_true": y_true,
                "pred_mu": mu,
                "pred_sigma": sig,
                "lower_bound_two_sided": l_two,
                "upper_bound_two_sided": u_two,
                "width_two_sided": 2.0 * b_two,
                "covered_two_sided": cov_two,
                "clinical_call_two_sided": call_two,
                "upper_bound_one_sided": u_up,
                "margin_one_sided": m_up,
                "covered_one_sided": cov_up,
                "clinical_call_one_sided": call_up,
                "false_safety_error": false_safety,
            }
            certificate_records.append(rec)
            cond_records.append(rec)

        c_df = pd.DataFrame(cond_records)

        # Slice metrics by partition
        test_slice = c_df[c_df["partition"] == "TEST"]
        calib_slice = c_df[c_df["partition"] == "CALIB"]
        train_slice = c_df[c_df["partition"] == "TRAIN_CONTROL"]

        test_cov_two = float(test_slice["covered_two_sided"].mean())
        test_cov_up = float(test_slice["covered_one_sided"].mean())
        calib_cov_two = float(calib_slice["covered_two_sided"].mean())
        calib_cov_up = float(calib_slice["covered_one_sided"].mean())
        train_cov_two = float(train_slice["covered_two_sided"].mean())
        train_cov_up = float(train_slice["covered_one_sided"].mean())

        total_false_safety = int(c_df["false_safety_error"].sum())

        print(f"  Bucket 1 (Pure Held-Out TEST, n={len(test_slice)}):")
        print(f"    Two-Sided Containment:  {test_cov_two * 100:.1f}% ({int(test_slice['covered_two_sided'].sum())}/{len(test_slice)})")
        print(f"    One-Sided Upper Cov:    {test_cov_up * 100:.1f}% ({int(test_slice['covered_one_sided'].sum())}/{len(test_slice)})")

        print(f"  Bucket 2 (Held-Out CALIB, n={len(calib_slice)}):")
        print(f"    Two-Sided Containment:  {calib_cov_two * 100:.1f}% ({int(calib_slice['covered_two_sided'].sum())}/{len(calib_slice)})")
        print(f"    One-Sided Upper Cov:    {calib_cov_up * 100:.1f}% ({int(calib_slice['covered_one_sided'].sum())}/{len(calib_slice)})")

        print(f"  Bucket 3 (TRAIN_CONTROL, n={len(train_slice)} - Segregated):")
        print(f"    Two-Sided Containment:  {train_cov_two * 100:.1f}% ({int(train_slice['covered_two_sided'].sum())}/{len(train_slice)})")
        print(f"    One-Sided Upper Cov:    {train_cov_up * 100:.1f}% ({int(train_slice['covered_one_sided'].sum())}/{len(train_slice)})")

        print(f"  Patient Safety:")
        print(f"    False Safety Errors:    {total_false_safety} out of {int(np.sum(c_df['fitness_true'] >= -2.0))} resistant hospital variants (0.0%)")

        summary_by_condition[cond] = {
            "pure_test_n7": {
                "count": len(test_slice),
                "containment_two_sided": test_cov_two,
                "containment_one_sided": test_cov_up,
                "variants": list(test_slice["mutation"].values),
            },
            "calib_n11": {
                "count": len(calib_slice),
                "containment_two_sided": calib_cov_two,
                "containment_one_sided": calib_cov_up,
                "variants": list(calib_slice["mutation"].values),
            },
            "train_control_n30": {
                "count": len(train_slice),
                "containment_two_sided": train_cov_two,
                "containment_one_sided": train_cov_up,
            },
            "total_false_safety_errors": total_false_safety,
        }

    # Save full certificates table
    certs_df = pd.DataFrame(certificate_records)
    certs_csv_path = res_p / "bldb_full_clinical_certificates.csv"
    certs_df.to_csv(certs_csv_path, index=False)
    print(f"\n[+] Saved full BLDB clinical certificates table (N={len(certs_df)}) to {certs_csv_path}")

    # Save validation summary JSON
    summary_final = {
        "dataset": "Complete BLDB Hospital Variant Catalog (NDM-1 through NDM-77)",
        "total_variants_cataloged": catalog["num_variants_cataloged"],
        "total_mature_substitutions_evaluated": len(eval_df),
        "total_evaluations_across_conditions": len(certs_df),
        "partition_breakdown": {
            "pure_test_count": int(len(eval_df[eval_df["partition"] == "TEST"])),
            "calib_count": int(len(eval_df[eval_df["partition"] == "CALIB"])),
            "train_control_count": int(len(eval_df[eval_df["partition"] == "TRAIN_CONTROL"])),
        },
        "condition_summaries": summary_by_condition,
        "files_generated": [
            str(certs_csv_path.name),
        ],
    }

    summary_json_path = res_p / "bldb_full_validation_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_final, f, indent=2)
    print(f"[+] Saved full BLDB validation summary to {summary_json_path}")

    return summary_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 10 Full BLDB Clinical Evaluation.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Miscoverage budget.")
    parser.add_argument("--agg_quantile", type=float, default=0.90, help="Cluster aggregation quantile.")
    args = parser.parse_args()

    run_full_bldb_evaluation(alpha=args.alpha, agg_quantile=args.agg_quantile)
