#!/usr/bin/env python3
"""CLI script to execute Step 7 Clinical Validation and Resistance Decision Utility.

Evaluates certified prediction intervals on BLDB hospital variants, the M154L
allosteric super-variant, and clinical MIC resistance breakpoint decisions.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from zdyngnn.eval.clinical import (
    BLDB_CLINICAL_MUTATIONS,
    UNEXPOSED_VALIDATION_VARIANTS,
    classify_clinical_decision,
    evaluate_clinical_decision_metrics,
)
from zdyngnn.models.baselines import AA_TO_IDX
from zdyngnn.models.dataset import ProteinGraphContext
from zdyngnn.models.gnn import ZDynGNN


def run_clinical_evaluation(
    results_dir: str = "results",
    checkpoints_dir: str = "checkpoints",
    dynamics_path: str = "data/processed/dynamics_3spu.npz",
    dms_path: str = "data/processed/dms_missense_wide.csv",
    splits_path: str = "data/processed/splits_position_disjoint.json",
    conditions: List[str] = ["256.0_AMP", "16.0_CTX", "0.25_MEM"],
) -> dict:
    """Run full clinical validation pipeline."""
    res_dir = Path(results_dir)
    ckpt_dir = Path(checkpoints_dir)
    dyn = np.load(dynamics_path)
    dms_df = pd.read_csv(dms_path)
    ctx = ProteinGraphContext.from_processed_caches()
    device = torch.device("cpu")

    res_nums = dyn["residue_numbers"]
    pos_to_idx = {int(p): i for i, p in enumerate(res_nums)}
    dci_asym = dyn["dci_asym"]

    with open(splits_path) as f:
        split_data = json.load(f)
    train_positions = set(split_data["train_positions"])
    calib_positions = set(split_data["calib_positions"])
    test_positions = set(split_data["test_positions"])

    # Build node_x for 228 nodes: WT one-hot (20) + dynamics (14)
    n_nodes = ctx.num_nodes
    node_x = torch.zeros(n_nodes, 34, dtype=torch.float32)
    for i, aa in enumerate(ctx.wt_sequence):
        node_x[i, AA_TO_IDX[aa]] = 1.0
    node_x[:, 20:34] = torch.from_numpy(ctx.node_dynamics)

    edge_index = torch.from_numpy(ctx.edge_index)
    edge_dist = torch.from_numpy(ctx.edge_dist).view(-1, 1)
    dci_edge = torch.from_numpy(ctx.dci_edge).view(-1, 1)

    print("[*] Starting Step 7 Clinical Validation & Decision Utility...")
    print(f"    Conditions: {conditions}")

    all_bldb_records = []
    decision_records = []

    # Filter DMS table to the 22 strictly unexposed clinical and experimental validation variants
    catalog = UNEXPOSED_VALIDATION_VARIANTS
    bldb_map = {item["mutation"]: item for item in catalog}
    bldb_sub_df = dms_df[dms_df["mutation"].isin(bldb_map.keys())].copy().reset_index(drop=True)
    bldb_pos_idx = torch.tensor([pos_to_idx[int(p)] for p in bldb_sub_df["pos"].values], dtype=torch.long)
    bldb_wt_idx = torch.tensor([AA_TO_IDX[str(aa)] for aa in bldb_sub_df["wt"].values], dtype=torch.long)
    bldb_mut_idx = torch.tensor([AA_TO_IDX[str(aa)] for aa in bldb_sub_df["mut"].values], dtype=torch.long)

    for cond in conditions:
        ckpt_path = ckpt_dir / f"zdyn_gnn_{cond}.pt"
        if not ckpt_path.exists():
            print(f"[!] Warning: Checkpoint {ckpt_path} not found. Skipping {cond}.")
            continue

        # Load production quantile
        with open(res_dir / "conformal_summary.json") as f:
            conf_sum = json.load(f)
        q_prod = float(conf_sum[cond]["cluster_adjusted_conformal"]["quantile"])

        # Load model and run inference on BLDB variants
        ckpt = torch.load(ckpt_path, map_location=device)
        model = ZDynGNN(node_in_dim=34, num_rbf=16, hidden_dim=64, num_layers=3)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        score_col = f"{cond}_score"
        sd_col = f"{cond}_score_sd"

        with torch.no_grad():
            mu_bldb, sigma_bldb = model(
                node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
                mut_positions=bldb_pos_idx, mut_wt_idx=bldb_wt_idx, mut_aa_idx=bldb_mut_idx,
            )

        mu_np = mu_bldb.numpy().flatten()
        sigma_np = sigma_bldb.numpy().flatten()
        bounds_np = q_prod * sigma_np
        lower_np = mu_np - bounds_np
        upper_np = mu_np + bounds_np
        y_true_np = bldb_sub_df[score_col].values.astype(float)
        covered_np = (y_true_np >= lower_np) & (y_true_np <= upper_np)

        print(f"\n--- Condition: {cond} (q_prod = {q_prod:.3f}) ---")
        print(f"    BLDB Variants Evaluated: {len(bldb_sub_df)}")
        print(f"    Clinical Containment Rate: {np.mean(covered_np)*100:.1f}%")
        print(f"    Mean Certified Width: {np.mean(upper_np - lower_np):.3f} log2-units")

        for idx, row in bldb_sub_df.iterrows():
            mut_id = str(row["mutation"])
            meta = bldb_map[mut_id]
            p = int(row["pos"])
            part = "TRAIN" if p in train_positions else ("CALIB" if p in calib_positions else "TEST")

            call = classify_clinical_decision(lower_np[idx], upper_np[idx], mu_np[idx])

            record = {
                "condition": cond,
                "mutation": mut_id,
                "variants": meta.get("source", meta.get("variants", "Clinical isolate")),
                "phenotype": meta["phenotype"],
                "pos": p,
                "partition": meta.get("partition", part),
                "wt": str(row["wt"]),
                "mut": str(row["mut"]),
                "fitness_true": y_true_np[idx],
                "pred_mu": float(mu_np[idx]),
                "pred_sigma": float(sigma_np[idx]),
                "conformal_quantile": q_prod,
                "certified_bound": float(bounds_np[idx]),
                "lower_bound": float(lower_np[idx]),
                "upper_bound": float(upper_np[idx]),
                "interval_width": float(upper_np[idx] - lower_np[idx]),
                "covered": bool(covered_np[idx]),
                "clinical_call": call,
            }
            all_bldb_records.append(record)

        # Evaluate clinical decision metrics on held-out test mutations
        cert_file = res_dir / f"conformal_certificates_{cond}.csv"
        if cert_file.exists():
            cert_df = pd.read_csv(cert_file)
            dec_metrics = evaluate_clinical_decision_metrics(
                y_true=cert_df["fitness_score"].values,
                lower=cert_df["lower_cluster_adjusted"].values,
                upper=cert_df["upper_cluster_adjusted"].values,
                mu=cert_df["pred_mu"].values,
                condition=cond,
            )

            decision_records.append({
                "condition": cond,
                "num_test_mutations": dec_metrics.num_samples,
                "true_resistant": dec_metrics.num_true_resistant,
                "true_susceptible": dec_metrics.num_true_susceptible,
                "certified_containment": f"{dec_metrics.containment_rate*100:.1f}%",
                "false_safety_count": dec_metrics.false_safety_count,
                "triage_rate": f"{dec_metrics.triage_rate*100:.1f}%",
                "decisive_rate": f"{dec_metrics.decisive_rate*100:.1f}%",
            })

            print(f"    [Decision Utility] True Resistant: {dec_metrics.num_true_resistant}, False Safety Errors: {dec_metrics.false_safety_count}")

    # Save unexposed clinical and experimental validation certificates table
    bldb_all_df = pd.DataFrame(all_bldb_records)
    bldb_csv_path = res_dir / "bldb_variant_certificates.csv"
    bldb_all_df.to_csv(bldb_csv_path, index=False)
    print(f"\n[+] Saved strictly unexposed validation table (N={len(bldb_all_df)}) to {bldb_csv_path}")
    # Save Decision Metrics table
    decision_df = pd.DataFrame(decision_records)
    decision_csv_path = res_dir / "clinical_decision_metrics.csv"
    decision_df.to_csv(decision_csv_path, index=False)
    print(f"[+] Saved clinical decision metrics to {decision_csv_path}")
    print(decision_df)
    # 3. Dedicated Deep-Dive: M154L Allosteric Super-Variant
    print("\n[*] Conducting M154L Allosteric Deep Dive...")
    idx_154 = pos_to_idx[154]
    active_site_pos = [120, 122, 124, 189, 208, 250]
    active_site_names = ["His120", "His122", "Asp124", "His189", "Cys208", "His250"]

    couplings = {}
    for p, name in zip(active_site_pos, active_site_names):
        idx_p = pos_to_idx[p]
        c_val = float(dci_asym[idx_p, idx_154])
        couplings[name] = {
            "pos": p,
            "dci_asym_154_to_residue": c_val,
            "interpretation": "Driver" if c_val > 0 else "Sensor",
        }

    m154l_phenotypes = {}
    for cond in conditions:
        m154l_sub = bldb_all_df[(bldb_all_df["mutation"] == "M154L") & (bldb_all_df["condition"] == cond)]
        if len(m154l_sub) > 0:
            r = m154l_sub.iloc[0]
            m154l_phenotypes[cond] = {
                "fitness_true": float(r["fitness_true"]),
                "pred_mu": float(r["pred_mu"]),
                "pred_sigma": float(r["pred_sigma"]),
                "certified_bound": float(r["certified_bound"]),
                "lower_bound": float(r["lower_bound"]),
                "upper_bound": float(r["upper_bound"]),
                "covered": bool(r["covered"]),
                "clinical_call": str(r["clinical_call"]),
            }

    m154l_report = {
        "variant": "M154L",
        "position": 154,
        "structural_motif": "Alpha-3 helix",
        "clinical_significance": "Defining mutation in NDM-4, NDM-5, NDM-7, NDM-15 with enhanced zinc affinity under physiological starvation",
        "phenotypes_across_antibiotics": m154l_phenotypes,
        "allosteric_coupling_to_zinc_catalytic_pocket": couplings,
    }

    m154l_json_path = res_dir / "m154l_clinical_profile.json"
    with open(m154l_json_path, "w") as f:
        json.dump(m154l_report, f, indent=2)
    print(f"[+] Saved M154L allosteric profile to {m154l_json_path}")

    # 4. Final Manifest
    summary_final = {
        "clinical_validation_complete": True,
        "num_unexposed_validation_variants": len(UNEXPOSED_VALIDATION_VARIANTS),
        "total_records": len(bldb_all_df),
        "validation_containment_rates": {
            cond: float(bldb_all_df[bldb_all_df["condition"] == cond]["covered"].mean())
            for cond in conditions if len(bldb_all_df) > 0
        },
        "test_only_containment_rates": {
            cond: float(bldb_all_df[(bldb_all_df["condition"] == cond) & (bldb_all_df["partition"] == "TEST")]["covered"].mean())
            for cond in conditions if len(bldb_all_df) > 0
        },
        "m154l_containment": {
            cond: bool(m154l_phenotypes[cond]["covered"])
            for cond in conditions if cond in m154l_phenotypes
        },
        "decision_utility": decision_records,
        "files_generated": [
            str(bldb_csv_path.name),
            str(decision_csv_path.name),
            str(m154l_json_path.name),
        ],
    }
    summary_final_path = res_dir / "clinical_validation_summary.json"
    with open(summary_final_path, "w") as f:
        json.dump(summary_final, f, indent=2)

    return summary_final

def main():
    parser = argparse.ArgumentParser(description="Evaluate Z-DynGNN on BLDB clinical variants and MIC breakpoints.")
    parser.add_argument("--results", default="results", help="Results directory.")
    args = parser.parse_args()

    run_clinical_evaluation(results_dir=args.results)
    print("\n[+] Step 7 Clinical Validation Complete!")


if __name__ == "__main__":
    main()
