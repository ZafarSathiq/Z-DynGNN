#!/usr/bin/env python3
"""Training and Evaluation Script for Z-DynGNN and Empirical Baselines.

Trains Z-score Calibrated Dynamics-Aware Graph Neural Network (Z-DynGNN) using
Gaussian Negative Log-Likelihood (NLL) loss under position-disjoint partitioning.
Benchmarks against Site-Mean and Ridge regression baselines.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr

from zdyngnn.data.dms_loader import DMS_CONDITIONS
from zdyngnn.data.noise_ceiling import compute_condition_noise_ceiling
from zdyngnn.data.split import PositionDisjointSplit
from zdyngnn.models.baselines import AA_TO_IDX, RidgeBaseline, SiteMeanBaseline
from zdyngnn.models.dataset import ProteinGraphContext
from zdyngnn.models.gnn import ZDynGNN, gaussian_nll_loss


def prepare_mutation_tensors(
    df: pd.DataFrame,
    pos_to_idx: Dict[int, int],
    score_col: str,
    sd_col: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Extract and vectorize mutation indices, substitutions, and target scores."""
    pos_indices = [pos_to_idx[int(p)] for p in df["pos"].values]
    wt_indices = [AA_TO_IDX[str(aa)] for aa in df["wt"].values]
    mut_indices = [AA_TO_IDX[str(aa)] for aa in df["mut"].values]
    y_values = df[score_col].values.astype(np.float32)

    raw_sds = pd.to_numeric(df[sd_col], errors="coerce").values
    median_sd = float(np.nanmedian(raw_sds)) if not np.all(np.isnan(raw_sds)) else 0.35
    clean_sds = np.where(np.isnan(raw_sds), median_sd, raw_sds).astype(np.float32)

    return {
        "pos_idx": torch.tensor(pos_indices, dtype=torch.long, device=device),
        "wt_idx": torch.tensor(wt_indices, dtype=torch.long, device=device),
        "mut_idx": torch.tensor(mut_indices, dtype=torch.long, device=device),
        "y": torch.tensor(y_values, dtype=torch.float32, device=device).view(-1, 1),
        "sd": torch.tensor(clean_sds, dtype=torch.float32, device=device).view(-1, 1),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, r2_max: float, r_max: float) -> dict:
    """Compute regression metrics and noise-ceiling normalized performance."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    p_corr, _ = pearsonr(y_true, y_pred)
    s_corr, _ = spearmanr(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    norm_r2 = float(r2 / r2_max) if r2_max > 0 else 0.0
    norm_pearson = float(p_corr / r_max) if r_max > 0 else 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "r2_raw": r2,
        "r2_normalized": norm_r2,
        "pearson_raw": float(p_corr),
        "pearson_normalized": norm_pearson,
        "spearman": float(s_corr),
    }


def train_and_benchmark(
    condition: str = "256.0_AMP",
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 2e-3,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.1,
    seed: int = 42,
    splits_path: str = "data/processed/splits_position_disjoint.json",
    dms_path: str = "data/processed/dms_missense_wide.csv",
    out_dir: str = "results",
    checkpoints_dir: str = "checkpoints",
) -> dict:
    """Execute complete Z-DynGNN training and benchmark against empirical baselines."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_p = Path(out_dir)
    ckpt_p = Path(checkpoints_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    ckpt_p.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting Step 5 Z-DynGNN Training on {condition}...")
    print(f"    Device: {device}")
    print(f"    Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")

    # 1. Load Data and Partitions
    ctx = ProteinGraphContext.from_processed_caches()
    raw_dms = pd.read_csv(dms_path)

    score_col = f"{condition}_score"
    sd_col = f"{condition}_score_sd"
    if score_col not in raw_dms.columns:
        raise KeyError(f"Score column '{score_col}' not found in {dms_path}.")

    dms_df = raw_dms.copy()
    dms_df = dms_df.rename(columns={score_col: "fitness_score", sd_col: "replicate_sd"})

    with open(splits_path) as f:
        split_dict = json.load(f)
    split = PositionDisjointSplit.from_dict(split_dict)

    train_df = dms_df.iloc[split.train_mutation_indices].reset_index(drop=True)
    calib_df = dms_df.iloc[split.calib_mutation_indices].reset_index(drop=True)
    test_df = dms_df.iloc[split.test_mutation_indices].reset_index(drop=True)

    print(f"    Dataset: Train={len(train_df)}, Calib={len(calib_df)}, Test={len(test_df)}")

    # Compute Noise Ceiling for this condition
    ceiling_res = compute_condition_noise_ceiling(
        train_df["fitness_score"].tolist() + calib_df["fitness_score"].tolist() + test_df["fitness_score"].tolist(),
        train_df["replicate_sd"].tolist() + calib_df["replicate_sd"].tolist() + test_df["replicate_sd"].tolist(),
        condition_name=condition,
    )
    print(f"    Noise Ceiling: R2_max = {ceiling_res.r2_max:.4f}, r_max = {ceiling_res.r_max:.4f}, SNR = {ceiling_res.snr:.2f}")

    # 2. Benchmark Baselines on Held-Out Test Positions
    print("[*] Training and evaluating empirical baselines...")
    site_mean_model = SiteMeanBaseline().fit(train_df)
    sm_mu_test, sm_sigma_test = site_mean_model.predict(test_df)
    sm_metrics = compute_metrics(test_df["fitness_score"].values, sm_mu_test, ceiling_res.r2_max, ceiling_res.r_max)
    print(f"    [Baseline 1: Site-Mean] Test Pearson: {sm_metrics['pearson_raw']:.4f}, R2: {sm_metrics['r2_raw']:.4f}")

    node_feat_map = {int(p): ctx.node_dynamics[i] for i, p in enumerate(ctx.residue_numbers)}
    ridge_model = RidgeBaseline(alpha=10.0).fit(train_df, node_feat_map)
    ridge_mu_test, ridge_sigma_test = ridge_model.predict(test_df, node_feat_map)
    ridge_metrics = compute_metrics(test_df["fitness_score"].values, ridge_mu_test, ceiling_res.r2_max, ceiling_res.r_max)
    print(f"    [Baseline 2: Ridge Reg] Test Pearson: {ridge_metrics['pearson_raw']:.4f}, R2: {ridge_metrics['r2_raw']:.4f}")

    # 3. Assemble Static Protein Graph Tensors
    n_nodes = ctx.num_nodes
    wt_one_hot = torch.zeros(n_nodes, 20, dtype=torch.float32)
    for i, aa in enumerate(ctx.wt_sequence):
        wt_one_hot[i, AA_TO_IDX[aa]] = 1.0

    node_x = torch.cat([wt_one_hot, torch.from_numpy(ctx.node_dynamics)], dim=-1).to(device)  # (228, 34)
    edge_index = torch.from_numpy(ctx.edge_index).to(device)
    edge_dist = torch.from_numpy(ctx.edge_dist).view(-1, 1).to(device)
    dci_edge = torch.from_numpy(ctx.dci_edge).view(-1, 1).to(device)

    # 4. Vectorize Partitions
    train_tensors = prepare_mutation_tensors(train_df, ctx.pos_to_idx, "fitness_score", "replicate_sd", device)
    calib_tensors = prepare_mutation_tensors(calib_df, ctx.pos_to_idx, "fitness_score", "replicate_sd", device)
    test_tensors = prepare_mutation_tensors(test_df, ctx.pos_to_idx, "fitness_score", "replicate_sd", device)

    # 5. Model Architecture & Optimization
    model = ZDynGNN(
        node_in_dim=34,
        num_rbf=16,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_calib_loss = float("inf")
    best_checkpoint_path = ckpt_p / f"zdyn_gnn_{condition}.pt"

    print("\n[*] Training Z-DynGNN...")
    train_start_time = time.time()

    n_train = len(train_df)
    indices = np.arange(n_train)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(indices)

        epoch_train_loss = 0.0
        n_batches = int(np.ceil(n_train / batch_size))

        for b in range(n_batches):
            b_idx = indices[b * batch_size : (b + 1) * batch_size]
            b_pos = train_tensors["pos_idx"][b_idx]
            b_wt = train_tensors["wt_idx"][b_idx]
            b_mut = train_tensors["mut_idx"][b_idx]
            b_y = train_tensors["y"][b_idx]

            optimizer.zero_grad()
            mu, sigma = model(
                node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
                mut_positions=b_pos, mut_wt_idx=b_wt, mut_aa_idx=b_mut,
            )

            loss = gaussian_nll_loss(b_y, mu, sigma, reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_train_loss += loss.item() * len(b_idx)

        epoch_train_loss /= n_train

        # Validation on Calibration Set
        model.eval()
        with torch.no_grad():
            mu_calib, sigma_calib = model(
                node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
                mut_positions=calib_tensors["pos_idx"],
                mut_wt_idx=calib_tensors["wt_idx"],
                mut_aa_idx=calib_tensors["mut_idx"],
            )
            calib_loss = gaussian_nll_loss(calib_tensors["y"], mu_calib, sigma_calib, reduction="mean").item()

        scheduler.step(calib_loss)

        y_c_np = calib_tensors["y"].cpu().numpy().flatten()
        mu_c_np = mu_calib.cpu().numpy().flatten()
        calib_metrics = compute_metrics(y_c_np, mu_c_np, ceiling_res.r2_max, ceiling_res.r_max)

        # Checkpointing
        is_best = calib_loss < best_calib_loss
        if is_best:
            best_calib_loss = calib_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "calib_loss": calib_loss,
                "condition": condition,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
            }, best_checkpoint_path)

        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"    Epoch {epoch:2d}/{epochs:2d} | Train NLL: {epoch_train_loss:.4f} | "
                  f"Calib NLL: {calib_loss:.4f} | Calib Pearson: {calib_metrics['pearson_raw']:.4f} | "
                  f"Calib R2: {calib_metrics['r2_raw']:.4f} {'*' if is_best else ''}")

        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "calib_loss": calib_loss,
            "calib_pearson": calib_metrics["pearson_raw"],
            "calib_r2": calib_metrics["r2_raw"],
        })

    elapsed_time = time.time() - train_start_time
    print(f"\n[+] Training complete in {elapsed_time:.1f}s. Best Calib NLL: {best_calib_loss:.4f}")

    # 6. Final Evaluation on Held-Out Test Positions
    print("[*] Evaluating best checkpoint on strictly held-out test positions...")
    ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        mu_test, sigma_test = model(
            node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
            mut_positions=test_tensors["pos_idx"],
            mut_wt_idx=test_tensors["wt_idx"],
            mut_aa_idx=test_tensors["mut_idx"],
        )
        test_loss = gaussian_nll_loss(test_tensors["y"], mu_test, sigma_test, reduction="mean").item()

    y_test_np = test_tensors["y"].cpu().numpy().flatten()
    mu_test_np = mu_test.cpu().numpy().flatten()
    sigma_test_np = sigma_test.cpu().numpy().flatten()

    zdyn_test_metrics = compute_metrics(y_test_np, mu_test_np, ceiling_res.r2_max, ceiling_res.r_max)
    zdyn_test_metrics["test_nll"] = float(test_loss)
    zdyn_test_metrics["mean_pred_sigma"] = float(np.mean(sigma_test_np))
    zdyn_test_metrics["std_pred_sigma"] = float(np.std(sigma_test_np))

    print(f"\n[+] Z-DynGNN Test Performance (Held-out Positions):")
    print(f"    Pearson r (Raw):        {zdyn_test_metrics['pearson_raw']:.4f}")
    print(f"    Pearson r (Normalized): {zdyn_test_metrics['pearson_normalized']:.4f} (fraction of noise ceiling)")
    print(f"    Spearman rho:           {zdyn_test_metrics['spearman']:.4f}")
    print(f"    R^2 (Raw):              {zdyn_test_metrics['r2_raw']:.4f}")
    print(f"    R^2 (Normalized):       {zdyn_test_metrics['r2_normalized']:.4f} (fraction of explainable variance)")
    print(f"    RMSE:                   {zdyn_test_metrics['rmse']:.4f}")
    print(f"    Mean Aleatoric sigma:   {zdyn_test_metrics['mean_pred_sigma']:.4f}")

    # 7. Save Comparison Table & Test Predictions
    comparison_df = pd.DataFrame([
        {"model": "Site-Mean Baseline", **sm_metrics},
        {"model": "Ridge Regression", **ridge_metrics},
        {"model": "Z-DynGNN (Ours)", **zdyn_test_metrics},
    ])
    comp_path = out_p / f"model_comparison_{condition}.csv"
    comparison_df.to_csv(comp_path, index=False)
    print(f"\n[+] Saved model comparison table to {comp_path}")
    print(comparison_df[["model", "rmse", "pearson_raw", "pearson_normalized", "r2_raw", "r2_normalized"]])

    # Save test predictions DataFrame
    test_preds_df = test_df[["pos", "wt", "mut", "mutation", "group", "fitness_score", "replicate_sd"]].copy()
    test_preds_df["pred_mu"] = mu_test_np
    test_preds_df["pred_sigma"] = sigma_test_np
    test_preds_path = out_p / f"test_predictions_{condition}.csv"
    test_preds_df.to_csv(test_preds_path, index=False)

    # Save calibration predictions for Step 6 conformal calibration
    with torch.no_grad():
        mu_calib_eval, sigma_calib_eval = model(
            node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
            mut_positions=calib_tensors["pos_idx"],
            mut_wt_idx=calib_tensors["wt_idx"],
            mut_aa_idx=calib_tensors["mut_idx"],
        )
    calib_preds_df = calib_df[["pos", "wt", "mut", "mutation", "group", "fitness_score", "replicate_sd"]].copy()
    calib_preds_df["pred_mu"] = mu_calib_eval.cpu().numpy().flatten()
    calib_preds_df["pred_sigma"] = sigma_calib_eval.cpu().numpy().flatten()
    calib_preds_path = out_p / f"calib_predictions_{condition}.csv"
    calib_preds_df.to_csv(calib_preds_path, index=False)

    summary = {
        "condition": condition,
        "elapsed_time_sec": elapsed_time,
        "best_epoch": int(ckpt["epoch"]),
        "noise_ceiling": {
            "r2_max": ceiling_res.r2_max,
            "r_max": ceiling_res.r_max,
            "snr": ceiling_res.snr,
        },
        "metrics_test": {
            "site_mean_baseline": sm_metrics,
            "ridge_baseline": ridge_metrics,
            "zdyn_gnn": zdyn_test_metrics,
        },
        "checkpoint_path": str(best_checkpoint_path),
        "files_generated": [
            str(comp_path.name),
            str(test_preds_path.name),
            str(calib_preds_path.name),
        ],
    }

    summary_json_path = out_p / f"training_summary_{condition}.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Z-DynGNN and benchmark empirical baselines.")
    parser.add_argument("--condition", default="256.0_AMP", help="Target drug condition.")
    parser.add_argument("--epochs", type=int, default=35, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="GNN hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=3, help="GNN layers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    train_and_benchmark(
        condition=args.condition,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
