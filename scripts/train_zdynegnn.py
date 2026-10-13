#!/usr/bin/env python3
"""Training and Evaluation Script for 3D Physics-Equivariant ZDynEGNN (Upgrade C).

Trains E(3)-Equivariant Graph Neural Network (ZDynEGNN) from scratch on the 163
training positions using pure 3D Cartesian coordinates, directional DCI_asym modulation,
and Gaussian Negative Log-Likelihood (NLL) loss.
Zero pre-trained language models (No ESM-2). Pure structural biophysics.
Preserves original ZDynGNN baseline checkpoints completely untouched.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
import torch.optim as optim

from zdyngnn.data.dms_loader import extract_condition_dataset
from zdyngnn.data.noise_ceiling import compute_condition_noise_ceiling
from zdyngnn.data.split import PositionDisjointSplit, verify_disjoint_invariants
from zdyngnn.models.baselines import AA_TO_IDX
from zdyngnn.models.dataset import ProteinGraphContext
from zdyngnn.models.gnn import ZDynEGNN, gaussian_nll_loss


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


def train_zdynegnn(
    condition: str = "256.0_AMP",
    epochs: int = 35,
    batch_size: int = 128,
    lr: float = 2e-3,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.1,
    seed: int = 42,
    checkpoints_dir: str = "checkpoints",
    results_dir: str = "results",
) -> dict:
    """Train 3D-Equivariant ZDynEGNN from scratch on 163 training positions."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    ckpt_p = Path(checkpoints_dir)
    res_p = Path(results_dir)
    ckpt_p.mkdir(parents=True, exist_ok=True)
    res_p.mkdir(parents=True, exist_ok=True)

    print("=====================================================================")
    print(f"TRAINING 3D-EQUIVARIANT ZDYNEGNN (UPGRADE C) FROM SCRATCH")
    print(f"Condition: {condition} | Device: {device} | Seed: {seed}")
    print("=====================================================================")

    # 1. Load protein graph context with 3D coordinates
    ctx = ProteinGraphContext.from_processed_caches()
    if ctx.coords is None:
        struct_data = np.load("data/processed/structure_3spu_ca.npz")
        ctx.coords = struct_data["coords"].astype(np.float32)

    raw_dms = pd.read_csv("data/processed/dms_missense_wide.csv")
    score_col = f"{condition}_score"
    sd_col = f"{condition}_score_sd"
    dms_df = raw_dms.copy()
    dms_df = dms_df.rename(columns={score_col: "fitness_score", sd_col: "replicate_sd"})
    with open("data/processed/splits_position_disjoint.json") as f:
        split_dict = json.load(f)
    split = PositionDisjointSplit.from_dict(split_dict)
    verify_disjoint_invariants(split, df=dms_df)

    train_df = dms_df.iloc[split.train_mutation_indices].reset_index(drop=True)
    calib_df = dms_df.iloc[split.calib_mutation_indices].reset_index(drop=True)
    test_df = dms_df.iloc[split.test_mutation_indices].reset_index(drop=True)

    print(f"    Train: {len(train_df)} muts ({split.num_train_positions} pos) - GRADIENTS ENABLED")
    print(f"    Calib: {len(calib_df)} muts ({split.num_calib_positions} pos) - EARLY STOPPING")
    print(f"    Test:  {len(test_df)} muts ({split.num_test_positions} pos) - 100% BLIND")

    # Noise ceiling
    ceiling_res = compute_condition_noise_ceiling(
        dms_df["fitness_score"].values,
        dms_df["replicate_sd"].values,
        condition_name=condition,
    )
    print(f"    Noise Ceiling: r_max = {ceiling_res.r_max:.4f}, R2_max = {ceiling_res.r2_max:.4f}, SNR = {ceiling_res.snr:.2f}")

    # 3. Assemble Graph Tensors
    n_nodes = ctx.num_nodes
    wt_one_hot = torch.zeros(n_nodes, 20, dtype=torch.float32)
    for i, aa in enumerate(ctx.wt_sequence):
        wt_one_hot[i, AA_TO_IDX[aa]] = 1.0

    node_x = torch.cat([wt_one_hot, torch.from_numpy(ctx.node_dynamics)], dim=-1).to(device)  # (228, 34)
    coords = torch.from_numpy(ctx.coords).to(device)  # (228, 3) 3D Cartesian coordinates
    edge_index = torch.from_numpy(ctx.edge_index).to(device)
    edge_dist = torch.from_numpy(ctx.edge_dist).view(-1, 1).to(device)
    dci_edge = torch.from_numpy(ctx.dci_edge).view(-1, 1).to(device)

    train_tensors = prepare_mutation_tensors(train_df, ctx.pos_to_idx, "fitness_score", "replicate_sd", device)
    calib_tensors = prepare_mutation_tensors(calib_df, ctx.pos_to_idx, "fitness_score", "replicate_sd", device)
    test_tensors = prepare_mutation_tensors(test_df, ctx.pos_to_idx, "fitness_score", "replicate_sd", device)

    # 4. Instantiate 3D-Equivariant Model (ZDynEGNN)
    model = ZDynEGNN(
        node_in_dim=34,
        num_rbf=16,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_calib_loss = float("inf")
    best_checkpoint_path = ckpt_p / f"zdynegnn_{condition}.pt"

    print("\n[*] Training 3D-Equivariant ZDynEGNN from Epoch 0...")
    train_start_time = time.time()
    n_train = len(train_df)
    indices = np.arange(n_train)

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
                node_x, coords, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
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
                node_x, coords, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
                mut_positions=calib_tensors["pos_idx"],
                mut_wt_idx=calib_tensors["wt_idx"],
                mut_aa_idx=calib_tensors["mut_idx"],
            )
            calib_loss = gaussian_nll_loss(calib_tensors["y"], mu_calib, sigma_calib, reduction="mean").item()

        scheduler.step(calib_loss)

        y_c_np = calib_tensors["y"].cpu().numpy().flatten()
        mu_c_np = mu_calib.cpu().numpy().flatten()
        calib_metrics = compute_metrics(y_c_np, mu_c_np, ceiling_res.r2_max, ceiling_res.r_max)

        is_best = calib_loss < best_calib_loss
        if is_best:
            best_calib_loss = calib_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "calib_loss": calib_loss,
                "condition": condition,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "model_type": "ZDynEGNN_3D_Physics_Upgrade",
            }, best_checkpoint_path)

        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"    Epoch {epoch:2d}/{epochs:2d} | Train NLL: {epoch_train_loss:.4f} | "
                  f"Calib NLL: {calib_loss:.4f} | Calib r: {calib_metrics['pearson_raw']:.4f} | "
                  f"Calib R2: {calib_metrics['r2_raw']:.4f} {'*' if is_best else ''}")

    elapsed_time = time.time() - train_start_time
    print(f"\n[+] Training complete in {elapsed_time:.1f}s. Best Calib NLL: {best_calib_loss:.4f}")

    # 5. Final Evaluation on Strictly Held-Out Test Positions (Unseen Positions)
    print("\n[*] Evaluating ZDynEGNN on strictly held-out test positions...")
    ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        mu_test, sigma_test = model(
            node_x, coords, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
            mut_positions=test_tensors["pos_idx"],
            mut_wt_idx=test_tensors["wt_idx"],
            mut_aa_idx=test_tensors["mut_idx"],
        )

    y_test_np = test_tensors["y"].cpu().numpy().flatten()
    mu_test_np = mu_test.cpu().numpy().flatten()
    sigma_test_np = sigma_test.cpu().numpy().flatten()

    egnn_test_metrics = compute_metrics(y_test_np, mu_test_np, ceiling_res.r2_max, ceiling_res.r_max)
    egnn_test_metrics["mean_pred_sigma"] = float(np.mean(sigma_test_np))

    print(f"\n[+] ZDynEGNN (3D-Equivariant Physics) Test Performance (Unseen Positions):")
    print(f"    Pearson r (Raw):        {egnn_test_metrics['pearson_raw']:.4f}")
    print(f"    Pearson r (Normalized): {egnn_test_metrics['pearson_normalized']:.4f} (fraction of noise ceiling)")
    print(f"    Spearman rho:           {egnn_test_metrics['spearman']:.4f}")
    print(f"    R^2 (Raw):              {egnn_test_metrics['r2_raw']:.4f}")
    print(f"    R^2 (Normalized):       {egnn_test_metrics['r2_normalized']:.4f}")
    print(f"    RMSE:                   {egnn_test_metrics['rmse']:.4f}")

    # Load baseline metrics for direct comparison
    baseline_comp_path = res_p / f"model_comparison_{condition}.csv"
    if baseline_comp_path.exists():
        old_comp = pd.read_csv(baseline_comp_path)
        old_zdyn = old_comp[old_comp["model"] == "Z-DynGNN (Ours)"].iloc[0]
        print(f"\n--- Side-by-Side Comparison vs Baseline Z-DynGNN ---")
        print(f"  Z-DynGNN (Baseline 1D):       Pearson r = {old_zdyn['pearson_raw']:.4f}, R2 = {old_zdyn['r2_raw']:.4f}, RMSE = {old_zdyn['rmse']:.4f}")
        print(f"  ZDynEGNN (3D Equivariant):    Pearson r = {egnn_test_metrics['pearson_raw']:.4f}, R2 = {egnn_test_metrics['r2_raw']:.4f}, RMSE = {egnn_test_metrics['rmse']:.4f}")

    # Save test predictions DataFrame
    test_preds_df = test_df[["pos", "wt", "mut", "mutation", "group", "fitness_score", "replicate_sd"]].copy()
    test_preds_df["pred_mu"] = mu_test_np
    test_preds_df["pred_sigma"] = sigma_test_np
    test_preds_path = res_p / f"test_predictions_zdynegnn_{condition}.csv"
    test_preds_df.to_csv(test_preds_path, index=False)
    print(f"[+] Saved test predictions to {test_preds_path}")

    return egnn_test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D-Equivariant ZDynEGNN.")
    parser.add_argument("--condition", default="256.0_AMP", help="Drug condition.")
    parser.add_argument("--epochs", type=int, default=35, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--num_layers", type=int, default=3, help="EGNN layers.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    args = parser.parse_args()

    train_zdynegnn(
        condition=args.condition,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seed=args.seed,
    )
