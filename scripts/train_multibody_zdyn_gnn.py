#!/usr/bin/env python3
"""Train and Benchmark Multi-Body Upgraded Z-DynGNN.

Trains Z-DynGNN with higher-order multi-body cumulants and 6-body catalytic zinc
features (Dodd et al. 2026 / Science Advances) from scratch on the 163 training positions.
Performs cluster-adjusted conformal calibration on the 32 calibration positions,
and benchmarks on the 33 unseen test positions against the baseline Z-DynGNN.

Zero data leakage: strictly position-disjoint, zero exposure of clinical hospital variants.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr

from zdyngnn.conformal.calibration import (
    calibrate_cluster_conformal,
    compute_studentized_residuals,
)
from zdyngnn.data.dms_loader import DMS_CONDITIONS
from zdyngnn.data.noise_ceiling import compute_condition_noise_ceiling
from zdyngnn.data.split import PositionDisjointSplit
from zdyngnn.models.baselines import AA_TO_IDX
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


def train_and_benchmark_multibody(
    condition: str = "256.0_AMP",
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 2e-3,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.1,
    seed: int = 42,
    device_str: str = "cpu",
    multibody_source: str = "3spu",
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)

    print("=" * 80)
    print(f"PATH 2: TRAINING MULTI-BODY UPGRADED Z-DYNGNN (Dodd et al. 2026)")
    print(f"Condition: {condition} | Device: {device} | Seed: {seed}")
    print("=" * 80)

    # 1. Load Data, Splits, and Context
    ctx = ProteinGraphContext.from_processed_caches()
    with open("data/processed/splits_position_disjoint.json") as f:
        split_dict = json.load(f)
    split = PositionDisjointSplit.from_dict(split_dict)
    # Load multi-body features tensor from specified crystal structure source
    mb_file = Path(f"data/processed/multibody_{multibody_source.lower()}.npz")
    if not mb_file.exists():
        mb_file = Path("data/processed/multibody_3spu.npz")
    print(f"[+] Using {multibody_source.upper()} multi-body features from {mb_file} for {condition}!")

    if not mb_file.exists():
        raise FileNotFoundError(f"Missing {mb_file}. Run scripts/compute_multibody_biophysics.py first.")
    mb_data = np.load(mb_file)
    node_mb_feats = torch.from_numpy(mb_data["node_multibody_features"]).float().to(device)
    print(f"[+] Loaded node multi-body features of shape {node_mb_feats.shape} from {mb_file}")

    raw_dms = pd.read_csv("data/processed/dms_missense_wide.csv")
    score_col = f"{condition}_score"
    sd_col = f"{condition}_score_sd"
    dms_df = raw_dms.rename(columns={score_col: "fitness_score", sd_col: "replicate_sd"})

    train_df = dms_df.iloc[split.train_mutation_indices].reset_index(drop=True)
    calib_df = dms_df.iloc[split.calib_mutation_indices].reset_index(drop=True)
    test_df = dms_df.iloc[split.test_mutation_indices].reset_index(drop=True)

    print(f"[*] Partitions: Train={len(train_df)} muts ({split.num_train_positions} pos) | "
          f"Calib={len(calib_df)} muts ({split.num_calib_positions} pos) | "
          f"Test={len(test_df)} muts ({split.num_test_positions} pos)")

    # Noise ceiling
    ceiling_res = compute_condition_noise_ceiling(
        dms_df["fitness_score"].tolist(),
        dms_df["replicate_sd"].tolist(),
        condition_name=condition,
    )
    print(f"[*] Noise Ceiling: r_max = {ceiling_res.r_max:.4f}, R2_max = {ceiling_res.r2_max:.4f}")

    # Assemble Static Protein Graph Tensors
    n_nodes = ctx.num_nodes
    wt_one_hot = torch.zeros(n_nodes, 20, dtype=torch.float32)
    for i, aa in enumerate(ctx.wt_sequence):
        wt_one_hot[i, AA_TO_IDX[aa]] = 1.0

    node_x = torch.cat([wt_one_hot, torch.from_numpy(ctx.node_dynamics)], dim=-1).to(device)
    edge_index = torch.from_numpy(ctx.edge_index).to(device)
    edge_dist = torch.from_numpy(ctx.edge_dist).view(-1, 1).to(device)
    dci_edge = torch.from_numpy(ctx.dci_edge).view(-1, 1).to(device)

    train_tensors = prepare_mutation_tensors(train_df, ctx.pos_to_idx, "fitness_score", "replicate_sd", device)
    calib_tensors = prepare_mutation_tensors(calib_df, ctx.pos_to_idx, "fitness_score", "replicate_sd", device)
    test_tensors = prepare_mutation_tensors(test_df, ctx.pos_to_idx, "fitness_score", "replicate_sd", device)

    # 2. Build Multi-Body ZDynGNN Model
    model = ZDynGNN(
        node_in_dim=34,
        num_rbf=16,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_multibody=True,
        multibody_dim=6,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_calib_loss = float("inf")
    ckpt_path = Path(f"checkpoints/zdyn_gnn_multibody_{condition}.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n[*] Training Multi-Body Z-DynGNN...")
    t0 = time.time()
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
                node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
                mut_positions=b_pos, mut_wt_idx=b_wt, mut_aa_idx=b_mut,
                multibody_feats=node_mb_feats,
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
                multibody_feats=node_mb_feats,
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
                "use_multibody": True,
            }, ckpt_path)

        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"    Epoch {epoch:2d}/{epochs:2d} | Train NLL: {epoch_train_loss:.4f} | "
                  f"Calib NLL: {calib_loss:.4f} | Calib Pearson: {calib_metrics['pearson_raw']:.4f} | "
                  f"Calib R2: {calib_metrics['r2_raw']:.4f} {'*' if is_best else ''}")

    print(f"[+] Multi-body training complete in {time.time() - t0:.1f}s. Checkpoint: {ckpt_path}")

    # 3. Load Best Multi-Body Model for Final Evaluation
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        mu_calib, sigma_calib = model(
            node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
            mut_positions=calib_tensors["pos_idx"],
            mut_wt_idx=calib_tensors["wt_idx"],
            mut_aa_idx=calib_tensors["mut_idx"],
            multibody_feats=node_mb_feats,
        )
        mu_test, sigma_test = model(
            node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
            mut_positions=test_tensors["pos_idx"],
            mut_wt_idx=test_tensors["wt_idx"],
            mut_aa_idx=test_tensors["mut_idx"],
            multibody_feats=node_mb_feats,
        )

    y_test_np = test_tensors["y"].cpu().numpy().flatten()
    mu_test_np = mu_test.cpu().numpy().flatten()
    sigma_test_np = sigma_test.cpu().numpy().flatten()

    y_calib_np = calib_tensors["y"].cpu().numpy().flatten()
    mu_calib_np = mu_calib.cpu().numpy().flatten()
    sigma_calib_np = sigma_calib.cpu().numpy().flatten()

    # 4. Conformal Calibration (Cluster-Adjusted, q_agg=0.90)
    calib_eval_df = calib_df.copy()
    calib_eval_df["pred_mu"] = mu_calib_np
    calib_eval_df["pred_sigma"] = sigma_calib_np

    test_eval_df = test_df.copy()
    test_eval_df["pred_mu"] = mu_test_np
    test_eval_df["pred_sigma"] = sigma_test_np

    cert, bounds_df = calibrate_cluster_conformal(
        calib_eval_df,
        test_eval_df,
        pos_col="pos",
        score_col="fitness_score",
        mu_col="pred_mu",
        sigma_col="pred_sigma",
        alpha=0.05,
        agg_quantile=0.90,
        condition=condition,
    )
    lower_bounds = bounds_df["lower_bound"].values
    upper_bounds = bounds_df["upper_bound"].values
    emp_coverage = cert.empirical_coverage * 100.0 if cert.empirical_coverage <= 1.0 else cert.empirical_coverage
    mean_width = cert.mean_interval_width
    median_width = cert.median_interval_width
    # Decoupled Triage
    sub_wt_decisions = np.sum(upper_bounds < 0.0)
    active_decisions = np.sum(lower_bounds >= -5.0)
    decisive_calls = sub_wt_decisions + active_decisions
    decisive_rate = (decisive_calls / len(y_test_np)) * 100.0

    # Safety Errors (U < -2.0 while true y >= -2.0)
    false_safety = np.sum((upper_bounds < -2.0) & (y_test_np >= -2.0))

    # Regression metrics
    mb_test_metrics = compute_metrics(y_test_np, mu_test_np, ceiling_res.r2_max, ceiling_res.r_max)

    # 5. Load Baseline Results for Side-by-Side Comparison (if baseline exists)
    baseline_ckpt_path = Path(f"checkpoints/zdyn_gnn_{condition}.pt")
    arc_ckpt_path = Path(f"archive/checkpoints/zdyn_gnn_{condition}.pt")
    has_baseline = baseline_ckpt_path.exists() or arc_ckpt_path.exists()

    if has_baseline:
        ckpt_to_load = baseline_ckpt_path if baseline_ckpt_path.exists() else arc_ckpt_path
        base_model = ZDynGNN(node_in_dim=34, hidden_dim=64, num_layers=3, use_multibody=False).to(device)
        base_ckpt = torch.load(ckpt_to_load, map_location=device)
        base_model.load_state_dict(base_ckpt["model_state_dict"])
        base_model.eval()

        with torch.no_grad():
            b_mu_calib, b_sig_calib = base_model(
                node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
                mut_positions=calib_tensors["pos_idx"],
                mut_wt_idx=calib_tensors["wt_idx"],
                mut_aa_idx=calib_tensors["mut_idx"],
            )
            b_mu_test, b_sig_test = base_model(
                node_x, edge_index, edge_dist=edge_dist, dci_edge=dci_edge,
                mut_positions=test_tensors["pos_idx"],
                mut_wt_idx=test_tensors["wt_idx"],
                mut_aa_idx=test_tensors["mut_idx"],
            )

        b_mu_test_np = b_mu_test.cpu().numpy().flatten()
        b_sig_test_np = b_sig_test.cpu().numpy().flatten()

        calib_eval_df["pred_mu"] = b_mu_calib.cpu().numpy().flatten()
        calib_eval_df["pred_sigma"] = b_sig_calib.cpu().numpy().flatten()
        test_eval_df["pred_mu"] = b_mu_test_np
        test_eval_df["pred_sigma"] = b_sig_test_np

        b_cert, b_bounds_df = calibrate_cluster_conformal(
            calib_eval_df,
            test_eval_df,
            pos_col="pos",
            score_col="fitness_score",
            mu_col="pred_mu",
            sigma_col="pred_sigma",
            alpha=0.05,
            agg_quantile=0.90,
            condition=condition,
        )
        b_lower = b_bounds_df["lower_bound"].values
        b_upper = b_bounds_df["upper_bound"].values
        b_emp_coverage = b_cert.empirical_coverage * 100.0 if b_cert.empirical_coverage <= 1.0 else b_cert.empirical_coverage
        b_mean_width = b_cert.mean_interval_width
        b_sub_wt = np.sum(b_upper < 0.0)
        b_active = np.sum(b_lower >= -5.0)
        b_decisive = b_sub_wt + b_active
        b_decisive_rate = (b_decisive / len(y_test_np)) * 100.0
        b_false_safety = np.sum((b_upper < -2.0) & (y_test_np >= -2.0))
        base_metrics = compute_metrics(y_test_np, b_mu_test_np, ceiling_res.r2_max, ceiling_res.r_max)
    else:
        b_mu_test_np = np.zeros_like(y_test_np)
        b_sig_test_np = np.zeros_like(y_test_np)
        b_lower = np.zeros_like(y_test_np)
        b_upper = np.zeros_like(y_test_np)
        b_emp_coverage = 0.0
        b_mean_width = 0.0
        b_decisive_rate = 0.0
        b_false_safety = 0
        base_metrics = {
            "pearson_raw": float("nan"),
            "pearson_normalized": float("nan"),
            "r2_raw": float("nan"),
            "rmse": float("nan"),
        }
    # 6. M154L Specific Assessment
    m154_mask = (test_df["pos"] == 154) & (test_df["mut"] == "L")
    if m154_mask.any():
        m154_idx = np.where(m154_mask)[0][0]
        m154_true = float(y_test_np[m154_idx])
        m154_base_mu = float(b_mu_test_np[m154_idx])
        m154_base_sig = float(b_sig_test_np[m154_idx])
        m154_mb_mu = float(mu_test_np[m154_idx])
        m154_mb_sig = float(sigma_test_np[m154_idx])
    else:
        m154_true = m154_base_mu = m154_base_sig = m154_mb_mu = m154_mb_sig = 0.0

    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD BENCHMARK: BASELINE Z-DYNGNN vs. MULTI-BODY UPGRADED Z-DYNGNN")
    print("=" * 80)
    print(f"{'Metric':<36} | {'Baseline (Pairwise)':<20} | {'Multi-Body (Dodd et al.)'}")
    print("-" * 80)
    print(f"{'Test Pearson r':<36} | {base_metrics['pearson_raw']:<20.4f} | {mb_test_metrics['pearson_raw']:.4f}")
    print(f"{'Normalized Pearson (r / r_max)':<36} | {base_metrics['pearson_normalized']*100:<19.2f}% | {mb_test_metrics['pearson_normalized']*100:.2f}%")
    print(f"{'Test R2 (Raw)':<36} | {base_metrics['r2_raw']:<20.4f} | {mb_test_metrics['r2_raw']:.4f}")
    print(f"{'Test RMSE':<36} | {base_metrics['rmse']:<20.4f} | {mb_test_metrics['rmse']:.4f}")
    print(f"{'Conformal Empirical Coverage (95%)':<36} | {b_emp_coverage:<19.2f}% | {emp_coverage:.2f}%")
    print(f"{'Mean Interval Width (2B_i)':<36} | {b_mean_width:<20.2f} | {mean_width:.2f}")
    print(f"{'Decisive Clinical Triage Rate':<36} | {b_decisive_rate:<19.1f}% | {decisive_rate:.1f}%")
    print(f"{'Critical False Safety Errors':<36} | {b_false_safety:<20d} | {false_safety:d}")
    if m154_mask.any():
        print("-" * 80)
        print(f"{'M154L Predicted mu (True = ' + str(round(m154_true, 2)) + ')':<36} | {m154_base_mu:<20.2f} | {m154_mb_mu:.2f}")
        print(f"{'M154L Aleatoric Scale sigma_i':<36} | {m154_base_sig:<20.2f} | {m154_mb_sig:.2f}")

    # Save Comparison
    comparison = {
        "condition": condition,
        "baseline": {
            "metrics": base_metrics,
            "coverage": b_emp_coverage,
            "mean_width": b_mean_width,
            "decisive_rate": b_decisive_rate,
            "false_safety_errors": int(b_false_safety),
            "m154l_mu": m154_base_mu,
            "m154l_sigma": m154_base_sig,
        },
        "multibody": {
            "metrics": mb_test_metrics,
            "coverage": emp_coverage,
            "mean_width": mean_width,
            "decisive_rate": decisive_rate,
            "false_safety_errors": int(false_safety),
            "m154l_mu": m154_mb_mu,
            "m154l_sigma": m154_mb_sig,
        },
    }

    comp_path = Path("results/multibody_model_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[+] Saved comparison JSON to {comp_path}")

    # Save test predictions table
    test_pred_df = test_df.copy()
    test_pred_df["base_mu"] = b_mu_test_np
    test_pred_df["base_sigma"] = b_sig_test_np
    test_pred_df["base_lower"] = b_lower
    test_pred_df["base_upper"] = b_upper
    test_pred_df["mb_mu"] = mu_test_np
    test_pred_df["mb_sigma"] = sigma_test_np
    test_pred_df["mb_lower"] = lower_bounds
    test_pred_df["mb_upper"] = upper_bounds

    pred_path = Path(f"results/multibody_predictions_{condition}.csv")
    test_pred_df.to_csv(pred_path, index=False)
    print(f"[+] Saved test predictions to {pred_path}")

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="256.0_AMP")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--multibody_source", default="3spu", choices=["3spu", "5zgy", "5o2e", "4hl2", "4rl0"])
    args = parser.parse_args()

    train_and_benchmark_multibody(
        condition=args.condition,
        epochs=args.epochs,
        device_str=args.device,
        multibody_source=args.multibody_source,
    )
