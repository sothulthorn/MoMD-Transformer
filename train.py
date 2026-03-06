"""
Training and evaluation script for MoMD Transformer.

Training procedure (Section 3.5, Fig. 5):
  For each batch of paired (vibration, current, label) samples:
    1. Forward vibration only  -> L_V_D, block_cls_vib
    2. Forward current only    -> L_C_D, block_cls_cur
    3. Forward both modalities -> L_VC_D
    4. Forward both with mask  -> L_msm
    5. L_D = (L_V + L_C + L_VC) / 3
    6. L_gkt from block_cls_vib and block_cls_cur
    7. L_all = L_D + lambda1 * L_gkt + lambda2 * L_msm
"""

import argparse
import csv
import os
import time
import numpy as np
import math
import torch
import torch.nn as nn

from tqdm import tqdm

import config
from model import MoMDTransformer
from dataset import get_dataloaders
from utils import (
    evaluate,
    save_training_history,
    plot_training_curves,
    plot_confusion_matrices,
    plot_tsne,
)


def train_one_epoch(model, train_loader, optimizer, device, mask_ratio, lambda_gkt, lambda_msm,
                    scheduler=None, max_grad_norm=0.0):
    model.train()
    total_loss = 0.0
    total_ld = 0.0
    total_gkt = 0.0
    total_msm = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for x_vib, x_cur, labels in pbar:
        x_vib = x_vib.to(device)
        x_cur = x_cur.to(device)
        labels = labels.to(device)

        # --- Pass 1: Vibration only (with block_cls for GKT) ---
        out_v = model(x_vib=x_vib, return_block_cls=True)
        loss_v = criterion(out_v["logits"], labels)

        # --- Pass 2: Current only (with block_cls for GKT) ---
        out_c = model(x_cur=x_cur, return_block_cls=True)
        loss_c = criterion(out_c["logits"], labels)

        # --- Pass 3: Multi-modal (no mask) ---
        out_vc = model(x_vib=x_vib, x_cur=x_cur)
        loss_vc = criterion(out_vc["logits"], labels)

        # --- Diagnosis loss (Eq. 20) ---
        loss_d = (loss_v + loss_c + loss_vc) / 3.0

        # --- GKT loss (Eq. 17) ---
        loss_gkt = MoMDTransformer.compute_gkt_loss(
            out_v["block_cls"], out_c["block_cls"]
        )

        # --- Pass 4: Multi-modal with masking for MSM ---
        out_msm = model(x_vib=x_vib, x_cur=x_cur, mask_ratio=mask_ratio)
        loss_msm = out_msm["msm_loss"]

        # --- Total loss (Eq. 21) ---
        loss_all = loss_d + lambda_gkt * loss_gkt + lambda_msm * loss_msm

        optimizer.zero_grad()
        loss_all.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Track metrics (using multi-modal predictions)
        total_loss += loss_all.item() * labels.size(0)
        total_ld += loss_d.item() * labels.size(0)
        total_gkt += loss_gkt.item() * labels.size(0)
        total_msm += loss_msm.item() * labels.size(0)
        preds = out_vc["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=total_loss / total, acc=correct / total * 100)

    n = total
    return {
        "loss": total_loss / n,
        "loss_d": total_ld / n,
        "loss_gkt": total_gkt / n,
        "loss_msm": total_msm / n,
        "acc": correct / n * 100,
    }


def run_experiment(args, seed, run_dir, label_names):
    """Run a single training experiment with a given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    data_config = config.PU_CONFIG if args.dataset == "pu" else config.PMSM_CONFIG
    num_classes = data_config["num_classes"]
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_config["data_dir"],
        batch_size=args.batch_size,
        split_ratio=data_config["split_ratio"],
        num_workers=args.num_workers,
        seed=seed,
    )

    # --- Model ---
    model = MoMDTransformer(
        num_classes=num_classes,
        signal_length=config.SIGNAL_LENGTH,
        segment_length=config.SEGMENT_LENGTH,
        embed_dim=config.EMBED_DIM,
        mlp_dim=config.MLP_DIM,
        num_heads=config.NUM_HEADS,
        depth=config.BLOCK_DEPTH,
        dropout=config.DROPOUT,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY,
    )

    # --- LR Schedule: linear warmup + cosine decay (per step) ---
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = config.WARMUP_EPOCHS * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Training ---
    best_val_acc = 0.0
    best_model_state = None
    history = []

    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Epochs")
    for epoch in epoch_pbar:
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            mask_ratio=config.MASK_RATIO,
            lambda_gkt=config.LAMBDA_GKT,
            lambda_msm=config.LAMBDA_MSM,
            scheduler=scheduler,
            max_grad_norm=config.MAX_GRAD_NORM,
        )

        # Per-modality validation
        val_acc_cur, _, _ = evaluate(model, val_loader, device, mode="current")
        val_acc_vib, _, _ = evaluate(model, val_loader, device, mode="vibration")
        val_acc_both, _, _ = evaluate(model, val_loader, device, mode="both")
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "loss": train_metrics["loss"],
            "loss_d": train_metrics["loss_d"],
            "loss_gkt": train_metrics["loss_gkt"],
            "loss_msm": train_metrics["loss_msm"],
            "train_acc": train_metrics["acc"],
            "val_cur": val_acc_cur,
            "val_vib": val_acc_vib,
            "val_both": val_acc_both,
            "time": elapsed,
        })

        epoch_pbar.set_postfix(
            loss=f"{train_metrics['loss']:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            cur=f"{val_acc_cur:.1f}%",
            vib=f"{val_acc_vib:.1f}%",
            both=f"{val_acc_both:.1f}%",
        )

        if val_acc_both > best_val_acc:
            best_val_acc = val_acc_both
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # --- Load best model ---
    model.load_state_dict(best_model_state)
    model.to(device)

    # --- Test ---
    results = {}
    for mode in ["current", "vibration", "both"]:
        acc, _, _ = evaluate(model, test_loader, device, mode=mode)
        results[mode] = acc
        print(f"  Test Acc ({mode:>9s}): {acc:.2f}%")

    # --- Save outputs for this run ---
    os.makedirs(run_dir, exist_ok=True)

    save_training_history(history, os.path.join(run_dir, "training_history.csv"))
    plot_training_curves(history, os.path.join(run_dir, "training_curve.png"))
    plot_confusion_matrices(
        model, test_loader, device, label_names,
        os.path.join(run_dir, "confusion_matrix.png"),
    )
    plot_tsne(
        model, test_loader, device, label_names,
        os.path.join(run_dir, "tsne.png"),
    )

    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    return results, model


def main():
    parser = argparse.ArgumentParser(description="MoMD Transformer Training")
    parser.add_argument("--dataset", type=str, default="pu", choices=["pu", "pmsm"])
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--repeats", type=int, default=config.NUM_REPEATS)
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_config = config.PU_CONFIG if args.dataset == "pu" else config.PMSM_CONFIG
    label_names = data_config["label_names"]

    all_results = {"current": [], "vibration": [], "both": []}
    best_overall_acc = 0.0
    best_run_model_path = None

    for run in range(1, args.repeats + 1):
        seed = config.SEED + run - 1
        run_dir = os.path.join(args.output_dir, args.dataset, f"run_{run}")
        print(f"\n{'='*60}")
        print(f"Run {run}/{args.repeats} (seed={seed})")
        print(f"{'='*60}")

        results, _ = run_experiment(args, seed, run_dir, label_names)

        for mode in ["current", "vibration", "both"]:
            all_results[mode].append(results[mode])

        if results["both"] > best_overall_acc:
            best_overall_acc = results["both"]
            best_run_model_path = os.path.join(run_dir, "model.pt")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Results over {args.repeats} runs:")
    print(f"{'='*60}")
    for mode in ["current", "vibration", "both"]:
        accs = np.array(all_results[mode])
        print(f"  {mode:>9s}: {accs.mean():.2f} +/- {accs.std():.2f}%")

    # Save summary CSV
    summary_path = os.path.join(args.output_dir, args.dataset, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "current_acc", "vibration_acc", "both_acc"])
        for i in range(args.repeats):
            writer.writerow([
                i + 1,
                all_results["current"][i],
                all_results["vibration"][i],
                all_results["both"][i],
            ])
        writer.writerow([
            "mean",
            np.mean(all_results["current"]),
            np.mean(all_results["vibration"]),
            np.mean(all_results["both"]),
        ])
        writer.writerow([
            "std",
            np.std(all_results["current"]),
            np.std(all_results["vibration"]),
            np.std(all_results["both"]),
        ])
    print(f"\nSummary saved to {summary_path}")

    # --- Final visualizations from best run ---
    print(f"\nGenerating final plots from best model ({best_run_model_path})...")
    num_classes = data_config["num_classes"]
    model = MoMDTransformer(
        num_classes=num_classes,
        signal_length=config.SIGNAL_LENGTH,
        segment_length=config.SEGMENT_LENGTH,
        embed_dim=config.EMBED_DIM,
        mlp_dim=config.MLP_DIM,
        num_heads=config.NUM_HEADS,
        depth=config.BLOCK_DEPTH,
        dropout=config.DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(best_run_model_path, map_location=device, weights_only=True))

    _, _, test_loader = get_dataloaders(
        data_dir=data_config["data_dir"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dataset_dir = os.path.join(args.output_dir, args.dataset)
    plot_confusion_matrices(
        model, test_loader, device, label_names,
        os.path.join(dataset_dir, "confusion_matrix.png"),
    )
    plot_tsne(
        model, test_loader, device, label_names,
        os.path.join(dataset_dir, "tsne.png"),
    )


if __name__ == "__main__":
    main()
