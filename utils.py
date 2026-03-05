"""
Evaluation and visualization utilities for MoMD Transformer.
"""

import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE


@torch.no_grad()
def evaluate(model, data_loader, device, mode="both"):
    """
    Evaluate model on a given loader.

    Args:
        mode: 'vibration', 'current', or 'both'
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x_vib, x_cur, labels in data_loader:
        x_vib = x_vib.to(device)
        x_cur = x_cur.to(device)
        labels = labels.to(device)

        if mode == "vibration":
            out = model(x_vib=x_vib)
        elif mode == "current":
            out = model(x_cur=x_cur)
        else:
            out = model(x_vib=x_vib, x_cur=x_cur)

        preds = out["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total * 100
    return acc, np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def extract_features(model, data_loader, device, mode="both"):
    """
    Extract class-token features from the last encoder block for t-SNE.

    Returns:
        features: (N, embed_dim)
        labels: (N,)
    """
    model.eval()
    all_features = []
    all_labels = []

    for x_vib, x_cur, labels in data_loader:
        x_vib = x_vib.to(device)
        x_cur = x_cur.to(device)

        if mode == "vibration":
            out = model(x_vib=x_vib, return_block_cls=True)
        elif mode == "current":
            out = model(x_cur=x_cur, return_block_cls=True)
        else:
            out = model(x_vib=x_vib, x_cur=x_cur, return_block_cls=True)

        # Use the class embedding from the last block
        cls_feat = out["block_cls"][-1]  # (B, embed_dim)
        all_features.append(cls_feat.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.concatenate(all_features), np.array(all_labels)


def save_training_history(history, filepath):
    """Save training history list-of-dicts to CSV."""
    if not history:
        return
    keys = history[0].keys()
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    print(f"  Saved {filepath}")


def plot_training_curves(history, filepath):
    """Plot loss and accuracy curves over epochs."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, [h["loss"] for h in history], label="Total Loss", linewidth=2)
    ax.plot(epochs, [h["loss_d"] for h in history], label="L_D", linestyle="--")
    ax.plot(epochs, [h["loss_gkt"] for h in history], label="L_GKT", linestyle="--")
    ax.plot(epochs, [h["loss_msm"] for h in history], label="L_MSM", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy curves
    ax = axes[1]
    ax.plot(epochs, [h["train_acc"] for h in history], label="Train (MM)", linewidth=2)
    ax.plot(epochs, [h["val_cur"] for h in history], label="Val Current", linewidth=2)
    ax.plot(epochs, [h["val_vib"] for h in history], label="Val Vibration", linewidth=2)
    ax.plot(epochs, [h["val_both"] for h in history], label="Val Both", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filepath}")


def plot_confusion_matrices(model, test_loader, device, label_names, filepath):
    """Plot confusion matrices for all three input modes side by side."""
    modes = ["current", "vibration", "both"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, mode in zip(axes, modes):
        _, preds, labels = evaluate(model, test_loader, device, mode=mode)
        cm = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[label_names[i] for i in range(len(label_names))],
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        acc = np.trace(cm) / cm.sum() * 100
        ax.set_title(f"{mode.capitalize()} ({acc:.2f}%)")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filepath}")


def plot_tsne(model, test_loader, device, label_names, filepath):
    """
    Plot t-SNE of class-token features for all three input modes (Fig. 12).
    """
    modes = ["current", "vibration", "both"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.Set1(np.linspace(0, 1, len(label_names)))

    for ax, mode in zip(axes, modes):
        features, labels = extract_features(model, test_loader, device, mode=mode)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels) - 1))
        embedded = tsne.fit_transform(features)

        for c in range(len(label_names)):
            mask = labels == c
            ax.scatter(
                embedded[mask, 0], embedded[mask, 1],
                c=[colors[c]], label=label_names[c], s=20, alpha=0.7,
            )
        ax.set_title(f"{mode.capitalize()}")
        ax.legend(fontsize=8, markerscale=2)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("t-SNE Feature Visualization", fontsize=14)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filepath}")
