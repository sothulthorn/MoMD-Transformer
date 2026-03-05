"""
Dataset classes for MoMD Transformer experiments.

Supports two datasets from the paper:
  - PU: Paderborn University bearing dataset (Table 2)
  - PMSM: Permanent Magnet Synchronous Motor stator dataset (Table 4)

Expected directory structure after preprocessing (run preprocess.py first):
  data/pu/
    vibration.npy   # (num_samples, 2048)
    current.npy     # (num_samples, 2048)
    labels.npy      # (num_samples,) with values in {0, 1, 2}
  data/pmsm/
    vibration.npy
    current.npy
    labels.npy
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class MultiModalFaultDataset(Dataset):
    """
    Multi-modal fault diagnosis dataset with paired vibration-current signals.

    Each sample is a tuple (vibration_signal, current_signal, label).
    """

    def __init__(self, vibration, current, labels):
        """
        Args:
            vibration: np.ndarray of shape (N, signal_length)
            current: np.ndarray of shape (N, signal_length)
            labels: np.ndarray of shape (N,)
        """
        self.vibration = torch.from_numpy(vibration).float()
        self.current = torch.from_numpy(current).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.vibration[idx], self.current[idx], self.labels[idx]


def load_dataset(data_dir):
    """Load preprocessed .npy files from data_dir."""
    vibration = np.load(os.path.join(data_dir, "vibration.npy"))
    current = np.load(os.path.join(data_dir, "current.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))
    return MultiModalFaultDataset(vibration, current, labels)


def split_dataset(dataset, split_ratio=(0.6, 0.2, 0.2), seed=42):
    """
    Split dataset into train/val/test subsets (default 3:1:1).

    Performs stratified split to maintain class balance.
    """
    rng = np.random.RandomState(seed)
    labels = dataset.labels.numpy()
    classes = np.unique(labels)

    train_idx, val_idx, test_idx = [], [], []

    for c in classes:
        c_indices = np.where(labels == c)[0]
        rng.shuffle(c_indices)
        n = len(c_indices)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        train_idx.extend(c_indices[:n_train])
        val_idx.extend(c_indices[n_train : n_train + n_val])
        test_idx.extend(c_indices[n_train + n_val :])

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def get_dataloaders(data_dir, batch_size=64, split_ratio=(0.6, 0.2, 0.2),
                    num_workers=4, seed=42):
    """Load data, split, and return DataLoaders."""
    dataset = load_dataset(data_dir)
    train_set, val_set, test_set = split_dataset(dataset, split_ratio, seed)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
