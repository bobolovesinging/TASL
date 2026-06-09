# -*- coding: utf-8 -*-
"""
NTU-60 Data Loader with FedPure-style Augmentation.
Supports: valid_crop_resize (temporal crop) + random_rot (skeleton rotation).
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ── Import FedPure tools ─────────────────────────────────────────────
FEDPURE_DIR = "/media/njit5/39e16d05-05ba-46b4-bc9b-47b3c98f1d4f/tlf/fedpure"
sys.path.insert(0, FEDPURE_DIR)
from feeders import tools


class NTU60AugDataset(Dataset):
    """NTU-60 dataset with FedPure augmentation: temporal crop + random rotation."""

    def __init__(self, x, y, window_size=64, random_rot=True, p_interval=[0.5, 1.0],
                 is_train=True):
        """
        Args:
            x: numpy array (N, C, T, V, M) = (N, 3, 50, 25, 2)
            y: numpy array (N,) labels
            window_size: output temporal length
            random_rot: apply random skeleton rotation
            p_interval: temporal crop ratio range (train: [0.5,1.0], test: [0.95])
            is_train: training mode (different augmentation behavior)
        """
        self.x = x
        self.y = y
        self.window_size = window_size
        self.random_rot = random_rot
        self.p_interval = p_interval
        self.is_train = is_train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        data = self.x[idx].copy()  # (C, T, V, M) = (3, 50, 25, 2)
        label = int(self.y[idx])

        # Count valid frames
        valid_frame_num = int(np.sum(data.sum(0).sum(-1).sum(-1) != 0))
        # Clamp to at least 64
        actual_T = data.shape[1]
        valid_frame_num = max(valid_frame_num, min(actual_T, self.window_size))

        # ── Temporal Crop + Resize ──
        data = tools.valid_crop_resize(
            data, valid_frame_num, self.p_interval, self.window_size)
        # data is now (C, window_size, V, M) = (3, 64, 25, 2)

        # ── Random Rotation (training only) ──
        if self.random_rot and self.is_train:
            data = tools.random_rot(data, theta=0.3)
            data = data.numpy()  # random_rot returns torch tensor

        # ── Convert to torch ──
        data = torch.from_numpy(data).float()  # (C, T, V, M)
        label = torch.tensor(label, dtype=torch.long)

        return data, label


def load_ntu60_augmented(data_dir, n_clients, batch_size=64, window_size=64,
                         random_rot=True, p_interval_train=[0.5, 1.0],
                         test_p_interval=0.95):
    """
    Load NTU-60 federated data WITH FedPure augmentation.

    Returns:
        client_loaders: list of DataLoader (training, with augmentation)
        test_loader: DataLoader (testing, without random_rot, center crop)
    """
    client_loaders = []

    for cid in range(n_clients):
        raw = np.load(
            os.path.join(data_dir, 'fedtrain', f'{cid}.npz'),
            allow_pickle=True
        )['data'].item()
        x, y = raw['x'], raw['y'].flatten()

        ds = NTU60AugDataset(
            x, y,
            window_size=window_size,
            random_rot=random_rot,
            p_interval=p_interval_train,
            is_train=True,
        )
        client_loaders.append(
            DataLoader(ds, batch_size=batch_size, shuffle=True))

    # Test loader: no random_rot, center crop
    raw_test = np.load(
        os.path.join(data_dir, 'fedtest', '0.npz'),
        allow_pickle=True
    )['data'].item()
    test_x, test_y = raw_test['x'], raw_test['y'].flatten()

    test_ds = NTU60AugDataset(
        test_x, test_y,
        window_size=window_size,
        random_rot=False,
        p_interval=[test_p_interval],
        is_train=False,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return client_loaders, test_loader
