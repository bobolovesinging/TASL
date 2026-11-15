"""MNIST and Fashion-MNIST data for audited TASL training."""

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST, MNIST


def dirichlet_partition(labels, n_clients, alpha, seed):
    rng = np.random.RandomState(seed)
    n_classes = len(np.unique(labels))
    client_indices = {i: [] for i in range(n_clients)}
    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        proportions = proportions / proportions.sum()
        splits = (proportions * len(class_idx)).astype(int)
        remainder = len(class_idx) - splits.sum()
        for r in range(remainder):
            splits[r % n_clients] += 1
        start = 0
        for i in range(n_clients):
            client_indices[i].extend(class_idx[start:start + splits[i]].tolist())
            start += splits[i]
    return client_indices


def load_client_data(dataset, root, *, clients, batch_size, alpha, seed):
    """Normalize grayscale images and partition the training set by label."""
    source = {"mnist": MNIST, "fashionmnist": FashionMNIST}[dataset]
    train = source(root, train=True, download=True)
    test = source(root, train=False, download=True)
    train_images = train.data[:, None].float() / 255.0
    test_images = test.data[:, None].float() / 255.0
    train_labels = train.targets.numpy()
    indices = dirichlet_partition(train_labels, clients, alpha=alpha, seed=seed)
    client_loaders = [
        DataLoader(
            TensorDataset(train_images[ids], train.targets[ids]),
            batch_size=batch_size,
            shuffle=True,
        )
        for ids in indices.values()
    ]
    test_loader = DataLoader(
        TensorDataset(test_images, test.targets),
        batch_size=batch_size,
    )
    return client_loaders, test_loader
