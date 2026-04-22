from __future__ import annotations

import numpy as np
from torch.utils.data import Subset

def _extract_targets(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        targets = np.asarray(dataset.targets)
    elif hasattr(dataset, "labels"):
        targets = np.asarray(dataset.labels)
    else:
        raise ValueError("Dataset must expose targets or labels for non-IID splitting")
    if targets.ndim == 1:
        return targets.astype(int)
    if targets.ndim == 2:
        primary = targets.argmax(axis=1)
        all_zero = targets.sum(axis=1) == 0
        primary[all_zero] = 0
        return primary.astype(int)
    raise ValueError("Unsupported target shape")

def non_iid_split(dataset, num_clients: int, alpha: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    targets = _extract_targets(dataset)
    num_classes = int(targets.max()) + 1
    class_indices = [np.where(targets == class_id)[0] for class_id in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    for indices in class_indices:
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        for client_id, split in enumerate(np.split(indices, split_points)):
            client_indices[client_id].extend(split.tolist())
    if all(len(indices) == 0 for indices in client_indices):
        raise ValueError("Dirichlet split produced no samples")
    for client_id, indices in enumerate(client_indices):
        if not indices:
            donor_id = int(np.argmax([len(item) for item in client_indices]))
            client_indices[client_id].append(client_indices[donor_id].pop())
    return [Subset(dataset, indices) for indices in client_indices]
