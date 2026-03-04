import numpy as np
from torch.utils.data import DataLoader, Subset


# Split dataset into non-IID subsets using Dirichlet distribution
def non_iid_split(dataset, num_clients, alpha=0.5):
    labels = np.array([y for _, y in dataset])
    class_indices = [np.where(labels == i)[0] for i in np.unique(labels)]
    client_indices = [[] for _ in range(num_clients)]

    for cls_idx in class_indices:
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        splits = np.split(cls_idx, proportions)
        for i, idx in enumerate(splits):
            client_indices[i].extend(idx.tolist())
    return [Subset(dataset, idx) for idx in client_indices]