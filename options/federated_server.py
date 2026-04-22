from __future__ import annotations

import copy
import random
from typing import Sequence

import torch

from options.aggregation_utils import SecureAggregator


class FederatedServer:
    def __init__(self, clients: list, global_model: torch.nn.Module, aggregator: SecureAggregator, client_fraction: float, device: str) -> None:
        self.clients = clients
        self.global_model = global_model.to(device)
        self.aggregator = aggregator
        self.client_fraction = client_fraction
        self.device = device
        self.previous_global_update: torch.Tensor | None = None

    def sample_clients(self) -> list:
        count = max(1, int(round(self.client_fraction * len(self.clients))))
        return random.sample(self.clients, count)

    @torch.no_grad()
    def broadcast_model(self, selected_clients: Sequence) -> None:
        global_state = copy.deepcopy(self.global_model.state_dict())
        for client in selected_clients:
            client.load_global_model(global_state)

    def train_round(self, round_idx: int, local_epochs: int) -> dict[str, float]:
        selected_clients = self.sample_clients()
        participant_ids = [client.client_id for client in selected_clients]
        total_selected_samples = sum(client.num_samples for client in selected_clients)
        self.broadcast_model(selected_clients)

        uploads = []
        for client in selected_clients:
            aggregation_weight = client.num_samples / max(total_selected_samples, 1)
            uploads.append(client.local_train(round_idx, participant_ids, self.previous_global_update, local_epochs, aggregation_weight))

        recovered_update, support_stats = self.aggregator.aggregate([item.encrypted_sketch for item in uploads])
        self.aggregator.apply_update(recovered_update)
        self.previous_global_update = recovered_update.detach().clone()
        return {
            "num_selected_clients": float(len(selected_clients)),
            "mean_upload_density": float(sum(item.upload_density for item in uploads) / len(uploads)),
            "mean_sparse_update_norm": float(sum(item.sparse_update_norm for item in uploads) / len(uploads)),
            "support_consistency": float(support_stats["consistency"]),
            "stable_ratio": float(support_stats["stable_ratio"]),
        }
