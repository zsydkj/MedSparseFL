from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from models.Update import (
    build_block_gsn_features,
    build_ste_sparse_update,
    clone_parameter_state,
    flatten_parameter_delta,
    recalibrate_probabilities,
    sparsify_with_residual,
    straight_through_block_mask,
)
from options.privacy_utils import CountSketch, PairwiseMasking, PaillierAHE, quantize_tensor


@dataclass
class ClientUpload:
    client_id: int
    encrypted_sketch: list[list[int]]
    num_samples: int
    aggregation_weight: float
    upload_density: float
    sparse_update_norm: float


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        dataloader,
        model: torch.nn.Module,
        gsn: torch.nn.Module,
        gsn_optimizer: torch.optim.Optimizer,
        lr: float,
        device: str,
        total_clients: int,
        local_sample_count: int,
        task_type: str,
        target_upload_ratio: float,
        ema_decay: float,
        gsn_reg_lambda: float,
        count_sketch: CountSketch,
        masking: PairwiseMasking,
        he: PaillierAHE,
        quantization_scale_exp: int,
        block_size: int,
    ) -> None:
        self.client_id = client_id
        self.dataloader = dataloader
        self.model = model.to(device)
        self.gsn = gsn.to(device)
        self.gsn_optimizer = gsn_optimizer
        self.lr = lr
        self.device = device
        self.total_clients = total_clients
        self.local_sample_count = local_sample_count
        self.task_type = task_type
        self.target_upload_ratio = target_upload_ratio
        self.ema_decay = ema_decay
        self.gsn_reg_lambda = gsn_reg_lambda
        self.count_sketch = count_sketch
        self.masking = masking
        self.he = he
        self.quantization_scale_exp = quantization_scale_exp
        self.block_size = block_size
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.residual: Optional[torch.Tensor] = None
        self.ema_state: Optional[torch.Tensor] = None

    @property
    def num_samples(self) -> int:
        return self.local_sample_count

    def load_global_model(self, global_state: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(global_state, strict=True)

    def _client_feature(self, aggregation_weight: float) -> torch.Tensor:
        feature = torch.zeros(1 + self.total_clients, device=self.device, dtype=torch.float32)
        feature[0] = float(aggregation_weight)
        feature[1 + self.client_id] = 1.0
        return feature

    def _criterion(self) -> nn.Module:
        return nn.BCEWithLogitsLoss() if self.task_type == "multilabel" else nn.CrossEntropyLoss()

    def local_train(
        self,
        round_idx: int,
        participant_ids: Sequence[int],
        global_reference_update: Optional[torch.Tensor],
        local_epochs: int,
        aggregation_weight: float,
    ) -> ClientUpload:
        criterion = self._criterion()
        reference_state = clone_parameter_state(self.model)
        self.model.train()

        for _ in range(local_epochs):
            for inputs, targets in self.dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(inputs)
                if self.task_type == "multiclass":
                    targets = targets.long()
                else:
                    targets = targets.float()
                loss = criterion(logits, targets)
                loss.backward()
                self.optimizer.step()

        dense_update = flatten_parameter_delta(self.model, reference_state)
        corrected_update = dense_update if self.residual is None else dense_update + self.residual
        block_features, updated_ema = build_block_gsn_features(
            corrected_update,
            self.ema_state,
            global_reference_update,
            self.block_size,
            self.ema_decay,
        )
        client_feature = self._client_feature(aggregation_weight)
        raw_probabilities = self.gsn(block_features, client_feature)
        block_probabilities = recalibrate_probabilities(raw_probabilities, self.target_upload_ratio)
        hard_mask, ste_mask = straight_through_block_mask(block_probabilities)
        _, sparse_update, updated_residual = sparsify_with_residual(dense_update, hard_mask, self.residual, self.block_size)

        ste_sparse_update = build_ste_sparse_update(dense_update, ste_mask, self.residual, self.block_size)
        self.gsn_optimizer.zero_grad()
        proxy_loss = torch.mean((ste_sparse_update - dense_update.detach()) ** 2) + self.gsn_reg_lambda * block_probabilities.mean()
        proxy_loss.backward()
        self.gsn_optimizer.step()

        self.residual = updated_residual.detach()
        self.ema_state = updated_ema.detach()

        weighted_sparse_update = sparse_update * float(aggregation_weight)
        sketch = self.count_sketch.sketch(weighted_sparse_update)
        quantized_sketch = quantize_tensor(sketch, self.quantization_scale_exp)
        pairwise_mask = self.masking.generate_mask(self.client_id, participant_ids, round_idx, quantized_sketch.shape, self.device)
        masked_sketch = quantized_sketch + pairwise_mask
        encrypted_sketch = self.he.encrypt_tensor(masked_sketch)

        return ClientUpload(
            client_id=self.client_id,
            encrypted_sketch=encrypted_sketch,
            num_samples=self.local_sample_count,
            aggregation_weight=float(aggregation_weight),
            upload_density=float(hard_mask.mean().item()),
            sparse_update_norm=float(sparse_update.norm().item()),
        )
