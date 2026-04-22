from __future__ import annotations

import torch

from models.Update import apply_model_delta
from options.privacy_utils import CountSketch, PaillierAHE, dequantize_tensor
from options.support_utils import HistoricalSupportTracker


class SecureAggregator:
    def __init__(
        self,
        model: torch.nn.Module,
        count_sketch: CountSketch,
        he: PaillierAHE,
        support_tracker: HistoricalSupportTracker,
        quantization_scale_exp: int,
        device: str,
    ) -> None:
        self.model = model
        self.count_sketch = count_sketch
        self.he = he
        self.support_tracker = support_tracker
        self.quantization_scale_exp = quantization_scale_exp
        self.device = device
        self.model_dim = sum(parameter.numel() for parameter in self.model.parameters())

    def aggregate(self, encrypted_sketches: list[list[list[int]]]) -> tuple[torch.Tensor, dict[str, float]]:
        aggregated_ciphertext = self.he.aggregate_ciphertexts(encrypted_sketches)
        aggregated_quantized_sketch = self.he.decrypt_tensor(aggregated_ciphertext, self.device)
        aggregated_sketch = dequantize_tensor(aggregated_quantized_sketch, self.quantization_scale_exp)
        weighted_average_update = self.count_sketch.recover(aggregated_sketch, self.model_dim)
        stabilized_update, support_stats = self.support_tracker.reweight(weighted_average_update)
        return stabilized_update, support_stats

    @torch.no_grad()
    def apply_update(self, update_vector: torch.Tensor) -> None:
        apply_model_delta(self.model, update_vector)
