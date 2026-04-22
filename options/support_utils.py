from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import torch


class HistoricalSupportTracker:
    def __init__(
        self,
        vector_dim: int,
        support_ratio: float = 0.1,
        window_size: int = 5,
        epsilon: float = 0.3,
        min_frequency: float = 0.6,
    ) -> None:
        self.vector_dim = vector_dim
        self.support_ratio = support_ratio
        self.window_size = window_size
        self.epsilon = epsilon
        self.min_frequency = min_frequency
        self.history: Deque[torch.Tensor] = deque(maxlen=window_size)

    def extract_support(self, vector: torch.Tensor) -> torch.Tensor:
        k = max(1, int(self.support_ratio * vector.numel()))
        support = torch.zeros_like(vector, dtype=torch.bool)
        indices = torch.topk(vector.abs(), k=k, largest=True).indices
        support[indices] = True
        return support

    def stable_reference(self) -> Optional[torch.Tensor]:
        if not self.history:
            return None
        stacked = torch.stack([mask.float() for mask in self.history], dim=0)
        return stacked.mean(dim=0) >= self.min_frequency

    def reweight(self, update_vector: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        current_support = self.extract_support(update_vector)
        stable_support = self.stable_reference()
        if stable_support is None:
            self.history.append(current_support.detach().cpu())
            return update_vector, {"consistency": 1.0, "stable_ratio": float(current_support.float().mean().item())}
        stable_support = stable_support.to(update_vector.device)
        overlap = (current_support & stable_support).float().sum()
        current_count = current_support.float().sum().clamp_min(1.0)
        consistency = float((overlap / current_count).item())
        weights = torch.full_like(update_vector, self.epsilon)
        weights[stable_support] = 1.0
        stabilized = update_vector * weights
        self.history.append(current_support.detach().cpu())
        return stabilized, {"consistency": consistency, "stable_ratio": float(stable_support.float().mean().item())}
