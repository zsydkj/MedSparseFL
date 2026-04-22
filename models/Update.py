from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def clone_parameter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


@torch.no_grad()
def flatten_parameter_delta(model: torch.nn.Module, reference_state: dict[str, torch.Tensor]) -> torch.Tensor:
    chunks = []
    for name, parameter in model.named_parameters():
        chunks.append((parameter.detach() - reference_state[name].to(parameter.device)).reshape(-1))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def apply_model_delta(model: torch.nn.Module, delta_vector: torch.Tensor) -> None:
    pointer = 0
    for parameter in model.parameters():
        numel = parameter.numel()
        parameter.add_(delta_vector[pointer:pointer + numel].view_as(parameter).to(parameter.device, parameter.dtype))
        pointer += numel


@torch.no_grad()
def build_block_partitions(vector: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    vector = vector.reshape(-1)
    original_dim = vector.numel()
    pad = (block_size - original_dim % block_size) % block_size
    if pad > 0:
        vector = F.pad(vector, (0, pad))
    return vector.view(-1, block_size), original_dim


@torch.no_grad()
def expand_block_values(block_values: torch.Tensor, vector_dim: int, block_size: int) -> torch.Tensor:
    return block_values.repeat_interleave(block_size)[:vector_dim]


@torch.no_grad()
def build_block_gsn_features(
    corrected_update: torch.Tensor,
    ema_state: Optional[torch.Tensor],
    global_reference: Optional[torch.Tensor],
    block_size: int,
    ema_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks, _ = build_block_partitions(corrected_update.detach(), block_size)
    mean_abs = blocks.abs().mean(dim=1)
    if ema_state is None or ema_state.numel() != mean_abs.numel():
        updated_ema = mean_abs.clone()
    else:
        updated_ema = ema_decay * ema_state.to(mean_abs.device) + (1.0 - ema_decay) * mean_abs
    if global_reference is None:
        cosine = torch.zeros_like(mean_abs)
    else:
        reference_blocks, _ = build_block_partitions(global_reference.detach().to(corrected_update.device), block_size)
        numerator = (blocks * reference_blocks).sum(dim=1)
        denominator = blocks.norm(dim=1) * reference_blocks.norm(dim=1) + 1e-12
        cosine = (numerator / denominator).clamp(-1.0, 1.0)
    features = torch.stack([mean_abs, updated_ema, cosine], dim=1)
    return features, updated_ema


@torch.no_grad()
def recalibrate_probabilities(probabilities: torch.Tensor, target_ratio: float) -> torch.Tensor:
    mean_prob = probabilities.mean().clamp_min(1e-8)
    scaled = probabilities * (target_ratio / mean_prob)
    return scaled.clamp(0.0, 1.0)


@torch.no_grad()
def sample_hard_block_mask(block_probabilities: torch.Tensor) -> torch.Tensor:
    return torch.bernoulli(block_probabilities).to(block_probabilities.dtype)


def straight_through_block_mask(block_probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hard_mask = sample_hard_block_mask(block_probabilities)
    ste_mask = hard_mask.detach() - block_probabilities.detach() + block_probabilities
    return hard_mask, ste_mask


@torch.no_grad()
def sparsify_with_residual(
    dense_update: torch.Tensor,
    hard_mask: torch.Tensor,
    residual: Optional[torch.Tensor],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    corrected_update = dense_update if residual is None else dense_update + residual
    expanded_mask = expand_block_values(hard_mask, corrected_update.numel(), block_size)
    sparse_update = corrected_update * expanded_mask
    updated_residual = corrected_update - sparse_update
    return corrected_update, sparse_update, updated_residual


def build_ste_sparse_update(
    dense_update: torch.Tensor,
    ste_mask: torch.Tensor,
    residual: Optional[torch.Tensor],
    block_size: int,
) -> torch.Tensor:
    corrected_update = dense_update if residual is None else dense_update + residual
    expanded_mask = expand_block_values(ste_mask, corrected_update.numel(), block_size)
    return corrected_update * expanded_mask
