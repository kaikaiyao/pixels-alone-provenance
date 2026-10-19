from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RecoveredHead:
    weights: torch.Tensor
    bias: torch.Tensor
    anchor_index: int

    def to(self, device: torch.device | str) -> "RecoveredHead":
        return RecoveredHead(
            weights=self.weights.to(device),
            bias=self.bias.to(device),
            anchor_index=self.anchor_index,
        )

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.weights.T + self.bias

    def probabilities(self, features: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits(features), dim=1)


def recover_softmax_relative_head(
    features: torch.Tensor,
    probabilities: torch.Tensor,
    anchor_index: int = 0,
) -> RecoveredHead:
    if features.ndim != 2:
        raise ValueError("features must be 2D.")
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be 2D.")
    if anchor_index != 0:
        raise NotImplementedError("Only anchor_index=0 is currently supported.")
    probs = probabilities.clamp_min(1e-8)
    design = torch.cat([features, torch.ones(features.shape[0], 1, device=features.device)], dim=1)
    recovered_rows = [torch.zeros(design.shape[1], device=features.device)]
    anchor_probs = probs[:, anchor_index]
    for class_index in range(probabilities.shape[1]):
        if class_index == anchor_index:
            continue
        response = torch.log(probs[:, class_index] / anchor_probs)
        solution = torch.linalg.lstsq(design, response.unsqueeze(1)).solution.squeeze(1)
        recovered_rows.append(solution)
    stacked = torch.stack(recovered_rows, dim=0)
    return RecoveredHead(weights=stacked[:, :-1], bias=stacked[:, -1], anchor_index=anchor_index)


def relative_parameter_error(true_weights: torch.Tensor, true_bias: torch.Tensor, recovered: RecoveredHead) -> dict[str, float]:
    target_weights = true_weights - true_weights[0:1]
    target_bias = true_bias - true_bias[0]
    weight_error = torch.norm(recovered.weights - target_weights) / torch.norm(target_weights).clamp_min(1e-12)
    bias_error = torch.norm(recovered.bias - target_bias) / torch.norm(target_bias).clamp_min(1e-12)
    return {
        "weight_error": float(weight_error.item()),
        "bias_error": float(bias_error.item()),
    }


def score_error(probabilities: torch.Tensor, recovered_probabilities: torch.Tensor) -> float:
    return float((probabilities - recovered_probabilities).abs().mean().item())
