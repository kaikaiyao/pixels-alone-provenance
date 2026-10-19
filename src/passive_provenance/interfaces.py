from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from passive_provenance.matrix import InterfaceSpec, target_weights


@dataclass(frozen=True)
class QueryReply:
    payload: list[float] | int | float | str
    target_score: float
    abstained: bool
    budget_exhausted: bool


def score_from_probabilities(probabilities: torch.Tensor, class_names: tuple[str, ...] | list[str], target: str) -> torch.Tensor:
    weights = target_weights(class_names, target, uniform_ai=(target == "coarse_ai"))
    device = probabilities.device
    weight_vector = torch.tensor([float(weights.get(name, 0.0)) for name in class_names], device=device, dtype=probabilities.dtype)
    score = probabilities @ weight_vector
    if target in {"fake", "ai"}:
        return score.clamp(0.0, 1.0)
    return score


def calibrate_binary_threshold(probabilities: np.ndarray, labels: np.ndarray, class_names: tuple[str, ...] | list[str], target: str) -> float:
    if target in class_names:
        target_index = tuple(class_names).index(target)
        target_scores = probabilities[:, target_index]
    else:
        target_scores = score_from_probabilities(torch.from_numpy(probabilities), class_names, target).numpy()
    positives = labels.astype(np.int32)
    best_threshold = 0.5
    best_margin = -float("inf")
    for threshold in np.unique(target_scores):
        predictions = (target_scores >= threshold).astype(np.int32)
        tpr = float(((predictions == 1) & (positives == 1)).sum()) / max(1, int((positives == 1).sum()))
        fpr = float(((predictions == 1) & (positives == 0)).sum()) / max(1, int((positives == 0).sum()))
        margin = tpr - fpr
        if margin > best_margin:
            best_margin = margin
            best_threshold = float(threshold)
    return best_threshold


def calibrate_abstention_margin(scores: np.ndarray, threshold: float, fraction: float = 0.1) -> float:
    if scores.size == 0:
        return 0.0
    distances = np.abs(scores - threshold)
    index = min(int(np.ceil(float(fraction) * scores.size)) - 1, scores.size - 1)
    ordered = np.sort(distances)
    return float(max(0.0, ordered[max(index, 0)]))


class InterfaceWrapper:
    def __init__(
        self,
        spec: InterfaceSpec,
        class_names: tuple[str, ...],
        *,
        target_label: str,
        binary_threshold: float = 0.5,
        abstention_margin: float = 0.0,
    ) -> None:
        self.spec = spec
        self.class_names = tuple(class_names)
        self.target_label = target_label
        self.binary_threshold = float(binary_threshold)
        self.abstention_margin = float(abstention_margin)
        self.query_count = 0

    def reset(self) -> None:
        self.query_count = 0

    def query(self, probabilities: torch.Tensor) -> QueryReply:
        self.query_count += 1
        budget_exhausted = self.spec.rate_limit is not None and self.query_count > self.spec.rate_limit
        score = float(score_from_probabilities(probabilities.unsqueeze(0), self.class_names, self.target_label)[0].item())
        if budget_exhausted:
            return QueryReply(payload="budget_exhausted", target_score=score, abstained=False, budget_exhausted=True)

        if self.spec.kind == "full_probability_vector":
            return QueryReply(payload=[float(item) for item in probabilities.tolist()], target_score=score, abstained=False, budget_exhausted=False)
        if self.spec.kind == "single_target_score":
            return QueryReply(payload=score, target_score=score, abstained=False, budget_exhausted=False)
        if self.spec.kind == "quantized_target_score":
            levels = 2 ** int(self.spec.quantization_bits or 8) - 1
            quantized = round(score * levels) / max(1, levels)
            return QueryReply(payload=float(quantized), target_score=float(quantized), abstained=False, budget_exhausted=False)
        if self.spec.kind == "top1_label":
            payload = int(torch.argmax(probabilities).item())
            return QueryReply(payload=payload, target_score=score, abstained=False, budget_exhausted=False)
        if self.spec.kind == "binary_accept_reject":
            abstained = self.spec.abstention and abs(score - self.binary_threshold) <= self.abstention_margin
            if abstained:
                return QueryReply(payload="abstain", target_score=score, abstained=True, budget_exhausted=False)
            payload = int(score >= self.binary_threshold)
            return QueryReply(payload=payload, target_score=score, abstained=False, budget_exhausted=False)
        raise ValueError(f"Unsupported interface kind {self.spec.kind}.")


def build_interface_wrapper(
    spec: InterfaceSpec,
    class_names: tuple[str, ...] | list[str],
    *,
    target_label: str,
    binary_threshold: float = 0.5,
    abstention_margin: float = 0.0,
) -> InterfaceWrapper:
    return InterfaceWrapper(
        spec,
        tuple(class_names),
        target_label=target_label,
        binary_threshold=binary_threshold,
        abstention_margin=abstention_margin,
    )


def payload_dimension(spec: InterfaceSpec, class_count: int) -> int:
    if spec.kind == "full_probability_vector":
        return class_count
    return 1
