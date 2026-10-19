from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from passive_provenance.matrix import target_weights
from passive_provenance.models.d1 import LinearSoftmaxHead


def project_linf(original: torch.Tensor, adversarial: torch.Tensor, eps: float) -> torch.Tensor:
    delta = (adversarial - original).clamp(min=-eps, max=eps)
    return (original + delta).clamp(0.0, 1.0)


def score_from_probs(probabilities: torch.Tensor, target: str, pi: float | None = None) -> torch.Tensor:
    if target == "real":
        return probabilities[:, 0]
    if target == "sd15":
        return probabilities[:, 1]
    if target == "sdxl":
        return probabilities[:, 2]
    if target == "fake":
        return LinearSoftmaxHead.fake_probability(probabilities)
    if target == "mixture":
        if pi is None:
            raise ValueError("pi is required for mixture targets.")
        return LinearSoftmaxHead.mixture_probability(probabilities, pi)
    raise ValueError(f"Unsupported target {target}.")


def targeted_loss(probabilities: torch.Tensor, target: str, pi: float | None = None) -> torch.Tensor:
    scores = score_from_probs(probabilities, target=target, pi=pi).clamp_min(1e-8)
    return -torch.log(scores)


def generic_score_from_probs(probabilities: torch.Tensor, class_names: tuple[str, ...] | list[str], target: str) -> torch.Tensor:
    weights = target_weights(class_names, target, uniform_ai=(target == "coarse_ai"))
    vector = torch.tensor([float(weights.get(name, 0.0)) for name in class_names], device=probabilities.device, dtype=probabilities.dtype)
    return probabilities @ vector


def generic_targeted_loss(probabilities: torch.Tensor, class_names: tuple[str, ...] | list[str], target: str) -> torch.Tensor:
    return -torch.log(generic_score_from_probs(probabilities, class_names, target).clamp_min(1e-8))


@dataclass
class PGDResult:
    adversarial: torch.Tensor
    scores: torch.Tensor


def pgd_targeted(
    model_forward: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    restarts: int,
    target: str,
    pi: float | None = None,
) -> PGDResult:
    device = images.device
    best_scores = torch.full((images.shape[0],), -float("inf"), device=device)
    best_adv = images.clone()

    for restart_idx in range(restarts):
        if restart_idx == 0:
            current = images.clone()
        else:
            current = images + torch.empty_like(images).uniform_(-eps, eps)
            current = project_linf(images, current, eps)
        current = current.detach()

        for _ in range(steps):
            current.requires_grad_(True)
            probabilities = model_forward(current)
            losses = targeted_loss(probabilities, target=target, pi=pi)
            gradient = torch.autograd.grad(losses.sum(), current)[0]
            current = current.detach() - step_size * gradient.sign()
            current = project_linf(images, current, eps)

        with torch.no_grad():
            final_scores = score_from_probs(model_forward(current), target=target, pi=pi)
            improved = final_scores > best_scores
            best_scores = torch.where(improved, final_scores, best_scores)
            mask = improved.view(-1, 1, 1, 1)
            best_adv = torch.where(mask, current, best_adv)

    return PGDResult(adversarial=best_adv.detach(), scores=best_scores.detach())


def pgd_targeted_generic(
    model_forward: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    *,
    class_names: tuple[str, ...] | list[str],
    eps: float,
    steps: int,
    step_size: float,
    restarts: int,
    target: str,
) -> PGDResult:
    device = images.device
    best_scores = torch.full((images.shape[0],), -float("inf"), device=device)
    best_adv = images.clone()

    for restart_idx in range(restarts):
        if restart_idx == 0:
            current = images.clone()
        else:
            current = images + torch.empty_like(images).uniform_(-eps, eps)
            current = project_linf(images, current, eps)
        current = current.detach()

        for _ in range(steps):
            current.requires_grad_(True)
            probabilities = model_forward(current)
            losses = generic_targeted_loss(probabilities, class_names=tuple(class_names), target=target)
            gradient = torch.autograd.grad(losses.sum(), current)[0]
            current = current.detach() - step_size * gradient.sign()
            current = project_linf(images, current, eps)

        with torch.no_grad():
            final_scores = generic_score_from_probs(model_forward(current), tuple(class_names), target)
            improved = final_scores > best_scores
            best_scores = torch.where(improved, final_scores, best_scores)
            mask = improved.view(-1, 1, 1, 1)
            best_adv = torch.where(mask, current, best_adv)

    return PGDResult(adversarial=best_adv.detach(), scores=best_scores.detach())


def pgd_maximize_cross_entropy(
    model_forward_logits: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    eps: float,
    steps: int,
    step_size: float,
    restarts: int,
) -> PGDResult:
    device = images.device
    best_losses = torch.full((images.shape[0],), -float("inf"), device=device)
    best_adv = images.clone()

    for restart_idx in range(restarts):
        if restart_idx == 0:
            current = images.clone()
        else:
            current = images + torch.empty_like(images).uniform_(-eps, eps)
            current = project_linf(images, current, eps)
        current = current.detach()

        for _ in range(steps):
            current.requires_grad_(True)
            logits = model_forward_logits(current)
            losses = F.cross_entropy(logits, labels, reduction="none")
            gradient = torch.autograd.grad(losses.sum(), current)[0]
            current = current.detach() + step_size * gradient.sign()
            current = project_linf(images, current, eps)

        with torch.no_grad():
            final_losses = F.cross_entropy(model_forward_logits(current), labels, reduction="none")
            improved = final_losses > best_losses
            best_losses = torch.where(improved, final_losses, best_losses)
            mask = improved.view(-1, 1, 1, 1)
            best_adv = torch.where(mask, current, best_adv)

    return PGDResult(adversarial=best_adv.detach(), scores=best_losses.detach())
