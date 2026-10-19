from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from sklearn import metrics as sk_metrics

from passive_provenance.utils.image_ops import linf_distance, psnr_from_tensors


def compute_multiclass_metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    accuracy = float(sk_metrics.accuracy_score(labels, predictions))
    macro_f1 = float(sk_metrics.f1_score(labels, predictions, average="macro"))
    confusion = sk_metrics.confusion_matrix(labels, predictions).tolist()
    fake_labels = (labels != 0).astype(np.int32)
    fake_scores = probabilities[:, 1:].sum(axis=1)
    fake_auc = float(sk_metrics.roc_auc_score(fake_labels, fake_scores))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "fake_auc": fake_auc,
        "confusion_matrix": confusion,
    }


def compute_lpips_batch(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    *,
    batch_size: int = 8,
    device: torch.device | str | None = None,
    lpips_factory: Callable[[], torch.nn.Module] | None = None,
) -> torch.Tensor:
    if lpips_factory is None:
        try:
            import lpips  # type: ignore
        except Exception:
            return torch.full((original.shape[0],), float("nan"), device=original.device)
        lpips_factory = lambda: lpips.LPIPS(net="alex")
    metric_device = torch.device(device) if device is not None else original.device
    model = lpips_factory().to(metric_device)
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, original.shape[0], batch_size):
            original_batch = original[start : start + batch_size].to(metric_device)
            adversarial_batch = adversarial[start : start + batch_size].to(metric_device)
            chunk = model(original_batch * 2.0 - 1.0, adversarial_batch * 2.0 - 1.0).flatten()
            chunks.append(chunk.detach().to(original.device))
    return torch.cat(chunks, dim=0)


def summarize_attack_distances(original: torch.Tensor, adversarial: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "linf": linf_distance(original, adversarial),
        "psnr": psnr_from_tensors(original, adversarial),
    }


def robust_gap(clean_target_acceptance: float, attacked_target_scores: np.ndarray) -> float:
    return float(clean_target_acceptance - attacked_target_scores.mean())
