from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEmulator(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int, hidden_dim: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def probabilities(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.forward(features)
        if self.mode == "vector":
            return F.softmax(logits, dim=1)
        if self.mode == "binary":
            fake_prob = torch.sigmoid(logits.squeeze(1))
            return torch.stack([1.0 - fake_prob, fake_prob], dim=1)
        raise ValueError(f"Unsupported probability mode {self.mode}.")


@dataclass(frozen=True)
class EmulatorFitMetrics:
    emulator_eps95: float
    emulator_max_error: float
    train_loss: float
    val_loss: float


def _loss_for_mode(prediction: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "vector":
        return F.mse_loss(F.softmax(prediction, dim=1), target)
    if mode == "scalar":
        return F.mse_loss(prediction.squeeze(1), target.squeeze(1))
    if mode == "binary":
        return F.binary_cross_entropy_with_logits(prediction.squeeze(1), target.squeeze(1))
    if mode == "label":
        return F.cross_entropy(prediction, target.squeeze(1).long())
    if mode == "ternary":
        return F.cross_entropy(prediction, target.squeeze(1).long())
    raise ValueError(f"Unsupported emulator mode {mode}.")


def fit_emulator(
    *,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    mode: str,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
) -> tuple[FeatureEmulator, EmulatorFitMetrics]:
    output_dim = train_targets.shape[1] if train_targets.ndim == 2 else 1
    if mode in {"label", "ternary"}:
        output_dim = int(train_targets.max().item()) + 1
    model = FeatureEmulator(train_features.shape[1], output_dim, hidden_dim, "vector" if output_dim > 1 and mode == "vector" else "binary")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    last_train = float("inf")
    for _ in range(epochs):
        prediction = model(train_features)
        loss = _loss_for_mode(prediction, train_targets, mode)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_train = float(loss.item())
        with torch.no_grad():
            val_prediction = model(val_features)
            val_loss = float(_loss_for_mode(val_prediction, val_targets, mode).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Emulator fitting did not produce a checkpoint.")
    model.load_state_dict(best_state)
    with torch.no_grad():
        val_prediction = model(val_features)
        if mode == "vector":
            errors = (F.softmax(val_prediction, dim=1) - val_targets).abs().max(dim=1).values
        elif mode == "scalar":
            errors = (val_prediction.squeeze(1) - val_targets.squeeze(1)).abs()
        elif mode == "binary":
            errors = (torch.sigmoid(val_prediction.squeeze(1)) - val_targets.squeeze(1)).abs()
        else:
            hard = val_prediction.argmax(dim=1)
            errors = (hard != val_targets.squeeze(1).long()).float()
    metrics = EmulatorFitMetrics(
        emulator_eps95=float(torch.quantile(errors, 0.95).item()) if errors.numel() else 0.0,
        emulator_max_error=float(errors.max().item()) if errors.numel() else 0.0,
        train_loss=last_train,
        val_loss=best_val,
    )
    return model, metrics
