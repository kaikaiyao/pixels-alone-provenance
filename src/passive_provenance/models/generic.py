from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from passive_provenance.models.d1 import load_open_clip_encoder
from passive_provenance.utils.image_ops import differentiable_clip_preprocess


class FrozenClipBackbone(nn.Module):
    def __init__(self, clip_model_name: str, clip_pretrained: str, input_size: int, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.input_size = int(input_size)
        self.mean = tuple(float(item) for item in mean)
        self.std = tuple(float(item) for item in std)
        self.encoder = load_open_clip_encoder(clip_model_name, clip_pretrained)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        feature_dim = getattr(self.encoder.visual, "output_dim", None)
        if feature_dim is None:
            raise ValueError("Unable to infer CLIP feature dimension.")
        self.feature_dim = int(feature_dim)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return differentiable_clip_preprocess(images, self.input_size, self.mean, self.std)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        processed = self.preprocess(images)
        features = self.encoder.encode_image(processed)
        return F.normalize(features, dim=-1)


class VerifierHead(nn.Module):
    head_type: str

    def probabilities(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def effective_linear_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class SoftmaxLinearHead(VerifierHead):
    head_type = "softmax_linear"

    def __init__(self, feature_dim: int, num_classes: int, *, use_bias: bool = True, temperature: float = 1.0, augment_bias: bool = False) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.temperature = float(temperature)
        self.augment_bias = bool(augment_bias)
        if self.augment_bias:
            self.linear = nn.Linear(feature_dim + 1, num_classes, bias=False)
        else:
            self.linear = nn.Linear(feature_dim, num_classes, bias=use_bias)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        if self.augment_bias:
            ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
            features = torch.cat([features, ones], dim=1)
        return self.linear(features) / self.temperature

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits(features)

    def probabilities(self, features: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits(features), dim=1)

    def effective_linear_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.linear.weight
        if self.augment_bias:
            scaled_weight = weight[:, :-1] / self.temperature
            scaled_bias = weight[:, -1] / self.temperature
            return scaled_weight.detach(), scaled_bias.detach()
        bias = self.linear.bias
        if bias is None:
            bias = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
        return (weight / self.temperature).detach(), (bias / self.temperature).detach()


class BinaryLogisticHead(VerifierHead):
    head_type = "binary_logistic"

    def __init__(self, feature_dim: int, *, use_bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1, bias=use_bias)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits(features)

    def probabilities(self, features: torch.Tensor) -> torch.Tensor:
        fake_prob = torch.sigmoid(self.logits(features))
        return torch.stack([1.0 - fake_prob, fake_prob], dim=1)

    def effective_linear_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.linear.weight.detach()
        bias = self.linear.bias.detach() if self.linear.bias is not None else torch.zeros(1, device=weight.device, dtype=weight.dtype)
        effective_weight = torch.cat([torch.zeros_like(weight), weight], dim=0)
        effective_bias = torch.cat([torch.zeros_like(bias), bias], dim=0)
        return effective_weight, effective_bias


class OneVsRestLogisticHead(VerifierHead):
    head_type = "ovr_logistic"

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits(features)

    def probabilities(self, features: torch.Tensor) -> torch.Tensor:
        raw = torch.sigmoid(self.logits(features))
        return raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def effective_linear_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("OVR logistic does not expose a single softmax-equivalent linear system.")


class GeluMLPHead(VerifierHead):
    head_type = "mlp_gelu"

    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits(features)

    def probabilities(self, features: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits(features), dim=1)

    def effective_linear_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("MLP heads are not exactly recoverable with the linear solver.")


def build_head(head_type: str, feature_dim: int, num_classes: int, hidden_dim: int = 256) -> VerifierHead:
    if head_type == "softmax_linear":
        return SoftmaxLinearHead(feature_dim, num_classes)
    if head_type == "temperature_scaled_softmax":
        return SoftmaxLinearHead(feature_dim, num_classes, temperature=0.75)
    if head_type == "intercept_augmented_softmax":
        return SoftmaxLinearHead(feature_dim, num_classes, use_bias=False, augment_bias=True)
    if head_type == "binary_logistic":
        return BinaryLogisticHead(feature_dim)
    if head_type == "ovr_logistic":
        return OneVsRestLogisticHead(feature_dim, num_classes)
    if head_type == "mlp_gelu":
        return GeluMLPHead(feature_dim, num_classes, hidden_dim=hidden_dim)
    raise ValueError(f"Unsupported head type {head_type}.")


class ClipFeatureVerifier(nn.Module):
    def __init__(
        self,
        *,
        verifier_id: str,
        class_names: Sequence[str],
        clip_model_name: str,
        clip_pretrained: str,
        input_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        head_type: str,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.verifier_id = verifier_id
        self.class_names = tuple(class_names)
        self.backbone = FrozenClipBackbone(clip_model_name, clip_pretrained, input_size, mean, std)
        self.head = build_head(head_type, self.backbone.feature_dim, len(self.class_names), hidden_dim=hidden_dim)
        self.head_type = head_type

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(images)

    def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.head.forward(self.encode(images))

    def forward_probs(self, images: torch.Tensor) -> torch.Tensor:
        return self.head.probabilities(self.encode(images))

    def head_probabilities(self, features: torch.Tensor) -> torch.Tensor:
        return self.head.probabilities(features)

    def effective_linear_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head.effective_linear_parameters()


@dataclass
class GenericVerifierCheckpoint:
    verifier_id: str
    benchmark_id: str
    family: str
    head_type: str
    class_names: tuple[str, ...]
    clip_model_name: str
    clip_pretrained: str
    input_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    hidden_dim: int
    head_state_dict: dict[str, torch.Tensor]


def save_generic_verifier_checkpoint(model: ClipFeatureVerifier, path: str, *, benchmark_id: str, family: str, hidden_dim: int = 256) -> None:
    torch.save(
        {
            "verifier_id": model.verifier_id,
            "benchmark_id": benchmark_id,
            "family": family,
            "head_type": model.head_type,
            "class_names": model.class_names,
            "clip_model_name": model.backbone.clip_model_name,
            "clip_pretrained": model.backbone.clip_pretrained,
            "input_size": model.backbone.input_size,
            "mean": model.backbone.mean,
            "std": model.backbone.std,
            "hidden_dim": hidden_dim,
            "head_state_dict": model.head.state_dict(),
        },
        path,
    )


def load_generic_verifier_checkpoint(path: str) -> ClipFeatureVerifier:
    payload = torch.load(path, map_location="cpu")
    model = ClipFeatureVerifier(
        verifier_id=payload["verifier_id"],
        class_names=payload["class_names"],
        clip_model_name=payload["clip_model_name"],
        clip_pretrained=payload["clip_pretrained"],
        input_size=int(payload["input_size"]),
        mean=payload["mean"],
        std=payload["std"],
        head_type=payload["head_type"],
        hidden_dim=int(payload.get("hidden_dim", 256)),
    )
    model.head.load_state_dict(payload["head_state_dict"])
    return model
