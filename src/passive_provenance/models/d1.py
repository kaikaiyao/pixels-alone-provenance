from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from passive_provenance.constants import CLASS_NAMES
from passive_provenance.utils.image_ops import differentiable_clip_preprocess


def load_open_clip_encoder(model_name: str, pretrained: str):
    import open_clip

    return open_clip.create_model(
        model_name,
        pretrained=pretrained,
        cache_dir=os.environ.get("PP_OPEN_CLIP_CACHE"),
    )


class LinearSoftmaxHead(nn.Module):
    def __init__(self, feature_dim: int, class_names: Sequence[str] = CLASS_NAMES) -> None:
        super().__init__()
        self.class_names = tuple(class_names)
        self.linear = nn.Linear(feature_dim, len(self.class_names))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)

    def probabilities(self, features: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(features), dim=1)

    @staticmethod
    def fake_probability(probabilities: torch.Tensor) -> torch.Tensor:
        return probabilities[:, 1] + probabilities[:, 2]

    @staticmethod
    def mixture_probability(probabilities: torch.Tensor, pi: float) -> torch.Tensor:
        return pi * probabilities[:, 1] + (1.0 - pi) * probabilities[:, 2]


class FrozenClipVerifier(nn.Module):
    def __init__(
        self,
        clip_model_name: str,
        clip_pretrained: str,
        input_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        class_names: Sequence[str] = CLASS_NAMES,
    ) -> None:
        super().__init__()
        self.class_names = tuple(class_names)
        self.input_size = input_size
        self.mean = tuple(float(item) for item in mean)
        self.std = tuple(float(item) for item in std)
        self.encoder = load_open_clip_encoder(clip_model_name, clip_pretrained)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        feature_dim = getattr(self.encoder.visual, "output_dim", None)
        if feature_dim is None:
            raise ValueError("Unable to infer CLIP feature dimension.")
        self.head = LinearSoftmaxHead(feature_dim=feature_dim, class_names=self.class_names)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return differentiable_clip_preprocess(images, self.input_size, self.mean, self.std)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        processed = self.preprocess(images)
        features = self.encoder.encode_image(processed)
        return F.normalize(features, dim=-1)

    def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(images))

    def forward_probs(self, images: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward_logits(images), dim=1)


@dataclass
class VerifierCheckpoint:
    clip_model_name: str
    clip_pretrained: str
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    class_names: tuple[str, ...]
    head_state_dict: dict[str, torch.Tensor]


def save_verifier_checkpoint(model: FrozenClipVerifier, path: str) -> None:
    payload = {
        "clip_model_name": model.encoder.visual.image_size if hasattr(model.encoder.visual, "image_size") else None,
        "clip_model_name_text": getattr(model.encoder, "model_name", None),
    }
    torch.save(
        {
            "clip_model_name": getattr(model.encoder, "model_name", None) or "ViT-B-32",
            "clip_pretrained": "openai",
            "input_size": model.input_size,
            "mean": model.mean,
            "std": model.std,
            "class_names": model.class_names,
            "head_state_dict": model.head.state_dict(),
            "meta": payload,
        },
        path,
    )


def load_verifier_checkpoint(path: str, clip_model_name: str | None = None, clip_pretrained: str | None = None) -> FrozenClipVerifier:
    payload = torch.load(path, map_location="cpu")
    model = FrozenClipVerifier(
        clip_model_name=clip_model_name or payload["clip_model_name"],
        clip_pretrained=clip_pretrained or payload["clip_pretrained"],
        input_size=int(payload["input_size"]),
        mean=payload["mean"],
        std=payload["std"],
        class_names=payload["class_names"],
    )
    model.head.load_state_dict(payload["head_state_dict"])
    return model
