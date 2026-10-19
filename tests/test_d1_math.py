from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from passive_provenance.models.d1 import LinearSoftmaxHead, load_open_clip_encoder
from passive_provenance.pipelines.exact_recovery import recover_softmax_relative_head, score_error


def test_fake_probability_is_sum_of_model_scores():
    head = LinearSoftmaxHead(feature_dim=4)
    features = torch.randn(3, 4)
    probabilities = head.probabilities(features)
    assert torch.allclose(LinearSoftmaxHead.fake_probability(probabilities), probabilities[:, 1] + probabilities[:, 2])


def test_mixture_probability_matches_interpolation():
    probabilities = torch.tensor([[0.1, 0.3, 0.6]])
    mixture = LinearSoftmaxHead.mixture_probability(probabilities, 0.25)
    assert torch.allclose(mixture, torch.tensor([0.25 * 0.3 + 0.75 * 0.6]))


def test_exact_recovery_recovers_synthetic_softmax_head():
    torch.manual_seed(7)
    features = torch.randn(40, 5)
    true_weights = torch.randn(3, 5)
    true_bias = torch.randn(3)
    logits = features @ true_weights.T + true_bias
    probabilities = torch.softmax(logits, dim=1)
    recovered = recover_softmax_relative_head(features, probabilities)
    recovered_probs = recovered.probabilities(features)
    assert score_error(probabilities, recovered_probs) < 1e-5


def test_open_clip_encoder_uses_configured_cache(monkeypatch, tmp_path):
    captured = {}
    sentinel = object()

    def create_model(model_name, *, pretrained, cache_dir):
        captured.update(
            {"model_name": model_name, "pretrained": pretrained, "cache_dir": cache_dir}
        )
        return sentinel

    monkeypatch.setitem(sys.modules, "open_clip", SimpleNamespace(create_model=create_model))
    cache_dir = str(tmp_path / "open-clip")
    monkeypatch.setenv("PP_OPEN_CLIP_CACHE", cache_dir)
    result = load_open_clip_encoder("ViT-B-32", "openai")

    assert result is sentinel
    assert captured == {
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "cache_dir": cache_dir,
    }
