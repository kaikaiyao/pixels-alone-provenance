from __future__ import annotations

import torch
import torch.nn.functional as F

from passive_provenance.attacks.pgd import pgd_maximize_cross_entropy, pgd_targeted, project_linf, targeted_loss
from passive_provenance.stages.attack_blackbox import ExternalDetectorAttackVerifier, score_from_emulator_logits


def test_project_linf_respects_epsilon():
    original = torch.zeros(2, 3, 4, 4)
    adversarial = torch.full_like(original, 0.2)
    projected = project_linf(original, adversarial, 0.1)
    assert torch.all(projected <= 0.1 + 1e-6)


def test_targeted_loss_prefers_higher_target_score():
    low = torch.tensor([[0.9, 0.05, 0.05]])
    high = torch.tensor([[0.1, 0.8, 0.1]])
    assert targeted_loss(high, "sd15").item() < targeted_loss(low, "sd15").item()


def test_pgd_restarts_keep_best_candidate():
    class SimpleModel:
        def forward_probs(self, images: torch.Tensor) -> torch.Tensor:
            score = images.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
            return torch.stack([1.0 - score, score * 0.5, score * 0.5], dim=1)

    images = torch.zeros(2, 3, 8, 8)
    result = pgd_targeted(SimpleModel().forward_probs, images, eps=0.1, steps=5, step_size=0.025, restarts=2, target="fake")
    assert result.adversarial.shape == images.shape
    assert torch.all(result.scores >= 0.0)


def test_pgd_maximize_cross_entropy_increases_loss():
    class SimpleLogitModel:
        def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
            score = images.mean(dim=(1, 2, 3))
            return torch.stack([1.0 - score, score], dim=1)

    images = torch.full((2, 3, 8, 8), 0.1)
    labels = torch.ones(2, dtype=torch.long)
    base_loss = F.cross_entropy(SimpleLogitModel().forward_logits(images), labels, reduction="none")
    result = pgd_maximize_cross_entropy(
        SimpleLogitModel().forward_logits,
        images,
        labels,
        eps=0.1,
        steps=5,
        step_size=0.025,
        restarts=2,
    )
    adv_loss = F.cross_entropy(SimpleLogitModel().forward_logits(result.adversarial), labels, reduction="none")
    assert result.adversarial.shape == images.shape
    assert torch.all(adv_loss >= base_loss)


def test_blackbox_score_mapping_respects_label_and_ternary_semantics():
    class_names = ("real", "sd15", "sdxl", "sd21", "pixart")

    label_logits = torch.log(
        torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.15, 0.25],
            ],
            dtype=torch.float32,
        )
    )
    fake_score = score_from_emulator_logits(label_logits, mode="label", class_names=class_names, target_label="fake")
    assert torch.allclose(fake_score, torch.tensor([0.9]))

    ternary_logits = torch.log(torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float32))
    fake_accept = score_from_emulator_logits(ternary_logits, mode="ternary", class_names=class_names, target_label="fake")
    real_accept = score_from_emulator_logits(ternary_logits, mode="ternary", class_names=class_names, target_label="real")
    assert torch.allclose(fake_accept, torch.tensor([0.5]))
    assert torch.allclose(real_accept, torch.tensor([0.2]))


def test_external_detector_attack_verifier_uses_public_backbone_for_surrogate_and_detector_for_eval():
    class DummyBackbone:
        def encode(self, images: torch.Tensor) -> torch.Tensor:
            return images.mean(dim=(2, 3))

    class DummyDetector:
        class_names = ("real", "fake")

        def predict_tensor_probabilities(self, images: torch.Tensor) -> torch.Tensor:
            fake_score = images.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
            return torch.stack([1.0 - fake_score, fake_score], dim=1).cpu()

    verifier = ExternalDetectorAttackVerifier(DummyBackbone(), DummyDetector())
    images = torch.full((2, 3, 4, 4), 0.25)
    features = verifier.encode(images)
    probabilities = verifier.forward_probs(images)

    assert features.shape == (2, 3)
    assert probabilities.shape == (2, 2)
    assert torch.allclose(probabilities[:, 1], torch.full((2,), 0.25))
