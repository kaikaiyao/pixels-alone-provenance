from __future__ import annotations

import pytest
import torch

from passive_provenance.stages.perceptual_audit import (
    audit_outcome,
    select_visual_examples,
    source_initially_correct,
    summarize_rows,
)
from passive_provenance.utils.perceptual import ssim_per_image


def test_ssim_identity_is_one() -> None:
    images = torch.rand(2, 3, 32, 32)
    scores = ssim_per_image(images, images)
    assert torch.allclose(scores, torch.ones_like(scores), atol=1e-6)


def test_ssim_decreases_for_perturbed_image() -> None:
    images = torch.full((1, 3, 32, 32), 0.5)
    perturbed = images.clone()
    perturbed[:, :, 8:24, 8:24] = 0.75
    score = ssim_per_image(images, perturbed).item()
    assert 0.0 < score < 1.0


def _metric_row(success: int, value: float, *, initially_correct: int = 1) -> dict[str, object]:
    return {
        "success": success,
        "initially_correct": initially_correct,
        "success_on_initially_correct": int(success and initially_correct),
        "linf": value,
        "psnr": 40.0 + value,
        "ssim": 0.99 - value,
        "lpips_alex": value,
    }


def test_visual_examples_include_success_and_failure_when_available() -> None:
    rows = [_metric_row(1, 0.01), _metric_row(1, 0.02), _metric_row(0, 0.03)]
    selected = select_visual_examples(rows, 2)
    assert [int(row["success"]) for row in selected] == [1, 0]


def test_summary_separates_success_and_failure_metrics() -> None:
    rows = [
        _metric_row(1, 0.01),
        _metric_row(0, 0.03),
        _metric_row(1, 0.02, initially_correct=0),
    ]
    summary = summarize_rows(rows)
    assert summary["n"] == 3
    assert summary["initially_correct_n"] == 2
    assert summary["asr"] == pytest.approx(2 / 3)
    assert summary["asr_initially_correct"] == pytest.approx(0.5)
    assert summary["induced_success_n"] == 1
    assert summary["attack_failure_n"] == 1
    assert summary["preexisting_target_n"] == 1
    assert summary["induced_success"]["n"] == 1
    assert summary["attack_failure"]["n"] == 1


def test_source_correctness_uses_binary_and_directional_multiclass_rules() -> None:
    probabilities = torch.tensor(
        [
            [0.70, 0.20, 0.10],
            [0.10, 0.75, 0.15],
            [0.10, 0.20, 0.70],
        ]
    )
    assert source_initially_correct("human_to_fake", probabilities, 0.5).tolist() == [1, 0, 0]
    assert source_initially_correct("ai_to_real", probabilities, 0.5).tolist() == [0, 1, 1]
    assert source_initially_correct("sd15_to_sdxl", probabilities, 0.5).tolist() == [0, 1, 0]
    assert source_initially_correct("sdxl_to_sd15", probabilities, 0.5).tolist() == [0, 0, 1]


def test_audit_outcome_does_not_call_preexisting_target_an_induced_success() -> None:
    assert audit_outcome(_metric_row(1, 0.01, initially_correct=0)) == "preexisting_target"
    assert audit_outcome(_metric_row(1, 0.01, initially_correct=1)) == "induced_success"
