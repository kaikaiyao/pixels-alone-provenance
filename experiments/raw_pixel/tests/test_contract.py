from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("raw_pixel_experiment", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(path: Path, *, split: str, prompt_id: int, generator_id: str):
    path.touch()
    return MODULE.ManifestRow(
        benchmark_id="b1",
        prompt_id=prompt_id,
        image_id=prompt_id,
        split=split,
        generator_id=generator_id,
        seed=prompt_id,
        image_path=str(path),
    )


def test_manifest_validation_enforces_prompt_disjointness(tmp_path: Path) -> None:
    rows = [
        row(tmp_path / "train.png", split="train", prompt_id=1, generator_id="real"),
        row(tmp_path / "val.png", split="val", prompt_id=2, generator_id="sd15"),
        row(tmp_path / "test.png", split="test", prompt_id=3, generator_id="sdxl"),
    ]
    summary = MODULE.validate_manifest(rows)
    assert summary["prompt_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}
    leaked = list(rows) + [
        row(tmp_path / "leaked.png", split="test", prompt_id=1, generator_id="pixart")
    ]
    with pytest.raises(RuntimeError, match="Prompt leakage"):
        MODULE.validate_manifest(leaked)


def test_normalization_preserves_eval_shape_and_gradient() -> None:
    images = torch.rand(2, 3, 224, 224, requires_grad=True)
    processed = MODULE.normalize_eval_batch(
        images,
        mean=MODULE.IMAGENET_MEAN,
        std=MODULE.IMAGENET_STD,
    )
    assert processed.shape == (2, 3, 224, 224)
    processed.sum().backward()
    assert images.grad is not None


def test_different_source_sizes_share_one_eval_attack_domain(tmp_path: Path) -> None:
    first_path = tmp_path / "wide.png"
    second_path = tmp_path / "tall.png"
    Image.new("RGB", (640, 360), color=(32, 64, 96)).save(first_path)
    Image.new("RGB", (300, 700), color=(96, 64, 32)).save(second_path)
    rows = [
        MODULE.ManifestRow("b1", 1, 1, "test", "real", 1, str(first_path)),
        MODULE.ManifestRow("b1", 2, 2, "test", "sd15", 2, str(second_path)),
    ]
    batch = MODULE.load_raw_batch(rows, torch.device("cpu"))
    assert batch.shape == (2, 3, 224, 224)
    assert 0.0 <= float(batch.min()) <= float(batch.max()) <= 1.0


def test_fake_attack_sampling_is_generator_balanced(tmp_path: Path) -> None:
    rows = []
    for generator_index, generator_id in enumerate(MODULE.FAKE_GENERATORS):
        for index in range(3):
            rows.append(
                row(
                    tmp_path / f"{generator_id}_{index}.png",
                    split="test",
                    prompt_id=generator_index * 10 + index,
                    generator_id=generator_id,
                )
            )
    selected = MODULE.stratified_fake_rows(rows, limit=8, seed=7)
    counts = {name: sum(item.generator_id == name for item in selected) for name in MODULE.FAKE_GENERATORS}
    assert counts == {name: 2 for name in MODULE.FAKE_GENERATORS}


def test_linf_projection_in_toy_surrogate_attack() -> None:
    class ToyClip(torch.nn.Module):
        def encode_image(self, images: torch.Tensor) -> torch.Tensor:
            return images.mean(dim=(2, 3))

    surrogate = MODULE.ScoreMLP(feature_dim=3, hidden_dim=4)
    originals = torch.full((2, 3, 224, 224), 0.5)
    adversarial = MODULE.targeted_pgd_surrogate(
        originals=originals,
        clip_model=ToyClip(),
        surrogate=surrogate,
        target_label=1,
        eps=8.0 / 255.0,
        steps=2,
        restarts=1,
        seed=11,
    )
    assert adversarial.shape == originals.shape
    assert float((adversarial - originals).abs().max()) <= 8.0 / 255.0 + 1e-6
    assert float(adversarial.min()) >= 0.0
    assert float(adversarial.max()) <= 1.0


@pytest.mark.parametrize(
    ("direction", "adversarial_fake_scores", "clean_target_mean", "expected_adversarial_target", "expected_gap"),
    [
        ("real_to_fake", [0.7, 0.9], 0.95, 0.8, 0.15),
        ("fake_to_real", [0.2, 0.4], 0.90, 0.7, 0.20),
    ],
)
def test_attack_summary_uses_directional_target_scores_and_labeled_asr_populations(
    direction: str,
    adversarial_fake_scores: list[float],
    clean_target_mean: float,
    expected_adversarial_target: float,
    expected_gap: float,
) -> None:
    cell = [
        {
            "initially_correct": 1,
            "success": 1,
            "clean_fake_score": 0.1,
            "adversarial_fake_score": adversarial_fake_scores[0],
            "linf": 8.0 / 255.0,
            "psnr": 32.0,
            "generator_id": "real" if direction == "real_to_fake" else "sd15",
        },
        {
            "initially_correct": 0,
            "success": 0,
            "clean_fake_score": 0.9,
            "adversarial_fake_score": adversarial_fake_scores[1],
            "linf": 7.0 / 255.0,
            "psnr": 34.0,
            "generator_id": "real" if direction == "real_to_fake" else "sdxl",
        },
    ]
    summary = MODULE.summarize_attack_direction(
        direction=direction,
        cell=cell,
        clean_target_metrics={
            "clean_target_acceptance_mean": clean_target_mean,
            "clean_target_distribution_n": 20,
            "clean_target_distribution_source_generators": {"fake" if direction == "real_to_fake" else "real": 20},
        },
    )

    assert summary["targeted_asr"] == pytest.approx(0.5)
    assert summary["targeted_asr_population"] == "all_attacked_sources"
    assert summary["targeted_asr_initially_correct"] == pytest.approx(1.0)
    assert summary["targeted_asr_all"] == pytest.approx(0.5)
    assert summary["clean_target_acceptance_mean"] == pytest.approx(clean_target_mean)
    assert summary["adversarial_source_target_score_mean"] == pytest.approx(expected_adversarial_target)
    assert summary["empirical_gap_upper_bound"] == pytest.approx(expected_gap)
    assert summary["empirical_gap_population"] == (
        "all_attacked_sources_vs_full_clean_test_target_distribution"
    )


def test_validation_gate_allows_attack_but_test_controls_claim_eligibility() -> None:
    decision = MODULE.clean_evidence_decision(val_auc=0.90, test_auc=0.80, threshold=0.85)
    assert decision["attack_execution_gate"] == "validation_auc"
    assert decision["attack_execution_gate_pass"] is True
    assert decision["gate_pass"] is True
    assert decision["test_auc_reporting_only"] is True
    assert decision["claim_eligible"] is False


def test_full_scientific_contract_is_fixed(tmp_path: Path) -> None:
    args = SimpleNamespace(subset="full", **MODULE.FULL_SCIENTIFIC_HYPERPARAMETERS)
    MODULE.validate_full_scientific_contract(args, tmp_path / "sample_manifest_full.csv")
    args.clip_model = "ViT-L-14"
    with pytest.raises(RuntimeError, match="clip_model='ViT-L-14'"):
        MODULE.validate_full_scientific_contract(args, tmp_path / "sample_manifest_full.csv")
    args.clip_model = "ViT-B-32"
    args.attack_restarts = 6
    with pytest.raises(RuntimeError, match="FULL_SCIENTIFIC_CONTRACT_MISMATCH"):
        MODULE.validate_full_scientific_contract(args, tmp_path / "sample_manifest_full.csv")


def test_manifest_resolves_relative_paths_with_optional_image_root(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    absolute_image = tmp_path / "absolute.png"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MODULE.ManifestRow.__dataclass_fields__))
        writer.writeheader()
        for index, image_path in enumerate(("images/relative.png", str(absolute_image))):
            writer.writerow({
                "benchmark_id": "expanded",
                "prompt_id": index,
                "image_id": index,
                "split": "train",
                "generator_id": "real",
                "seed": index,
                "image_path": image_path,
            })
    default_rows = MODULE.read_manifest(manifest)
    assert default_rows[0].image_path == str(tmp_path / "images/relative.png")
    assert default_rows[1].image_path == str(absolute_image)
    custom_root = tmp_path / "external-data"
    custom_rows = MODULE.read_manifest(manifest, custom_root)
    assert custom_rows[0].image_path == str(custom_root / "images/relative.png")
    assert custom_rows[1].image_path == str(absolute_image)


def test_staged_victim_provenance_rejects_manifest_or_checkpoint_mismatch() -> None:
    args = SimpleNamespace(arch="resnet18", seed=20260723)
    payload = {
        "architecture": "resnet18",
        "seed": 20260723,
        "manifest_sha256": "manifest-hash",
        "victim_family": "raw_pixel_imagenet_resnet",
    }
    metrics = {
        **payload,
        "checkpoint_sha256": "checkpoint-hash",
    }
    MODULE.validate_victim_provenance(
        checkpoint_payload=payload,
        clean_metrics=metrics,
        args=args,
        manifest_sha256="manifest-hash",
        checkpoint_sha256="checkpoint-hash",
    )
    with pytest.raises(RuntimeError, match="VICTIM_PROVENANCE_MISMATCH"):
        MODULE.validate_victim_provenance(
            checkpoint_payload=payload,
            clean_metrics=metrics,
            args=args,
            manifest_sha256="different-manifest",
            checkpoint_sha256="checkpoint-hash",
        )
