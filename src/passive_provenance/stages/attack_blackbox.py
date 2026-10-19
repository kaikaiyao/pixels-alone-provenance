from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from passive_provenance.attacks.pgd import generic_score_from_probs
from passive_provenance.config import load_config
from passive_provenance.constants import RESULT_COLUMNS, SUMMARY_COLUMNS
from passive_provenance.external_detectors import build_external_detector
from passive_provenance.matrix import (
    attack_results_path,
    attack_summary_path,
    emulator_checkpoint_path,
    emulator_metrics_path,
    get_interface_spec,
    get_verifier_spec,
    verifier_checkpoint_path,
    verifier_metrics_path,
)
from passive_provenance.models.generic import FrozenClipBackbone, load_generic_verifier_checkpoint
from passive_provenance.pipelines.emulator import FeatureEmulator
from passive_provenance.utils.io import read_json, write_csv, write_json
from passive_provenance.utils.metrics import robust_gap, summarize_attack_distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attack a verifier through a learned emulator.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--verifier-id", required=True)
    parser.add_argument("--interface-id", required=True)
    parser.add_argument("--source-label")
    parser.add_argument("--target-label")
    return parser.parse_args()


def image_tensor(path: str) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def load_emulator_payload(path: str) -> tuple[FeatureEmulator, dict[str, object]]:
    payload = torch.load(path, map_location="cpu")
    model = FeatureEmulator(int(payload["input_dim"]), int(payload["output_dim"]), int(payload["hidden_dim"]), "vector" if int(payload["output_dim"]) > 1 else "binary")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def score_from_emulator_logits(logits: torch.Tensor, *, mode: str, class_names: tuple[str, ...], target_label: str) -> torch.Tensor:
    if mode == "vector":
        probabilities = F.softmax(logits, dim=1)
        return generic_score_from_probs(probabilities, class_names, target_label)
    if mode == "label":
        probabilities = F.softmax(logits, dim=1)
        return generic_score_from_probs(probabilities, class_names, target_label)
    if mode == "scalar":
        score = logits.squeeze(1)
        return score if target_label != "real" else 1.0 - score
    if mode == "binary":
        fake_score = torch.sigmoid(logits.squeeze(1))
        return fake_score if target_label != "real" else 1.0 - fake_score
    if mode == "ternary":
        probabilities = F.softmax(logits, dim=1)
        target_index = 0 if target_label == "real" else 1
        return probabilities[:, target_index]
    raise ValueError(f"Unsupported emulator mode {mode}.")


def surrogate_target_score(emulator: FeatureEmulator, features: torch.Tensor, *, mode: str, class_names: tuple[str, ...], target_label: str) -> torch.Tensor:
    logits = emulator(features)
    return score_from_emulator_logits(logits, mode=mode, class_names=class_names, target_label=target_label)


class ExternalDetectorAttackVerifier:
    def __init__(self, backbone: FrozenClipBackbone, detector) -> None:
        self.backbone = backbone
        self.detector = detector
        self.class_names = tuple(detector.class_names)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(images)

    def forward_probs(self, images: torch.Tensor) -> torch.Tensor:
        return self.detector.predict_tensor_probabilities(images).to(images.device)


def attack_with_emulator(verifier, emulator, *, mode: str, class_names: tuple[str, ...], target_label: str, images: torch.Tensor, eps: float, steps: int, restarts: int) -> torch.Tensor:
    best_scores = torch.full((images.shape[0],), -float("inf"), device=images.device)
    best_adv = images.clone()
    step_size = eps / 4.0
    for restart_idx in range(restarts):
        if restart_idx == 0:
            current = images.clone()
        else:
            current = (images + torch.empty_like(images).uniform_(-eps, eps)).clamp(0.0, 1.0)
        current = current.detach()
        for _ in range(steps):
            current.requires_grad_(True)
            features = verifier.encode(current)
            scores = surrogate_target_score(emulator, features, mode=mode, class_names=class_names, target_label=target_label)
            loss = -torch.log(scores.clamp_min(1e-8)).sum()
            gradient = torch.autograd.grad(loss, current)[0]
            current = (current.detach() - step_size * gradient.sign()).clamp(0.0, 1.0)
            current = torch.max(torch.min(current, images + eps), images - eps)
        with torch.no_grad():
            features = verifier.encode(current)
            scores = surrogate_target_score(emulator, features, mode=mode, class_names=class_names, target_label=target_label)
            improved = scores > best_scores
            best_scores = torch.where(improved, scores, best_scores)
            best_adv = torch.where(improved.view(-1, 1, 1, 1), current, best_adv)
    return best_adv


def success(probabilities: torch.Tensor, class_names: tuple[str, ...], target_label: str, threshold: float) -> np.ndarray:
    if target_label in class_names:
        return (probabilities.argmax(dim=1).detach().cpu().numpy() == class_names.index(target_label)).astype(np.int32)
    scores = generic_score_from_probs(probabilities, class_names, target_label).detach().cpu().numpy()
    return (scores >= threshold).astype(np.int32)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    verifier_spec = get_verifier_spec(config, args.verifier_id)
    interface_spec = get_interface_spec(config, args.interface_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verifier_spec.family == "external_detector":
        backbone = FrozenClipBackbone(
            clip_model_name=config["d1"]["clip_model_name"],
            clip_pretrained=config["d1"]["clip_pretrained"],
            input_size=int(config["data"]["clip_input_size"]),
            mean=tuple(config["data"]["clip_mean"]),
            std=tuple(config["data"]["clip_std"]),
        ).to(device)
        backbone.eval()
        detector = build_external_detector(config, verifier_spec.verifier_id, device=device)
        verifier = ExternalDetectorAttackVerifier(backbone, detector)
    else:
        verifier = load_generic_verifier_checkpoint(str(verifier_checkpoint_path(config, verifier_spec.benchmark_id, verifier_spec.verifier_id)))
        verifier = verifier.to(device)
        verifier.eval()
    emulator, emulator_payload = load_emulator_payload(str(emulator_checkpoint_path(config, verifier_spec.benchmark_id, verifier_spec.verifier_id, interface_spec.interface_id)))
    emulator = emulator.to(device)
    mode = str(emulator_payload["mode"])
    target_label = args.target_label or str(emulator_payload["target_label"])
    source_label = args.source_label or ("real" if target_label != "real" else "fake")
    clean_metrics = read_json(verifier_metrics_path(config, verifier_spec.benchmark_id, verifier_spec.verifier_id))
    binary_threshold = float(clean_metrics.get("binary_threshold", 0.5))
    metrics = read_json(emulator_metrics_path(config, verifier_spec.benchmark_id, verifier_spec.verifier_id, interface_spec.interface_id))

    from passive_provenance.datasets import load_sample_manifest
    from passive_provenance.matrix import sample_manifest_path

    records = load_sample_manifest(sample_manifest_path(config, verifier_spec.benchmark_id, "full"))
    source_records = [
        record
        for record in records
        if record.split == "test" and ((source_label == "fake" and record.generator_id != "real") or record.generator_id == source_label)
    ]
    eps = max(float(item) for item in config["attacks"]["epsilons"])
    images = torch.stack([image_tensor(record.image_path) for record in source_records]).to(device)
    with torch.no_grad():
        clean_probs = verifier.forward_probs(images)
    adversarial = attack_with_emulator(
        verifier,
        emulator,
        mode=mode,
        class_names=verifier.class_names,
        target_label=target_label,
        images=images,
        eps=eps,
        steps=int(config["attacks"]["steps"]),
        restarts=int(config["attacks"]["restarts"]),
    )
    with torch.no_grad():
        adv_probs = verifier.forward_probs(adversarial)
    clean_scores = generic_score_from_probs(clean_probs, verifier.class_names, target_label).detach().cpu().numpy()
    adv_scores = generic_score_from_probs(adv_probs, verifier.class_names, target_label).detach().cpu().numpy()
    success_values = success(adv_probs, verifier.class_names, target_label, binary_threshold)
    distances = summarize_attack_distances(images, adversarial)
    rows = []
    for idx, record in enumerate(source_records):
        rows.append(
            {
                "benchmark_id": verifier_spec.benchmark_id,
                "verifier_id": verifier_spec.verifier_id,
                "interface_id": interface_spec.interface_id,
                "image_id": record.image_id,
                "split": "test",
                "source": source_label,
                "target": target_label,
                "detector": verifier_spec.verifier_id,
                "attack": "blackbox_emulator",
                "eps": eps,
                "pi": "",
                "seed": record.seed,
                "clean_target_score": float(clean_scores[idx]),
                "adv_target_score": float(adv_scores[idx]),
                "clean_pred": int(clean_probs[idx].argmax().item()),
                "adv_pred": int(adv_probs[idx].argmax().item()),
                "success": int(success_values[idx]),
                "linf": float(distances["linf"][idx].item()),
                "psnr": float(distances["psnr"][idx].item()),
                "lpips": "",
                "emulator_eps95": metrics["emulator_eps95"],
                "emulator_max_error": metrics["emulator_max_error"],
                "alpha_estimate": "",
                "recovered_whitebox_gap": "",
                "query_count": "",
                "query_budget": "",
                "optimizer": "pgd",
                "abstained": "",
                "rate_limited": int(interface_spec.rate_limit is not None),
            }
        )
    summary_rows = [
        {
            "benchmark_id": verifier_spec.benchmark_id,
            "verifier_id": verifier_spec.verifier_id,
            "interface_id": interface_spec.interface_id,
            "attack_family": "blackbox_emulator",
            "source": source_label,
            "target": target_label,
            "eps": eps,
            "pi": "",
            "asr": float(success_values.mean()),
            "robust_gap": robust_gap(float(clean_scores.mean()), adv_scores),
            "clean_target_acceptance": float(clean_scores.mean()),
            "adv_target_score_mean": float(adv_scores.mean()),
            "query_budget": interface_spec.rate_limit or "",
            "emulator_eps95": metrics["emulator_eps95"],
            "emulator_max_error": metrics["emulator_max_error"],
            "recovered_whitebox_gap": "",
            "alpha_estimate": "",
            "optimizer": "pgd",
        }
    ]
    results_path = attack_results_path(config, verifier_spec.benchmark_id, verifier_spec.verifier_id, "blackbox", interface_spec.interface_id)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(results_path, rows, RESULT_COLUMNS)
    write_csv(results_path.with_name("blackbox_summary.csv"), summary_rows, SUMMARY_COLUMNS)
    write_json(
        attack_summary_path(config, verifier_spec.benchmark_id, verifier_spec.verifier_id, "blackbox", interface_spec.interface_id),
        {"summary_rows": summary_rows, "source": source_label, "target": target_label},
    )


if __name__ == "__main__":
    main()
