from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from passive_provenance.attacks.pgd import generic_score_from_probs, pgd_targeted_generic
from passive_provenance.config import load_config
from passive_provenance.constants import RESULT_COLUMNS, SUMMARY_COLUMNS
from passive_provenance.matrix import (
    attack_results_path,
    attack_summary_path,
    get_verifier_spec,
    sample_manifest_path,
    verifier_checkpoint_path,
    verifier_metrics_path,
)
from passive_provenance.models.generic import load_generic_verifier_checkpoint
from passive_provenance.datasets import load_sample_manifest
from passive_provenance.utils.io import read_json, write_csv, write_json
from passive_provenance.utils.metrics import robust_gap, summarize_attack_distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run transfer-only attacks using a surrogate verifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--verifier-id", required=True)
    parser.add_argument("--surrogate-id", required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--target-label", required=True)
    return parser.parse_args()


def image_tensor(path: str) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def success(probabilities: torch.Tensor, class_names: tuple[str, ...], target_label: str, threshold: float) -> np.ndarray:
    if target_label in class_names:
        return (probabilities.argmax(dim=1).detach().cpu().numpy() == class_names.index(target_label)).astype(np.int32)
    scores = generic_score_from_probs(probabilities, class_names, target_label).detach().cpu().numpy()
    return (scores >= threshold).astype(np.int32)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    target_spec = get_verifier_spec(config, args.verifier_id)
    surrogate_spec = get_verifier_spec(config, args.surrogate_id)
    if target_spec.benchmark_id != surrogate_spec.benchmark_id:
        raise RuntimeError("Transfer attacks require surrogate and target verifiers on the same benchmark.")
    target_model = load_generic_verifier_checkpoint(str(verifier_checkpoint_path(config, target_spec.benchmark_id, target_spec.verifier_id)))
    surrogate_model = load_generic_verifier_checkpoint(str(verifier_checkpoint_path(config, surrogate_spec.benchmark_id, surrogate_spec.verifier_id)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model = target_model.to(device).eval()
    surrogate_model = surrogate_model.to(device).eval()
    records = load_sample_manifest(sample_manifest_path(config, target_spec.benchmark_id, "full"))
    source_records = [
        record
        for record in records
        if record.split == "test" and ((args.source_label == "fake" and record.generator_id != "real") or record.generator_id == args.source_label)
    ]
    images = torch.stack([image_tensor(record.image_path) for record in source_records]).to(device)
    eps = max(float(item) for item in config["attacks"]["epsilons"])
    with torch.no_grad():
        clean_probs = target_model.forward_probs(images)
    attack = pgd_targeted_generic(
        surrogate_model.forward_probs,
        images,
        class_names=surrogate_model.class_names,
        eps=eps,
        steps=int(config["attacks"]["steps"]),
        step_size=eps / 4.0,
        restarts=int(config["attacks"]["restarts"]),
        target=args.target_label,
    )
    with torch.no_grad():
        adv_probs = target_model.forward_probs(attack.adversarial)
    clean_metrics = read_json(verifier_metrics_path(config, target_spec.benchmark_id, target_spec.verifier_id))
    binary_threshold = float(clean_metrics.get("binary_threshold", 0.5))
    clean_scores = generic_score_from_probs(clean_probs, target_model.class_names, args.target_label).detach().cpu().numpy()
    adv_scores = generic_score_from_probs(adv_probs, target_model.class_names, args.target_label).detach().cpu().numpy()
    success_values = success(adv_probs, target_model.class_names, args.target_label, binary_threshold)
    distances = summarize_attack_distances(images, attack.adversarial)
    rows = []
    for idx, record in enumerate(source_records):
        rows.append(
            {
                "benchmark_id": target_spec.benchmark_id,
                "verifier_id": target_spec.verifier_id,
                "interface_id": "zero_query_transfer",
                "image_id": record.image_id,
                "split": "test",
                "source": args.source_label,
                "target": args.target_label,
                "detector": target_spec.verifier_id,
                "attack": "transfer_only",
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
                "emulator_eps95": "",
                "emulator_max_error": "",
                "alpha_estimate": "",
                "recovered_whitebox_gap": "",
                "query_count": 0,
                "query_budget": 0,
                "optimizer": "pgd",
                "abstained": "",
                "rate_limited": 0,
            }
        )
    summary_rows = [
        {
            "benchmark_id": target_spec.benchmark_id,
            "verifier_id": target_spec.verifier_id,
            "interface_id": "zero_query_transfer",
            "attack_family": "transfer_only",
            "source": args.source_label,
            "target": args.target_label,
            "eps": eps,
            "pi": "",
            "asr": float(success_values.mean()),
            "robust_gap": robust_gap(float(clean_scores.mean()), adv_scores),
            "clean_target_acceptance": float(clean_scores.mean()),
            "adv_target_score_mean": float(adv_scores.mean()),
            "query_budget": 0,
            "emulator_eps95": "",
            "emulator_max_error": "",
            "recovered_whitebox_gap": "",
            "alpha_estimate": "",
            "optimizer": "pgd",
        }
    ]
    results_path = attack_results_path(config, target_spec.benchmark_id, target_spec.verifier_id, "transfer")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(results_path, rows, RESULT_COLUMNS)
    write_csv(results_path.with_name("transfer_summary.csv"), summary_rows, SUMMARY_COLUMNS)
    write_json(attack_summary_path(config, target_spec.benchmark_id, target_spec.verifier_id, "transfer"), {"summary_rows": summary_rows})


if __name__ == "__main__":
    main()
