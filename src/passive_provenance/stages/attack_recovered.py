from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from passive_provenance.attacks.pgd import generic_score_from_probs, pgd_targeted_generic
from passive_provenance.config import load_config
from passive_provenance.constants import RESULT_COLUMNS, SUMMARY_COLUMNS
from passive_provenance.datasets import load_sample_manifest
from passive_provenance.matrix import (
    attack_results_path,
    attack_summary_path,
    feature_cache_path,
    get_verifier_spec,
    sample_manifest_path,
    verifier_checkpoint_path,
)
from passive_provenance.models.generic import load_generic_verifier_checkpoint
from passive_provenance.pipelines.exact_recovery import recover_softmax_relative_head, relative_parameter_error, score_error
from passive_provenance.utils.io import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover a score-revealing linear head and compare against white-box attacks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--verifier-id", required=True)
    parser.add_argument("--subset", choices=["full", "smoke"], default="full")
    parser.add_argument("--source-label")
    parser.add_argument("--target-label")
    return parser.parse_args()


def image_tensor(path: str) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def default_task(spec) -> tuple[str, str]:
    if spec.class_names == ("real", "fake"):
        return ("fake", "real")
    ai_names = [name for name in spec.class_names if name != "real"]
    if "sd15" in ai_names and "sdxl" in ai_names:
        return ("sd15", "sdxl")
    if len(ai_names) >= 2:
        return (ai_names[0], ai_names[1])
    return ("real", "fake")


def select_records(records, split: str, source: str):
    selected = []
    for record in records:
        if record.split != split:
            continue
        if source == "fake" and record.generator_id != "real":
            selected.append(record)
        elif record.generator_id == source:
            selected.append(record)
    return selected


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    spec = get_verifier_spec(config, args.verifier_id)
    if not spec.score_revealing:
        raise RuntimeError(f"attack-recovered only supports score-revealing verifiers, got {args.verifier_id}.")
    model = load_generic_verifier_checkpoint(str(verifier_checkpoint_path(config, spec.benchmark_id, spec.verifier_id)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    cache = torch.load(feature_cache_path(config, spec.benchmark_id, args.subset), map_location="cpu")
    probe_features = torch.cat([cache["splits"][split]["features"] for split in ("train", "val", "test")], dim=0)
    with torch.no_grad():
        probe_probs = model.head_probabilities(probe_features.to(device)).detach().cpu()
    source_label, target_label = default_task(spec)
    if args.source_label:
        source_label = args.source_label
    if args.target_label:
        target_label = args.target_label
    records = load_sample_manifest(sample_manifest_path(config, spec.benchmark_id, args.subset))
    source_records = select_records(records, "test", source_label)
    source_images = torch.stack([image_tensor(record.image_path) for record in source_records]).to(device)
    with torch.no_grad():
        native_probs = model.forward_probs(source_images)
    eps = max(float(item) for item in config["attacks"]["epsilons"])
    native_attack = pgd_targeted_generic(
        model.forward_probs,
        source_images,
        class_names=spec.class_names,
        eps=eps,
        steps=int(config["attacks"]["steps"]),
        step_size=eps / 4.0,
        restarts=int(config["attacks"]["restarts"]),
        target=target_label,
    )
    with torch.no_grad():
        native_adv_probs = model.forward_probs(native_attack.adversarial)
    native_scores = generic_score_from_probs(native_adv_probs, spec.class_names, target_label)

    requested_counts = [int(item) for item in config["exact_recovery"]["query_counts"]]
    available_queries = int(probe_features.shape[0])
    query_counts = sorted({min(item, available_queries) for item in requested_counts if item > 0} | {available_queries})
    true_weights, true_bias = model.effective_linear_parameters()
    summary_rows = []
    curve_points = []
    best_payload = None
    for query_count in query_counts:
        recovered = recover_softmax_relative_head(probe_features[:query_count], probe_probs[:query_count], anchor_index=int(config["exact_recovery"]["anchor_index"])).to(device)
        parameter_metrics = relative_parameter_error(true_weights.to(recovered.weights.device), true_bias.to(recovered.bias.device), recovered)

        def recovered_forward(images: torch.Tensor, recovered_head=recovered) -> torch.Tensor:
            features = model.encode(images)
            return recovered_head.probabilities(features)

        recovered_attack = pgd_targeted_generic(
            recovered_forward,
            source_images,
            class_names=spec.class_names,
            eps=eps,
            steps=int(config["attacks"]["steps"]),
            step_size=eps / 4.0,
            restarts=int(config["attacks"]["restarts"]),
            target=target_label,
        )
        with torch.no_grad():
            recovered_adv_probs = model.forward_probs(recovered_attack.adversarial)
            recovered_test_probs = recovered.probabilities(model.encode(source_images))
        recovered_scores = generic_score_from_probs(recovered_adv_probs, spec.class_names, target_label)
        point = {
            "query_count": query_count,
            "weight_error": parameter_metrics["weight_error"],
            "bias_error": parameter_metrics["bias_error"],
            "score_error": score_error(native_probs, recovered_test_probs),
            "recovered_whitebox_gap": float((native_scores - recovered_scores).abs().mean().item()),
            "asr": float((recovered_adv_probs.argmax(dim=1) == spec.class_names.index(target_label)).float().mean().item()) if target_label in spec.class_names else float((recovered_scores >= 0.5).float().mean().item()),
        }
        curve_points.append(point)
        summary_rows.append(
            {
                "benchmark_id": spec.benchmark_id,
                "verifier_id": spec.verifier_id,
                "interface_id": "full_vector",
                "attack_family": "recovered_head",
                "source": source_label,
                "target": target_label,
                "eps": eps,
                "pi": "",
                "asr": point["asr"],
                "robust_gap": "",
                "clean_target_acceptance": "",
                "adv_target_score_mean": float(recovered_scores.mean().item()),
                "query_budget": query_count,
                "emulator_eps95": point["score_error"],
                "emulator_max_error": "",
                "recovered_whitebox_gap": point["recovered_whitebox_gap"],
                "alpha_estimate": "",
                "optimizer": "pgd",
            }
        )
        if query_count == available_queries:
            best_payload = {
                "recovered_adv_probs": recovered_adv_probs,
                "recovered_scores": recovered_scores,
                "recovered_attack": recovered_attack,
                "metrics": point,
            }

    if best_payload is None:
        raise RuntimeError("attack-recovered did not produce a terminal recovered model.")
    rows = []
    for idx, record in enumerate(source_records):
        rows.append(
            {
                "benchmark_id": spec.benchmark_id,
                "verifier_id": spec.verifier_id,
                "interface_id": "full_vector",
                "image_id": record.image_id,
                "split": "test",
                "source": source_label,
                "target": target_label,
                "detector": f"{spec.verifier_id}_recovered",
                "attack": "recovered_head",
                "eps": eps,
                "pi": "",
                "seed": record.seed,
                "clean_target_score": float(generic_score_from_probs(native_probs, spec.class_names, target_label)[idx].item()),
                "adv_target_score": float(best_payload["recovered_scores"][idx].item()),
                "clean_pred": int(native_probs[idx].argmax().item()),
                "adv_pred": int(best_payload["recovered_adv_probs"][idx].argmax().item()),
                "success": int(best_payload["recovered_adv_probs"][idx].argmax().item() == spec.class_names.index(target_label)) if target_label in spec.class_names else int(best_payload["recovered_scores"][idx].item() >= 0.5),
                "linf": float((best_payload["recovered_attack"].adversarial[idx] - source_images[idx]).abs().max().item()),
                "psnr": "",
                "lpips": "",
                "emulator_eps95": best_payload["metrics"]["score_error"],
                "emulator_max_error": "",
                "alpha_estimate": "",
                "recovered_whitebox_gap": best_payload["metrics"]["recovered_whitebox_gap"],
                "query_count": available_queries,
                "query_budget": available_queries,
                "optimizer": "pgd",
                "abstained": "",
                "rate_limited": "",
            }
        )
    results_path = attack_results_path(config, spec.benchmark_id, spec.verifier_id, "recovered")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(results_path, rows, RESULT_COLUMNS)
    write_csv(results_path.with_name("recovered_summary.csv"), summary_rows, SUMMARY_COLUMNS)
    write_json(
        attack_summary_path(config, spec.benchmark_id, spec.verifier_id, "recovered"),
        {
            "benchmark_id": spec.benchmark_id,
            "verifier_id": spec.verifier_id,
            "source": source_label,
            "target": target_label,
            "available_queries": available_queries,
            "points": curve_points,
            "best": best_payload["metrics"],
        },
    )


if __name__ == "__main__":
    main()
