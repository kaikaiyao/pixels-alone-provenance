from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from passive_provenance.attacks.pgd import generic_score_from_probs, pgd_targeted, pgd_targeted_generic, score_from_probs
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
    parser = argparse.ArgumentParser(description="Run white-box PGD attacks against a differentiable verifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--verifier-id", required=True)
    parser.add_argument("--subset", choices=["full", "smoke"], default="full")
    return parser.parse_args()


def image_tensor(path: str) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def task_specs(benchmark_id: str, class_names: tuple[str, ...]) -> list[tuple[str, str]]:
    ai_names = [name for name in class_names if name not in {"real", "fake"}]
    if benchmark_id == "b0" and class_names == ("real", "sd15", "sdxl"):
        return [("real", "fake"), ("sd15", "real"), ("sdxl", "real"), ("sd15", "sdxl"), ("sdxl", "sd15")]
    if class_names == ("real", "fake"):
        return [("real", "fake"), ("fake", "real")]
    tasks = [("real", "coarse_ai")]
    if "real" in class_names:
        for source in ai_names:
            tasks.append((source, "real"))
    for source in ai_names:
        for target in ai_names:
            if source != target:
                tasks.append((source, target))
    return tasks


def select_records(records, split: str, source: str):
    selected = []
    for record in records:
        if record.split != split:
            continue
        if source == "real" and record.generator_id == "real":
            selected.append(record)
        elif source == "fake" and record.generator_id != "real":
            selected.append(record)
        elif record.generator_id == source:
            selected.append(record)
    return selected


def success_from_target(probabilities: torch.Tensor, class_names: tuple[str, ...], target: str, threshold: float) -> np.ndarray:
    if target in class_names:
        target_index = class_names.index(target)
        return (probabilities.argmax(dim=1).detach().cpu().numpy() == target_index).astype(np.int32)
    target_scores = generic_score_from_probs(probabilities, class_names, target).detach().cpu().numpy()
    if target == "real":
        return (probabilities.argmax(dim=1).detach().cpu().numpy() == class_names.index("real")).astype(np.int32)
    return (target_scores >= threshold).astype(np.int32)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    spec = get_verifier_spec(config, args.verifier_id)
    if not spec.differentiable:
        raise RuntimeError(f"attack-whitebox requires a differentiable verifier, got {args.verifier_id}.")
    model = load_generic_verifier_checkpoint(str(verifier_checkpoint_path(config, spec.benchmark_id, spec.verifier_id)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    records = load_sample_manifest(sample_manifest_path(config, spec.benchmark_id, args.subset))
    test_records = [record for record in records if record.split == "test"]
    clean_metrics = read_json(verifier_metrics_path(config, spec.benchmark_id, spec.verifier_id))
    binary_threshold = float(clean_metrics.get("binary_threshold", 0.5))
    rows = []
    summary_rows = []
    nested_summary: dict[str, dict[str, dict[str, float]]] = {}
    epsilons = [float(item) for item in config["attacks"]["epsilons"]]
    batch_size = int(config["attacks"]["batch_size"])

    clean_target_cache: dict[str, float] = {}
    for target in {target for _, target in task_specs(spec.benchmark_id, spec.class_names)}:
        targets = select_records(test_records, "test", "fake" if target == "coarse_ai" else target)
        if not targets:
            continue
        images = torch.stack([image_tensor(record.image_path) for record in targets]).to(device)
        with torch.no_grad():
            probs = model.forward_probs(images)
        if target in {"coarse_ai", "fake"}:
            clean_target_cache[target] = float(generic_score_from_probs(probs, spec.class_names, target).mean().item())
        else:
            clean_target_cache[target] = float(generic_score_from_probs(probs, spec.class_names, target).mean().item())

    for source, target in task_specs(spec.benchmark_id, spec.class_names):
        source_records = select_records(test_records, "test", source)
        if not source_records:
            continue
        task_name = f"{source}_to_{target}"
        nested_summary[task_name] = {}
        for eps in epsilons:
            batch_adv_scores = []
            batch_success = []
            for start in range(0, len(source_records), batch_size):
                batch_records = source_records[start : start + batch_size]
                clean_images = torch.stack([image_tensor(record.image_path) for record in batch_records]).to(device)
                with torch.no_grad():
                    clean_probs = model.forward_probs(clean_images)
                attack = pgd_targeted_generic(
                    model.forward_probs,
                    clean_images,
                    class_names=spec.class_names,
                    eps=eps,
                    steps=int(config["attacks"]["steps"]),
                    step_size=eps / 4.0,
                    restarts=int(config["attacks"]["restarts"]),
                    target=target,
                )
                with torch.no_grad():
                    adv_probs = model.forward_probs(attack.adversarial)
                clean_target_scores = generic_score_from_probs(clean_probs, spec.class_names, target).detach().cpu().numpy()
                adv_target_scores = generic_score_from_probs(adv_probs, spec.class_names, target).detach().cpu().numpy()
                success = success_from_target(adv_probs, spec.class_names, target, binary_threshold)
                distances = summarize_attack_distances(clean_images, attack.adversarial)
                batch_adv_scores.extend(list(adv_target_scores))
                batch_success.extend(list(success))
                for idx, record in enumerate(batch_records):
                    rows.append(
                        {
                            "benchmark_id": spec.benchmark_id,
                            "verifier_id": spec.verifier_id,
                            "interface_id": "full_vector",
                            "image_id": record.image_id,
                            "split": "test",
                            "source": source if source != "real" else "human",
                            "target": target,
                            "detector": spec.verifier_id,
                            "attack": "whitebox_pgd",
                            "eps": eps,
                            "pi": "",
                            "seed": record.seed,
                            "clean_target_score": float(clean_target_scores[idx]),
                            "adv_target_score": float(adv_target_scores[idx]),
                            "clean_pred": int(clean_probs[idx].argmax().item()),
                            "adv_pred": int(adv_probs[idx].argmax().item()),
                            "success": int(success[idx]),
                            "linf": float(distances["linf"][idx].item()),
                            "psnr": float(distances["psnr"][idx].item()),
                            "lpips": "",
                            "emulator_eps95": "",
                            "emulator_max_error": "",
                            "alpha_estimate": "",
                            "recovered_whitebox_gap": "",
                            "query_count": "",
                            "query_budget": "",
                            "optimizer": "pgd",
                            "abstained": "",
                            "rate_limited": "",
                        }
                    )
            task_summary = {
                "asr": float(np.mean(batch_success)) if batch_success else 0.0,
                "clean_target_acceptance": clean_target_cache.get(target, 0.0),
                "adv_target_score_mean": float(np.mean(batch_adv_scores)) if batch_adv_scores else 0.0,
                "robust_gap": robust_gap(clean_target_cache.get(target, 0.0), np.asarray(batch_adv_scores, dtype=np.float32)) if batch_adv_scores else 0.0,
            }
            nested_summary[task_name][str(eps)] = task_summary
            summary_rows.append(
                {
                    "benchmark_id": spec.benchmark_id,
                    "verifier_id": spec.verifier_id,
                    "interface_id": "full_vector",
                    "attack_family": "whitebox_pgd",
                    "source": source,
                    "target": target,
                    "eps": eps,
                    "pi": "",
                    "asr": task_summary["asr"],
                    "robust_gap": task_summary["robust_gap"],
                    "clean_target_acceptance": task_summary["clean_target_acceptance"],
                    "adv_target_score_mean": task_summary["adv_target_score_mean"],
                    "query_budget": "",
                    "emulator_eps95": "",
                    "emulator_max_error": "",
                    "recovered_whitebox_gap": "",
                    "alpha_estimate": "",
                    "optimizer": "pgd",
                }
            )

    if spec.benchmark_id == "b0" and spec.class_names == ("real", "sd15", "sdxl"):
        human_records = select_records(test_records, "test", "real")
        human_images = torch.stack([image_tensor(record.image_path) for record in human_records]).to(device)
        with torch.no_grad():
            human_clean_probs = model.forward_probs(human_images)
        for pi in [float(item) for item in config["attacks"]["mixture_pis"]]:
            for attack_name, target_name in [("oracle-pi", "mixture"), ("coarse-only", "fake")]:
                attack = pgd_targeted(
                    model.forward_probs,
                    human_images,
                    eps=max(epsilons),
                    steps=int(config["attacks"]["steps"]),
                    step_size=max(epsilons) / 4.0,
                    restarts=int(config["attacks"]["restarts"]),
                    target=target_name,
                    pi=pi if target_name == "mixture" else None,
                )
                with torch.no_grad():
                    adv_probs = model.forward_probs(attack.adversarial)
                clean_target = score_from_probs(human_clean_probs, target_name, pi if target_name == "mixture" else None).detach().cpu().numpy()
                adv_target = score_from_probs(adv_probs, target_name, pi if target_name == "mixture" else None).detach().cpu().numpy()
                success = (adv_target > clean_target).astype(np.int32)
                summary_rows.append(
                    {
                        "benchmark_id": spec.benchmark_id,
                        "verifier_id": spec.verifier_id,
                        "interface_id": "full_vector",
                        "attack_family": attack_name,
                        "source": "real",
                        "target": "mixture" if target_name == "mixture" else "fake",
                        "eps": max(epsilons),
                        "pi": pi,
                        "asr": float(success.mean()),
                        "robust_gap": robust_gap(float(clean_target.mean()), adv_target),
                        "clean_target_acceptance": float(clean_target.mean()),
                        "adv_target_score_mean": float(adv_target.mean()),
                        "query_budget": "",
                        "emulator_eps95": "",
                        "emulator_max_error": "",
                        "recovered_whitebox_gap": "",
                        "alpha_estimate": "",
                        "optimizer": "pgd",
                    }
                )

    results_path = attack_results_path(config, spec.benchmark_id, spec.verifier_id, "whitebox")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(results_path, rows, RESULT_COLUMNS)
    write_csv(results_path.with_name("whitebox_summary.csv"), summary_rows, SUMMARY_COLUMNS)
    write_json(
        attack_summary_path(config, spec.benchmark_id, spec.verifier_id, "whitebox"),
        {"benchmark_id": spec.benchmark_id, "verifier_id": spec.verifier_id, "tasks": nested_summary, "summary_rows": summary_rows},
    )


if __name__ == "__main__":
    main()
