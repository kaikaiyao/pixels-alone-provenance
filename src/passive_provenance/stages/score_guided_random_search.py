from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from passive_provenance.attacks.pgd import project_linf, generic_score_from_probs
from passive_provenance.config import load_config
from passive_provenance.constants import SUMMARY_COLUMNS
from passive_provenance.matrix import attack_summary_path, get_verifier_spec, sample_manifest_path, verifier_checkpoint_path, verifier_metrics_path
from passive_provenance.models.generic import load_generic_verifier_checkpoint
from passive_provenance.datasets import load_sample_manifest
from passive_provenance.utils.io import read_json, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score-guided random-search sanity check with threshold-crossing success.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--verifier-id", required=True)
    parser.add_argument("--source-label", default="real")
    parser.add_argument("--target-label", default="fake")
    parser.add_argument("--max-images", type=int, default=32)
    return parser.parse_args()


def image_tensor(path: str) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    spec = get_verifier_spec(config, args.verifier_id)
    if spec.family == "external_detector":
        raise RuntimeError("Score-guided random search currently supports CLIP-family verifiers only.")
    model = load_generic_verifier_checkpoint(str(verifier_checkpoint_path(config, spec.benchmark_id, spec.verifier_id)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    records = load_sample_manifest(sample_manifest_path(config, spec.benchmark_id, "full"))
    source_records = [
        record
        for record in records
        if record.split == "test" and ((args.source_label == "fake" and record.generator_id != "real") or record.generator_id == args.source_label)
    ][: args.max_images]
    clean_metrics = read_json(verifier_metrics_path(config, spec.benchmark_id, spec.verifier_id))
    threshold = float(clean_metrics.get("binary_threshold", 0.5))
    if args.target_label == "real" and spec.class_names == ("real", "fake"):
        threshold = 1.0 - threshold
    eps = max(float(item) for item in config["attacks"]["epsilons"])
    budgets = [int(item) for item in config["random_search"]["query_budgets"]]
    summary_rows = []
    for budget in budgets:
        successes = []
        for record in source_records:
            clean = image_tensor(record.image_path).unsqueeze(0).to(device)
            best = clean.clone()
            success = 0
            best_score = -float("inf")
            for _ in range(budget):
                proposal = best + torch.empty_like(best).uniform_(-eps, eps)
                proposal = project_linf(clean, proposal, eps)
                with torch.no_grad():
                    probs = model.forward_probs(proposal)
                score = float(generic_score_from_probs(probs, spec.class_names, args.target_label).item())
                if score > best_score:
                    best = proposal
                    best_score = score
                if score >= threshold:
                    success = 1
                    break
            successes.append(success)
        summary_rows.append(
            {
                "benchmark_id": spec.benchmark_id,
                "verifier_id": spec.verifier_id,
                "interface_id": "continuous_target_score",
                "attack_family": "score_guided_random_search",
                "source": args.source_label,
                "target": args.target_label,
                "eps": eps,
                "pi": "",
                "asr": float(np.mean(successes)) if successes else 0.0,
                "robust_gap": "",
                "clean_target_acceptance": "",
                "adv_target_score_mean": "",
                "query_budget": budget,
                "emulator_eps95": "",
                "emulator_max_error": "",
                "recovered_whitebox_gap": "",
                "alpha_estimate": "",
                "optimizer": "random_search",
            }
        )
    summary_path = attack_summary_path(config, spec.benchmark_id, spec.verifier_id, "score_guided_random_search")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(summary_path.with_name("score_guided_random_search_summary.csv"), summary_rows, SUMMARY_COLUMNS)
    write_json(summary_path, {"summary_rows": summary_rows, "max_images": args.max_images})


if __name__ == "__main__":
    main()
