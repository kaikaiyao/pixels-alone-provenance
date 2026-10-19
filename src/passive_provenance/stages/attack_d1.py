from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from passive_provenance.attacks.pgd import pgd_targeted, score_from_probs
from passive_provenance.config import load_config, resolve_artifact_path, resolve_artifact_relative
from passive_provenance.constants import RESULT_COLUMNS
from passive_provenance.datasets import load_triplet_manifest
from passive_provenance.models.d1 import FrozenClipVerifier
from passive_provenance.utils.io import read_json, write_csv, write_json
from passive_provenance.utils.metrics import robust_gap
from passive_provenance.utils.metrics import compute_lpips_batch, summarize_attack_distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E1 and E2 attacks on the D1 verifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--subset", choices=["full", "smoke"], default="smoke")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--clean-metrics-path")
    parser.add_argument("--attack-results-path")
    parser.add_argument("--attack-summary-path")
    parser.add_argument("--detector-name", default="d1_clip3")
    return parser.parse_args()


def output_path(config: dict[str, object], override: str | None, default_key: str) -> Path:
    if override:
        return resolve_artifact_relative(config, override)
    return resolve_artifact_path(config, default_key)


def load_checkpoint(config: dict[str, object], checkpoint_override: str | None = None) -> FrozenClipVerifier:
    checkpoint_path = output_path(config, checkpoint_override, "d1_checkpoint")
    payload = torch.load(checkpoint_path, map_location="cpu")
    model = FrozenClipVerifier(
        clip_model_name=payload["clip_model_name"],
        clip_pretrained=payload["clip_pretrained"],
        input_size=int(payload["input_size"]),
        mean=tuple(payload["mean"]),
        std=tuple(payload["std"]),
    )
    model.head.load_state_dict(payload["head_state_dict"])
    return model


def image_tensor(path: str) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def run_attack_batch(model, images, target, eps, steps, restarts, pi=None):
    step_size = eps / 4.0
    return pgd_targeted(model.forward_probs, images, eps=eps, steps=steps, step_size=step_size, restarts=restarts, target=target, pi=pi)


def targeted_success(
    *,
    target_group: str,
    adv_probs: torch.Tensor,
    clean_binary_threshold: float,
) -> np.ndarray:
    if target_group == "fake":
        return (score_from_probs(adv_probs, "fake").detach().cpu().numpy() >= clean_binary_threshold).astype(np.int32)
    if target_group == "real":
        return (score_from_probs(adv_probs, "fake").detach().cpu().numpy() < clean_binary_threshold).astype(np.int32)
    if target_group == "sd15":
        return (adv_probs.argmax(dim=1).detach().cpu().numpy() == 1).astype(np.int32)
    if target_group == "sdxl":
        return (adv_probs.argmax(dim=1).detach().cpu().numpy() == 2).astype(np.int32)
    raise ValueError(f"Unsupported target group {target_group}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest_key = "smoke_manifest" if args.subset == "smoke" else "dataset_manifest"
    records = load_triplet_manifest(resolve_artifact_path(config, manifest_key))
    compute_lpips = args.subset == "smoke"
    model = load_checkpoint(config, args.checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if not compute_lpips:
        print(
            "[attack-d1] skipping LPIPS for the full subset; use the perceptual_audit stage for paired perceptual metrics",
            flush=True,
        )

    task_specs = [
        ("human", "fake", lambda record: [(record.real_path, "human", record.image_id, record.seed)]),
        ("ai", "real", lambda record: [(record.sd15_path, "sd15", record.image_id, record.seed), (record.sdxl_path, "sdxl", record.image_id, record.seed)]),
        ("sd15", "sdxl", lambda record: [(record.sd15_path, "sd15", record.image_id, record.seed)]),
        ("sdxl", "sd15", lambda record: [(record.sdxl_path, "sdxl", record.image_id, record.seed)]),
    ]
    rows = []
    summary: dict[str, dict[str, float | dict[str, float]]] = {}
    epsilons = [float(item) for item in config["attacks"]["epsilons"]]
    clean_metrics_path = output_path(config, args.clean_metrics_path, "clean_metrics")
    clean_metrics = read_json(clean_metrics_path) if clean_metrics_path.exists() else {}
    clean_binary_threshold = float(clean_metrics.get("binary_threshold", 0.5))
    for source_group, target_group, selector in task_specs:
        selected = []
        for record in records:
            if record.split != "test":
                continue
            selected.extend(selector(record))
        if not selected:
            continue
        clean_images = torch.stack([image_tensor(path) for path, _, _, _ in selected]).to(device)
        if target_group == "fake":
            target_clean_images = torch.stack(
                [image_tensor(path) for record in records if record.split == "test" for path in (record.sd15_path, record.sdxl_path)]
            ).to(device)
        elif target_group == "real":
            target_clean_images = torch.stack([image_tensor(record.real_path) for record in records if record.split == "test"]).to(device)
        elif target_group == "sdxl":
            target_clean_images = torch.stack([image_tensor(record.sdxl_path) for record in records if record.split == "test"]).to(device)
        elif target_group == "sd15":
            target_clean_images = torch.stack([image_tensor(record.sd15_path) for record in records if record.split == "test"]).to(device)
        else:
            raise ValueError(target_group)
        with torch.no_grad():
            clean_probs = model.forward_probs(clean_images)
            target_clean_probs = model.forward_probs(target_clean_images)
        clean_target = score_from_probs(clean_probs, target_group).detach().cpu().numpy()
        target_clean_scores = score_from_probs(target_clean_probs, target_group).detach().cpu().numpy()
        clean_pred = clean_probs.argmax(dim=1).detach().cpu().numpy()
        task_name = f"{source_group}_to_{target_group}"
        summary[task_name] = {}
        for eps in epsilons:
            print(
                f"[attack-d1] task={task_name} eps={eps:.8f} images={len(selected)} starting",
                flush=True,
            )
            attack_result = run_attack_batch(
                model,
                clean_images,
                target=target_group,
                eps=eps,
                steps=int(config["attacks"]["steps"]),
                restarts=int(config["attacks"]["restarts"]),
            )
            with torch.no_grad():
                adv_probs = model.forward_probs(attack_result.adversarial)
            adv_target = score_from_probs(adv_probs, target_group).detach().cpu().numpy()
            adv_pred = adv_probs.argmax(dim=1).detach().cpu().numpy()
            distance = summarize_attack_distances(clean_images, attack_result.adversarial)
            lpips_values = None
            if compute_lpips:
                lpips_values = compute_lpips_batch(
                    clean_images,
                    attack_result.adversarial,
                    batch_size=int(config["attacks"].get("batch_size", 8)),
                    device="cpu",
                ).detach().cpu().numpy()
            success = targeted_success(
                target_group=target_group,
                adv_probs=adv_probs,
                clean_binary_threshold=clean_binary_threshold,
            )
            summary[task_name][str(eps)] = {
                "asr": float(success.mean()),
                "clean_target_acceptance": float(target_clean_scores.mean()),
                "adv_target_score_mean": float(adv_target.mean()),
                "robust_gap": robust_gap(float(target_clean_scores.mean()), adv_target),
            }
            print(
                "[attack-d1] "
                f"task={task_name} eps={eps:.8f} "
                f"asr={summary[task_name][str(eps)]['asr']:.4f} "
                f"robust_gap={summary[task_name][str(eps)]['robust_gap']:.6f}",
                flush=True,
            )
            for idx, (_, source_name, image_id, seed) in enumerate(selected):
                rows.append(
                    {
                        "image_id": image_id,
                        "split": "test",
                        "source": source_name,
                        "target": target_group,
                        "detector": args.detector_name,
                        "attack": "pgd_targeted",
                        "eps": eps,
                        "pi": "",
                        "seed": seed,
                        "clean_target_score": float(clean_target[idx]),
                        "adv_target_score": float(adv_target[idx]),
                        "clean_pred": int(clean_pred[idx]),
                        "adv_pred": int(adv_pred[idx]),
                        "success": int(success[idx]),
                        "linf": float(distance["linf"][idx].item()),
                        "psnr": float(distance["psnr"][idx].item()),
                        "lpips": "" if lpips_values is None else float(lpips_values[idx]),
                        "emulator_eps95": "",
                        "alpha_estimate": "",
                        "query_count": "",
                    }
                )

    mixture_pis = [float(item) for item in config["attacks"]["mixture_pis"]]
    human_records = [record for record in records if record.split == "test"]
    human_images = torch.stack([image_tensor(record.real_path) for record in human_records]).to(device)
    with torch.no_grad():
        human_clean_probs = model.forward_probs(human_images)
    for pi in mixture_pis:
        for attack_name, target_name in [("oracle-pi", "mixture"), ("coarse-only", "fake")]:
            attack_result = run_attack_batch(
                model,
                human_images,
                target=target_name,
                eps=max(epsilons),
                steps=int(config["attacks"]["steps"]),
                restarts=int(config["attacks"]["restarts"]),
                pi=pi if target_name == "mixture" else None,
            )
            with torch.no_grad():
                adv_probs = model.forward_probs(attack_result.adversarial)
            clean_target = score_from_probs(human_clean_probs, target_name, pi if target_name == "mixture" else None).detach().cpu().numpy()
            adv_target = score_from_probs(adv_probs, target_name, pi if target_name == "mixture" else None).detach().cpu().numpy()
            adv_pred = adv_probs.argmax(dim=1).detach().cpu().numpy()
            success = (adv_target > clean_target).astype(np.int32)
            for idx, record in enumerate(human_records):
                rows.append(
                    {
                        "image_id": record.image_id,
                        "split": "test",
                        "source": "human",
                        "target": "mixture" if target_name == "mixture" else "fake",
                        "detector": args.detector_name,
                        "attack": attack_name,
                        "eps": max(epsilons),
                        "pi": pi,
                        "seed": record.seed,
                        "clean_target_score": float(clean_target[idx]),
                        "adv_target_score": float(adv_target[idx]),
                        "clean_pred": int(human_clean_probs[idx].argmax().item()),
                        "adv_pred": int(adv_pred[idx]),
                        "success": int(success[idx]),
                        "linf": float((attack_result.adversarial[idx] - human_images[idx]).abs().max().item()),
                        "psnr": "",
                        "lpips": "",
                        "emulator_eps95": "",
                        "alpha_estimate": "",
                        "query_count": "",
                    }
                )

    write_csv(output_path(config, args.attack_results_path, "attack_results"), rows, RESULT_COLUMNS)
    write_json(
        output_path(config, args.attack_summary_path, "attack_summary"),
        {"clean_metrics": clean_metrics, "tasks": summary, "detector": args.detector_name},
    )


if __name__ == "__main__":
    main()
