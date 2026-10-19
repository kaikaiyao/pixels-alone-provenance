from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageDraw

from passive_provenance.attacks.pgd import pgd_targeted, score_from_probs
from passive_provenance.config import load_config, resolve_artifact_path, resolve_artifact_relative
from passive_provenance.datasets import load_triplet_manifest
from passive_provenance.models.d1 import FrozenClipVerifier
from passive_provenance.stages.attack_d1 import image_tensor, targeted_success
from passive_provenance.utils.image_ops import tensor_to_pil
from passive_provenance.utils.metrics import summarize_attack_distances
from passive_provenance.utils.perceptual import ssim_per_image


@dataclass(frozen=True)
class AuditItem:
    image_id: str
    seed: int
    path: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paired perceptual audit for successful targeted provenance attacks."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--clean-metrics-path")
    parser.add_argument("--output-dir", default="artifacts/generated/perceptual_audit")
    parser.add_argument("--samples-per-direction", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--eps", action="append", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--restarts", type=int)
    parser.add_argument("--save-examples-per-cell", type=int, default=2)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(config: dict[str, object], override: str | None) -> tuple[FrozenClipVerifier, Path]:
    checkpoint_path = (
        resolve_artifact_relative(config, override)
        if override
        else resolve_artifact_path(config, "d1_checkpoint")
    )
    payload = torch.load(checkpoint_path, map_location="cpu")
    model = FrozenClipVerifier(
        clip_model_name=payload["clip_model_name"],
        clip_pretrained=payload["clip_pretrained"],
        input_size=int(payload["input_size"]),
        mean=tuple(payload["mean"]),
        std=tuple(payload["std"]),
        class_names=tuple(payload.get("class_names", ("real", "sd15", "sdxl"))),
    )
    model.head.load_state_dict(payload["head_state_dict"])
    return model, checkpoint_path


def make_lpips_model(device: torch.device) -> torch.nn.Module:
    import lpips  # type: ignore

    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model


def implementation_provenance(config_path: Path) -> dict[str, object]:
    package_root = Path(__file__).resolve().parents[1]
    code_paths = sorted(package_root.rglob("*.py")) + [config_path]
    missing = [str(path) for path in code_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing perceptual-audit implementation inputs: {missing}")

    import lpips  # type: ignore

    lpips_root = Path(lpips.__file__).resolve().parent
    weight_paths = sorted(
        {
            *lpips_root.glob("weights/**/*.pth"),
            *Path(torch.hub.get_dir()).glob("checkpoints/*alex*"),
        }
    )
    open_clip_cache = os.environ.get("PP_OPEN_CLIP_CACHE")
    encoder_weight_paths = (
        sorted(path for path in Path(open_clip_cache).glob("**/*") if path.is_file())
        if open_clip_cache
        else []
    )
    return {
        "code": {
            str(path): {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in code_paths
        },
        "lpips_package_version": importlib.metadata.version("lpips"),
        "lpips_weight_files": {
            str(path): {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in weight_paths
        },
        "open_clip_cache": open_clip_cache,
        "open_clip_encoder_files": {
            str(path): {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in encoder_weight_paths
        },
        "container_image": os.environ.get("PP_CONTAINER_IMAGE", "unknown"),
    }


def lpips_per_image(
    model: torch.nn.Module, original: torch.Tensor, adversarial: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        return model(original * 2.0 - 1.0, adversarial * 2.0 - 1.0).flatten()


def select_items(records: list[object], samples_per_direction: int, seed: int) -> dict[str, list[AuditItem]]:
    test_records = [record for record in records if record.split == "test"]
    rng = random.Random(seed)
    rng.shuffle(test_records)
    selected = test_records[: min(samples_per_direction, len(test_records))]
    human = [AuditItem(str(record.image_id), int(record.seed), record.real_path, "human") for record in selected]
    sd15 = [AuditItem(str(record.image_id), int(record.seed), record.sd15_path, "sd15") for record in selected]
    sdxl = [AuditItem(str(record.image_id), int(record.seed), record.sdxl_path, "sdxl") for record in selected]
    ai_to_real = [item for pair in zip(sd15[::2], sdxl[1::2]) for item in pair]
    if len(ai_to_real) < len(selected):
        ai_to_real.extend((sd15 + sdxl)[: len(selected) - len(ai_to_real)])
    return {
        "human_to_fake": human,
        "ai_to_real": ai_to_real[: len(selected)],
        "sd15_to_sdxl": sd15,
        "sdxl_to_sd15": sdxl,
    }


def task_target(task_name: str) -> str:
    return {
        "human_to_fake": "fake",
        "ai_to_real": "real",
        "sd15_to_sdxl": "sdxl",
        "sdxl_to_sd15": "sd15",
    }[task_name]


def source_initially_correct(
    task_name: str,
    clean_probs: torch.Tensor,
    binary_threshold: float,
) -> np.ndarray:
    """Return whether each clean source is correctly classified for this task."""
    if task_name == "human_to_fake":
        return (
            score_from_probs(clean_probs, "fake").detach().cpu().numpy() < binary_threshold
        ).astype(np.int32)
    if task_name == "ai_to_real":
        return (
            score_from_probs(clean_probs, "fake").detach().cpu().numpy() >= binary_threshold
        ).astype(np.int32)
    clean_argmax = clean_probs.argmax(dim=1).detach().cpu().numpy()
    if task_name == "sd15_to_sdxl":
        return (clean_argmax == 1).astype(np.int32)
    if task_name == "sdxl_to_sd15":
        return (clean_argmax == 2).astype(np.int32)
    raise ValueError(f"Unsupported task {task_name}")


def audit_outcome(row: dict[str, object]) -> str:
    initially_correct = int(row["initially_correct"]) == 1
    success = int(row["success"]) == 1
    if initially_correct:
        return "induced_success" if success else "attack_failure"
    return "preexisting_target" if success else "other_clean_error"


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def select_visual_examples(
    rows: list[dict[str, object]], limit: int
) -> list[dict[str, object]]:
    """Prefer induced successes and failures among initially correct sources."""
    if limit <= 0:
        return []
    groups = [
        [row for row in rows if audit_outcome(row) == "induced_success"],
        [row for row in rows if audit_outcome(row) == "attack_failure"],
    ]
    if not any(groups):
        groups = [
            [row for row in rows if audit_outcome(row) == "preexisting_target"],
            [row for row in rows if audit_outcome(row) == "other_clean_error"],
        ]
    selected: list[dict[str, object]] = []
    offsets = [0, 0]
    while len(selected) < limit:
        added = False
        for group_index, group in enumerate(groups):
            offset = offsets[group_index]
            if offset < len(group) and len(selected) < limit:
                selected.append(group[offset])
                offsets[group_index] += 1
                added = True
        if not added:
            break
    return selected


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("Cannot summarize an empty perceptual-audit cell.")

    def metric_block(selected: list[dict[str, object]]) -> dict[str, object]:
        return {
            "n": len(selected),
            "linf": quantiles([float(row["linf"]) for row in selected]),
            "psnr": quantiles([float(row["psnr"]) for row in selected]),
            "ssim": quantiles([float(row["ssim"]) for row in selected]),
            "lpips_alex": quantiles([float(row["lpips_alex"]) for row in selected]),
        }

    initially_correct = [row for row in rows if int(row["initially_correct"]) == 1]
    induced_successes = [row for row in rows if audit_outcome(row) == "induced_success"]
    attack_failures = [row for row in rows if audit_outcome(row) == "attack_failure"]
    preexisting_targets = [row for row in rows if audit_outcome(row) == "preexisting_target"]
    summary: dict[str, object] = {
        **metric_block(rows),
        # Keep all-source ASR for direct compatibility with the submitted tables.
        "asr": float(np.mean([row["success"] for row in rows])),
        "asr_population": "all_attacked_sources",
        "initially_correct_n": len(initially_correct),
        "asr_initially_correct": (
            float(np.mean([row["success"] for row in initially_correct]))
            if initially_correct
            else None
        ),
        "induced_success_n": len(induced_successes),
        "attack_failure_n": len(attack_failures),
        "preexisting_target_n": len(preexisting_targets),
    }
    if induced_successes:
        summary["induced_success"] = metric_block(induced_successes)
    if attack_failures:
        summary["attack_failure"] = metric_block(attack_failures)
    return summary


def save_triplet(
    output_dir: Path,
    *,
    cell: str,
    rank: int,
    original: torch.Tensor,
    adversarial: torch.Tensor,
) -> tuple[Path, Path, Path]:
    cell_dir = output_dir / "examples" / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    clean_path = cell_dir / f"{rank:02d}_clean.png"
    adv_path = cell_dir / f"{rank:02d}_adversarial.png"
    delta_path = cell_dir / f"{rank:02d}_delta_8x.png"
    tensor_to_pil(original).save(clean_path)
    tensor_to_pil(adversarial).save(adv_path)
    amplified = (0.5 + 8.0 * (adversarial - original)).clamp(0.0, 1.0)
    tensor_to_pil(amplified).save(delta_path)
    return clean_path, adv_path, delta_path


def build_montage(output_dir: Path, saved_examples: list[dict[str, object]]) -> Path | None:
    if not saved_examples:
        return None
    tile = 256
    label_height = 34
    rows = []
    for example in saved_examples:
        images = [Image.open(str(example[key])).convert("RGB").resize((tile, tile)) for key in ("clean", "adv", "delta")]
        row = Image.new("RGB", (tile * 3, tile + label_height), "white")
        draw = ImageDraw.Draw(row)
        draw.text((6, 8), str(example["label"]), fill="black")
        for index, image in enumerate(images):
            row.paste(image, (index * tile, label_height))
        rows.append(row)
    montage = Image.new("RGB", (tile * 3, (tile + label_height) * len(rows)), "white")
    for index, row in enumerate(rows):
        montage.paste(row, (0, index * (tile + label_height)))
    path = output_dir / "attack_examples_montage.png"
    montage.save(path)
    return path


def run_batches(
    items: list[AuditItem],
    *,
    batch_size: int,
    operation: Callable[[list[AuditItem]], list[dict[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for start in range(0, len(items), batch_size):
        rows.extend(operation(items[start : start + batch_size]))
    return rows


def main() -> None:
    args = parse_args()
    if args.samples_per_direction <= 0 or args.batch_size <= 0:
        raise ValueError("Sample and batch sizes must be positive.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("The perceptual audit requires a real GPU and must run on EIDF.")
    device = torch.device("cuda:0")
    model, checkpoint_path = load_checkpoint(config, args.checkpoint_path)
    model = model.to(device).eval()
    lpips_model = make_lpips_model(device)
    manifest_path = resolve_artifact_path(config, "dataset_manifest")
    records = load_triplet_manifest(manifest_path)
    selected = select_items(records, args.samples_per_direction, args.seed)
    clean_metrics_path = (
        resolve_artifact_relative(config, args.clean_metrics_path)
        if args.clean_metrics_path
        else resolve_artifact_path(config, "clean_metrics")
    )
    with clean_metrics_path.open("r", encoding="utf-8") as handle:
        clean_metrics = json.load(handle)
    binary_threshold = float(clean_metrics.get("binary_threshold", 0.5))
    epsilons = args.eps or [float(config["attacks"]["epsilons"][0]), float(config["attacks"]["epsilons"][-1])]
    steps = args.steps or int(config["attacks"]["steps"])
    restarts = args.restarts or int(config["attacks"]["restarts"])
    output_dir = resolve_artifact_relative(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    saved_examples: list[dict[str, object]] = []

    for task_name, items in selected.items():
        target = task_target(task_name)
        for eps in epsilons:
            print(f"[perceptual-audit] task={task_name} eps={eps:.8f} n={len(items)} starting", flush=True)

            def operation(batch_items: list[AuditItem]) -> list[dict[str, object]]:
                originals = torch.stack([image_tensor(item.path) for item in batch_items]).to(device)
                with torch.no_grad():
                    clean_probs = model.forward_probs(originals)
                attack = pgd_targeted(
                    model.forward_probs,
                    originals,
                    eps=eps,
                    steps=steps,
                    step_size=eps / 4.0,
                    restarts=restarts,
                    target=target,
                )
                with torch.no_grad():
                    adv_probs = model.forward_probs(attack.adversarial)
                successes = targeted_success(
                    target_group=target,
                    adv_probs=adv_probs,
                    clean_binary_threshold=binary_threshold,
                )
                initially_correct = source_initially_correct(
                    task_name,
                    clean_probs,
                    binary_threshold,
                )
                clean_target_scores = score_from_probs(clean_probs, target).detach().cpu().numpy()
                adversarial_target_scores = score_from_probs(adv_probs, target).detach().cpu().numpy()
                clean_predictions = clean_probs.argmax(dim=1).detach().cpu().numpy()
                adversarial_predictions = adv_probs.argmax(dim=1).detach().cpu().numpy()
                distances = summarize_attack_distances(originals, attack.adversarial)
                ssim = ssim_per_image(originals, attack.adversarial)
                lpips = lpips_per_image(lpips_model, originals, attack.adversarial)
                batch_rows = []
                for index, item in enumerate(batch_items):
                    batch_rows.append(
                        {
                            "task": task_name,
                            "source": item.source,
                            "target": target,
                            "image_id": item.image_id,
                            "seed": item.seed,
                            "eps": eps,
                            "clean_prediction": int(clean_predictions[index]),
                            "adversarial_prediction": int(adversarial_predictions[index]),
                            "clean_target_score": float(clean_target_scores[index]),
                            "adversarial_target_score": float(adversarial_target_scores[index]),
                            "initially_correct": int(initially_correct[index]),
                            "success": int(successes[index]),
                            "success_on_initially_correct": int(
                                successes[index] and initially_correct[index]
                            ),
                            "linf": float(distances["linf"][index].item()),
                            "psnr": float(distances["psnr"][index].item()),
                            "ssim": float(ssim[index].item()),
                            "lpips_alex": float(lpips[index].item()),
                            "_clean": originals[index].detach().cpu(),
                            "_adv": attack.adversarial[index].detach().cpu(),
                        }
                    )
                return batch_rows

            cell_rows = run_batches(items, batch_size=args.batch_size, operation=operation)
            examples = select_visual_examples(cell_rows, args.save_examples_per_cell)
            for rank, row in enumerate(examples):
                outcome = audit_outcome(row)
                clean_path, adv_path, delta_path = save_triplet(
                    output_dir,
                    cell=f"{task_name}_{outcome}_eps_{eps:.8f}".replace(".", "p"),
                    rank=rank,
                    original=row.pop("_clean"),
                    adversarial=row.pop("_adv"),
                )
                saved_examples.append(
                    {
                        "label": f"{task_name}, {outcome}, eps={eps:.5f}",
                        "clean": clean_path,
                        "adv": adv_path,
                        "delta": delta_path,
                    }
                )
            for row in cell_rows:
                row.pop("_clean", None)
                row.pop("_adv", None)
            rows.extend(cell_rows)
            print(
                f"[perceptual-audit] task={task_name} eps={eps:.8f} "
                f"asr_all={np.mean([row['success'] for row in cell_rows]):.4f} "
                f"asr_initially_correct={summarize_rows(cell_rows)['asr_initially_correct']} "
                f"median_ssim={np.median([row['ssim'] for row in cell_rows]):.6f} "
                f"median_lpips={np.median([row['lpips_alex'] for row in cell_rows]):.6f}",
                flush=True,
            )

    row_path = output_dir / "per_image_metrics.csv"
    fieldnames = [
        "task",
        "source",
        "target",
        "image_id",
        "seed",
        "eps",
        "clean_prediction",
        "adversarial_prediction",
        "clean_target_score",
        "adversarial_target_score",
        "initially_correct",
        "success",
        "success_on_initially_correct",
        "linf",
        "psnr",
        "ssim",
        "lpips_alex",
    ]
    with row_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {"cells": {}, "overall_by_epsilon": {}}
    for task_name in selected:
        for eps in epsilons:
            cell = [row for row in rows if row["task"] == task_name and float(row["eps"]) == eps]
            key = f"{task_name}|{eps:.8f}"
            summary["cells"][key] = summarize_rows(cell)
            summary["cells"][key]["linf_violation_count"] = int(
                sum(float(row["linf"]) > eps + 1e-6 for row in cell)
            )
    for eps in epsilons:
        cell = [row for row in rows if float(row["eps"]) == eps]
        summary["overall_by_epsilon"][f"{eps:.8f}"] = summarize_rows(cell)
        summary["overall_by_epsilon"][f"{eps:.8f}"]["linf_violation_count"] = int(
            sum(float(row["linf"]) > eps + 1e-6 for row in cell)
        )

    montage_path = build_montage(output_dir, saved_examples)
    metadata = {
        "seed": args.seed,
        "samples_per_direction": args.samples_per_direction,
        "batch_size": args.batch_size,
        "epsilons": epsilons,
        "steps": steps,
        "restarts": restarts,
        "step_size_rule": "epsilon/4",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "clean_metrics": str(clean_metrics_path),
        "clean_metrics_sha256": file_sha256(clean_metrics_path),
        "implementation": implementation_provenance(config_path),
        "selected_ids": {key: [item.image_id for item in value] for key, value in selected.items()},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "montage": str(montage_path) if montage_path else None,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[perceptual-audit] complete output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
