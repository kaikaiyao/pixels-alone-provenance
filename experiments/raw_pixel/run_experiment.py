from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)
from torchvision.transforms import functional as tvf


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
FAKE_GENERATORS = ("sd15", "sdxl", "sd21", "pixart")
FULL_MANIFEST_NAME = "sample_manifest_full.csv"
FULL_SCIENTIFIC_HYPERPARAMETERS: dict[str, int | float | str] = {
    "arch": "resnet18",
    "seed": 20260723,
    "epochs": 10,
    "batch_size": 32,
    "eval_batch_size": 64,
    "backbone_lr": 1e-4,
    "head_lr": 1e-3,
    "weight_decay": 1e-4,
    "clean_auc_gate": 0.85,
    "clip_model": "ViT-B-32",
    "clip_pretrained": "openai",
    "query_variants": 2,
    "query_eps": 8.0 / 255.0,
    "query_batch_size": 16,
    "surrogate_hidden_dim": 256,
    "surrogate_epochs": 100,
    "surrogate_lr": 1e-3,
    "attack_eps": 8.0 / 255.0,
    "attack_steps": 40,
    "attack_restarts": 5,
    "attack_batch_size": 4,
    "attack_samples_per_direction": 128,
}


@dataclass(frozen=True)
class ManifestRow:
    benchmark_id: str
    prompt_id: int
    image_id: int
    split: str
    generator_id: str
    seed: int
    image_path: str

    @property
    def binary_label(self) -> int:
        return int(self.generator_id != "real")


class BinaryManifestDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, rows: Sequence[ManifestRow], transform: transforms.Compose) -> None:
        self.rows = list(rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, row.binary_label


class ScoreMLP(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a raw-pixel ImageNet ResNet victim on a prompt-disjoint image benchmark, "
            "fit a CLIP-feature MLP from victim score queries, and run targeted Linf transfer attacks."
        )
    )
    parser.add_argument("--stage", choices=("train-victim", "surrogate-attack", "all"), default="all")
    parser.add_argument("--subset", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--arch", choices=("resnet18",), default="resnet18")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--image-root",
        help="Base directory for relative image_path entries (default: the manifest directory).",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clean-auc-gate", type=float, default=0.85)
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--query-variants", type=int, default=2)
    parser.add_argument("--query-eps", type=float, default=8.0 / 255.0)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--surrogate-hidden-dim", type=int, default=256)
    parser.add_argument("--surrogate-epochs", type=int, default=60)
    parser.add_argument("--surrogate-lr", type=float, default=1e-3)
    parser.add_argument("--attack-eps", type=float, default=8.0 / 255.0)
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--attack-restarts", type=int, default=1)
    parser.add_argument("--attack-batch-size", type=int, default=4)
    parser.add_argument("--attack-samples-per-direction", type=int, default=64)
    parser.add_argument("--save-examples-per-direction", type=int, default=4)
    parser.add_argument("--container-image", default="unknown")
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_manifest(path: Path, image_root: Path | None = None) -> list[ManifestRow]:
    root = (image_root if image_root is not None else path.parent).expanduser().resolve()

    def resolve_image_path(value: str) -> str:
        image_path = Path(value).expanduser()
        return str(image_path if image_path.is_absolute() else root / image_path)

    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = [
        ManifestRow(
            benchmark_id=row["benchmark_id"],
            prompt_id=int(row["prompt_id"]),
            image_id=int(row["image_id"]),
            split=row["split"],
            generator_id=row["generator_id"],
            seed=int(row["seed"]),
            image_path=resolve_image_path(row["image_path"]),
        )
        for row in raw_rows
    ]
    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")
    return rows


def validate_manifest(rows: Sequence[ManifestRow]) -> dict[str, object]:
    split_prompts: dict[str, set[int]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    generator_counts: Counter[str] = Counter()
    missing: list[str] = []
    for row in rows:
        split_prompts[row.split].add(row.prompt_id)
        split_counts[row.split] += 1
        generator_counts[row.generator_id] += 1
        if not Path(row.image_path).is_file():
            missing.append(row.image_path)
    overlap = {
        "train_val": len(split_prompts["train"] & split_prompts["val"]),
        "train_test": len(split_prompts["train"] & split_prompts["test"]),
        "val_test": len(split_prompts["val"] & split_prompts["test"]),
    }
    unexpected_generators = sorted(set(generator_counts) - {"real", *FAKE_GENERATORS})
    if missing:
        raise FileNotFoundError(f"Manifest has {len(missing)} missing images; examples={missing[:5]}")
    if any(overlap.values()):
        raise RuntimeError(f"Prompt leakage detected across splits: {overlap}")
    if unexpected_generators:
        raise RuntimeError(f"Unexpected B1 generator IDs: {unexpected_generators}")
    return {
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "generator_counts": dict(sorted(generator_counts.items())),
        "prompt_counts": {key: len(value) for key, value in sorted(split_prompts.items())},
        "prompt_overlap": overlap,
        "missing_image_paths": 0,
    }


def split_rows(rows: Sequence[ManifestRow], split: str) -> list[ManifestRow]:
    selected = [row for row in rows if row.split == split]
    if not selected:
        raise RuntimeError(f"No rows for split={split}")
    return selected


def training_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_victim(arch: str, *, pretrained: bool) -> tuple[nn.Module, str]:
    if arch == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    else:
        raise ValueError(arch)
    weight_name = weights.meta.get("recipe", "none") if weights is not None else "none"
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model, str(weight_name)


def loader_for(
    rows: Sequence[ManifestRow],
    *,
    train: bool,
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = BinaryManifestDataset(rows, training_transform() if train else evaluation_transform())
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
        drop_last=False,
    )


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    index = int(np.argmax(np.where(finite, tpr - fpr, -np.inf)))
    return float(thresholds[index])


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    selected_threshold = select_threshold(labels, scores) if threshold is None else float(threshold)
    predictions = (scores >= selected_threshold).astype(np.int32)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "threshold": selected_threshold,
        "mean_real_score": float(scores[labels == 0].mean()),
        "mean_fake_score": float(scores[labels == 1].mean()),
        "samples": int(labels.size),
    }


def clean_evidence_decision(
    *,
    val_auc: float,
    test_auc: float,
    threshold: float,
) -> dict[str, object]:
    """Separate the validation-only execution gate from final claim eligibility."""
    validation_pass = float(val_auc) >= float(threshold)
    claim_eligible = validation_pass and float(test_auc) >= float(threshold)
    return {
        "auc_gate": float(threshold),
        "attack_execution_gate": "validation_auc",
        "attack_execution_gate_value": float(val_auc),
        "attack_execution_gate_pass": validation_pass,
        # Retained as a compatibility alias; it now means the validation-only execution gate.
        "gate_pass": validation_pass,
        "test_auc_reporting_only": True,
        "claim_eligible": claim_eligible,
        "claim_eligibility_rule": "validation_auc >= gate and test_auc >= gate",
        "fixed_run_decision_rule": (
            "If claim_eligible is false, report the result as non-supporting; do not switch architecture "
            "or tune and rerun based on the test result."
        ),
    }


def validate_full_scientific_contract(args: argparse.Namespace, manifest: Path) -> None:
    """Fail closed if the final full run drifts from its predeclared setup."""
    if args.subset != "full":
        return
    mismatches: list[str] = []
    if manifest.name != FULL_MANIFEST_NAME:
        mismatches.append(f"manifest.name={manifest.name!r} expected={FULL_MANIFEST_NAME!r}")
    for name, expected in FULL_SCIENTIFIC_HYPERPARAMETERS.items():
        actual = getattr(args, name)
        if isinstance(expected, float):
            matches = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        else:
            matches = actual == expected
        if not matches:
            mismatches.append(f"{name}={actual!r} expected={expected!r}")
    if mismatches:
        raise RuntimeError("FULL_SCIENTIFIC_CONTRACT_MISMATCH: " + "; ".join(mismatches))


def validate_victim_provenance(
    *,
    checkpoint_payload: dict[str, object],
    clean_metrics: dict[str, object],
    args: argparse.Namespace,
    manifest_sha256: str,
    checkpoint_sha256: str,
) -> None:
    """Reject staged surrogate runs whose victim artifacts do not match the current contract."""
    expected = {
        "architecture": str(args.arch),
        "seed": int(args.seed),
        "manifest_sha256": manifest_sha256,
    }
    mismatches: list[str] = []
    for name, expected_value in expected.items():
        checkpoint_value = checkpoint_payload.get(name)
        metrics_value = clean_metrics.get(name)
        if checkpoint_value != expected_value:
            mismatches.append(
                f"checkpoint.{name}={checkpoint_value!r} expected={expected_value!r}"
            )
        if metrics_value != expected_value:
            mismatches.append(f"clean_metrics.{name}={metrics_value!r} expected={expected_value!r}")
    metrics_checkpoint_sha256 = clean_metrics.get("checkpoint_sha256")
    if metrics_checkpoint_sha256 != checkpoint_sha256:
        mismatches.append(
            "clean_metrics.checkpoint_sha256="
            f"{metrics_checkpoint_sha256!r} actual={checkpoint_sha256!r}"
        )
    if checkpoint_payload.get("victim_family") != "raw_pixel_imagenet_resnet":
        mismatches.append(
            f"checkpoint.victim_family={checkpoint_payload.get('victim_family')!r} "
            "expected='raw_pixel_imagenet_resnet'"
        )
    if clean_metrics.get("victim_family") != "raw_pixel_imagenet_resnet":
        mismatches.append(
            f"clean_metrics.victim_family={clean_metrics.get('victim_family')!r} "
            "expected='raw_pixel_imagenet_resnet'"
        )
    if mismatches:
        raise RuntimeError("VICTIM_PROVENANCE_MISMATCH: " + "; ".join(mismatches))


@torch.inference_mode()
def predict_loader(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, batch_labels in loader:
        logits = model(images.to(device, non_blocking=True))
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        scores.append(probabilities.detach().cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(labels), np.concatenate(scores)


def checkpoint_payload(
    model: nn.Module,
    *,
    args: argparse.Namespace,
    threshold: float,
    manifest_sha256: str,
    weight_name: str,
) -> dict[str, object]:
    return {
        "architecture": args.arch,
        "victim_family": "raw_pixel_imagenet_resnet",
        "input": "RGB pixels with ImageNet normalization",
        "class_names": ("real", "fake"),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "threshold": float(threshold),
        "manifest_sha256": manifest_sha256,
        "pretrained_weight_recipe": weight_name,
        "seed": int(args.seed),
        "created_at_utc": utc_now(),
    }


def train_victim(
    args: argparse.Namespace,
    rows: Sequence[ManifestRow],
    run_dir: Path,
    device: torch.device,
    manifest_sha256: str,
) -> dict[str, object]:
    train_rows = split_rows(rows, "train")
    val_rows = split_rows(rows, "val")
    test_rows = split_rows(rows, "test")
    train_loader = loader_for(
        train_rows,
        train=True,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
    )
    val_loader = loader_for(
        val_rows,
        train=False,
        batch_size=args.eval_batch_size,
        workers=args.workers,
        seed=args.seed + 1,
    )
    test_loader = loader_for(
        test_rows,
        train=False,
        batch_size=args.eval_batch_size,
        workers=args.workers,
        seed=args.seed + 2,
    )
    log(f"victim_init arch={args.arch} pretrained=ImageNet device={device} train_rows={len(train_rows)}")
    model, weight_name = build_victim(args.arch, pretrained=True)
    model = model.to(device)
    train_counts = Counter(row.binary_label for row in train_rows)
    class_weights = torch.tensor(
        [len(train_rows) / (2.0 * train_counts[index]) for index in (0, 1)],
        device=device,
        dtype=torch.float32,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    head_parameters = list(model.fc.parameters())
    head_parameter_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_parameter_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_auc = -math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []
    training_start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_start = time.monotonic()
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * images.shape[0]
            seen += images.shape[0]
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(train_loader):
                log(
                    f"victim_train epoch={epoch}/{args.epochs} batch={batch_index}/{len(train_loader)} "
                    f"loss={running_loss / max(1, seen):.6f} samples={seen}/{len(train_rows)}"
                )
        val_labels, val_scores = predict_loader(model, val_loader, device)
        val_metrics = binary_metrics(val_labels, val_scores)
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "val": val_metrics,
            "elapsed_seconds": time.monotonic() - epoch_start,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        }
        history.append(epoch_record)
        log(
            f"victim_epoch_complete epoch={epoch}/{args.epochs} train_loss={epoch_record['train_loss']:.6f} "
            f"val_auc={val_metrics['auc']:.6f} val_bal_acc={val_metrics['balanced_accuracy']:.6f} "
            f"epoch_seconds={epoch_record['elapsed_seconds']:.1f}"
        )
        if val_metrics["auc"] > best_auc:
            best_auc = float(val_metrics["auc"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        scheduler.step()
    if best_state is None:
        raise RuntimeError("Victim training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    val_labels, val_scores = predict_loader(model, val_loader, device)
    val_metrics = binary_metrics(val_labels, val_scores)
    test_labels, test_scores = predict_loader(model, test_loader, device)
    test_metrics = binary_metrics(test_labels, test_scores, threshold=val_metrics["threshold"])
    checkpoint = checkpoint_payload(
        model,
        args=args,
        threshold=float(val_metrics["threshold"]),
        manifest_sha256=manifest_sha256,
        weight_name=weight_name,
    )
    checkpoint_path = run_dir / "victim_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    evidence_decision = clean_evidence_decision(
        val_auc=float(val_metrics["auc"]),
        test_auc=float(test_metrics["auc"]),
        threshold=float(args.clean_auc_gate),
    )
    metrics: dict[str, object] = {
        "victim_family": "raw_pixel_imagenet_resnet",
        "architecture": args.arch,
        "seed": int(args.seed),
        "manifest_sha256": manifest_sha256,
        "pretrained_weight_recipe": weight_name,
        "best_epoch": best_epoch,
        "val": val_metrics,
        "test": test_metrics,
        **evidence_decision,
        "history": history,
        "total_training_seconds": time.monotonic() - training_start,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    atomic_write_json(run_dir / "clean_metrics.json", metrics)
    log(
        f"victim_complete best_epoch={best_epoch} val_auc={val_metrics['auc']:.6f} "
        f"test_auc={test_metrics['auc']:.6f} attack_gate_pass={metrics['attack_execution_gate_pass']} "
        f"claim_eligible={metrics['claim_eligible']} checkpoint={checkpoint_path}"
    )
    return metrics


def load_victim(
    run_dir: Path,
    device: torch.device,
    *,
    args: argparse.Namespace,
    manifest_sha256: str,
) -> tuple[nn.Module, dict[str, object]]:
    checkpoint_path = run_dir / "victim_checkpoint.pt"
    metrics_path = run_dir / "clean_metrics.json"
    if not checkpoint_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Missing victim artifacts under {run_dir}")
    checkpoint_sha256 = file_sha256(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validate_victim_provenance(
        checkpoint_payload=payload,
        clean_metrics=metrics,
        args=args,
        manifest_sha256=manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    model, _ = build_victim(str(payload["architecture"]), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, metrics


def normalize_eval_batch(
    images: torch.Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
        raise ValueError(f"Expected evaluation-domain tensors [B,3,224,224], got {tuple(images.shape)}")
    mean_tensor = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean_tensor) / std_tensor


def victim_fake_scores(model: nn.Module, raw_images: torch.Tensor) -> torch.Tensor:
    processed = normalize_eval_batch(
        raw_images,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    return torch.softmax(model(processed), dim=1)[:, 1]


def target_acceptance_scores(
    fake_scores: torch.Tensor,
    target_label: int,
) -> torch.Tensor:
    """Convert victim fake probabilities into acceptance for the requested target class."""
    if target_label not in (0, 1):
        raise ValueError(f"target_label must be 0 or 1, got {target_label}")
    return fake_scores if target_label == 1 else 1.0 - fake_scores


def load_raw_image(path: str) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        resized = tvf.resize(rgb, 256, interpolation=transforms.InterpolationMode.BICUBIC)
        cropped = tvf.center_crop(resized, [224, 224])
        return tvf.to_tensor(cropped)


def load_raw_batch(rows: Sequence[ManifestRow], device: torch.device) -> torch.Tensor:
    tensors = [load_raw_image(row.image_path) for row in rows]
    batch = torch.stack(tensors)
    if tuple(batch.shape[1:]) != (3, 224, 224):
        raise RuntimeError(f"Evaluation-domain batch has unexpected shape {tuple(batch.shape)}")
    return batch.to(device, non_blocking=True)


@torch.inference_mode()
def clean_target_distribution_metrics(
    *,
    rows: Sequence[ManifestRow],
    victim: nn.Module,
    target_label: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    """Measure target-class acceptance on the full clean test target distribution."""
    target_rows = [row for row in split_rows(rows, "test") if row.binary_label == target_label]
    if not target_rows:
        raise RuntimeError(f"No test rows found for target_label={target_label}")
    target_scores: list[torch.Tensor] = []
    total_batches = math.ceil(len(target_rows) / batch_size)
    for batch_index, start in enumerate(range(0, len(target_rows), batch_size), start=1):
        batch_rows = target_rows[start : start + batch_size]
        raw_images = load_raw_batch(batch_rows, device)
        fake_scores = victim_fake_scores(victim, raw_images)
        target_scores.append(target_acceptance_scores(fake_scores, target_label).detach().cpu())
        if batch_index == 1 or batch_index == total_batches:
            log(
                f"clean_target_distribution target_label={target_label} batch={batch_index}/{total_batches} "
                f"images={min(start + batch_size, len(target_rows))}/{len(target_rows)}"
            )
    scores = torch.cat(target_scores)
    return {
        "clean_target_acceptance_mean": float(scores.mean().item()),
        "clean_target_distribution_n": int(scores.numel()),
        "clean_target_distribution_source_generators": dict(
            sorted(Counter(row.generator_id for row in target_rows).items())
        ),
    }


def load_clip(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, int]:
    import open_clip

    model = open_clip.create_model(
        args.clip_model,
        pretrained=args.clip_pretrained,
        cache_dir=os.environ.get("PP_OPEN_CLIP_CACHE"),
    )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    feature_dim = getattr(model.visual, "output_dim", None)
    if feature_dim is None:
        raise RuntimeError("Unable to infer CLIP feature dimension")
    return model, int(feature_dim)


def clip_features(model: nn.Module, raw_images: torch.Tensor) -> torch.Tensor:
    processed = normalize_eval_batch(
        raw_images,
        mean=CLIP_MEAN,
        std=CLIP_STD,
    )
    features = model.encode_image(processed)
    return F.normalize(features, dim=1)


@torch.inference_mode()
def collect_score_query_bank(
    *,
    rows: Sequence[ManifestRow],
    victim: nn.Module,
    clip_model: nn.Module,
    variants: int,
    eps: float,
    batch_size: int,
    seed: int,
    device: torch.device,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if variants < 1:
        raise ValueError("query variants must be positive")
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    total_batches = math.ceil(len(rows) / batch_size)
    for batch_index, start in enumerate(range(0, len(rows), batch_size), start=1):
        batch_rows = rows[start : start + batch_size]
        originals = load_raw_batch(batch_rows, device)
        for variant in range(variants):
            if variant == 0:
                query_images = originals
            else:
                noise = torch.empty_like(originals).uniform_(-eps, eps, generator=generator)
                query_images = (originals + noise).clamp(0.0, 1.0)
            victim_scores = victim_fake_scores(victim, query_images)
            query_features = clip_features(clip_model, query_images)
            features.append(query_features.detach().cpu())
            targets.append(victim_scores.detach().cpu())
        if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
            log(
                f"score_queries split={split} batch={batch_index}/{total_batches} "
                f"images={min(start + batch_size, len(rows))}/{len(rows)} variants={variants}"
            )
    return torch.cat(features), torch.cat(targets)


def fit_score_surrogate(
    *,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    feature_dim: int,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[ScoreMLP, dict[str, object]]:
    model = ScoreMLP(feature_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_dataset = TensorDataset(train_features, train_targets)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_features_device = val_features.to(device)
    val_targets_device = val_targets.to(device)
    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            logits = model(batch_features)
            loss = F.binary_cross_entropy_with_logits(logits, batch_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_features.shape[0]
            total += batch_features.shape[0]
        model.eval()
        with torch.no_grad():
            val_predictions = torch.sigmoid(model(val_features_device))
            val_loss = float(F.mse_loss(val_predictions, val_targets_device).item())
            errors = (val_predictions - val_targets_device).abs()
        record = {
            "epoch": float(epoch),
            "train_bce": total_loss / max(1, total),
            "val_mse": val_loss,
            "val_mae": float(errors.mean().item()),
            "val_eps95": float(torch.quantile(errors, 0.95).item()),
        }
        history.append(record)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            log(
                f"surrogate_train epoch={epoch}/{epochs} train_bce={record['train_bce']:.6f} "
                f"val_mse={record['val_mse']:.6f} val_eps95={record['val_eps95']:.6f}"
            )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Surrogate training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        predictions = torch.sigmoid(model(val_features_device))
        errors = (predictions - val_targets_device).abs()
        centered_prediction = predictions - predictions.mean()
        centered_target = val_targets_device - val_targets_device.mean()
        correlation = (
            (centered_prediction * centered_target).mean()
            / (centered_prediction.std(unbiased=False) * centered_target.std(unbiased=False)).clamp_min(1e-12)
        )
    metrics: dict[str, object] = {
        "best_epoch": best_epoch,
        "val_mse": float(F.mse_loss(predictions, val_targets_device).item()),
        "val_mae": float(errors.mean().item()),
        "val_eps95": float(torch.quantile(errors, 0.95).item()),
        "val_max_error": float(errors.max().item()),
        "val_pearson": float(correlation.item()),
        "train_queries": int(train_features.shape[0]),
        "val_queries": int(val_features.shape[0]),
        "history": history,
    }
    return model, metrics


def stratified_fake_rows(rows: Sequence[ManifestRow], limit: int, seed: int) -> list[ManifestRow]:
    rng = random.Random(seed)
    groups: dict[str, list[ManifestRow]] = {name: [] for name in FAKE_GENERATORS}
    for row in rows:
        if row.generator_id in groups:
            groups[row.generator_id].append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)
    selected: list[ManifestRow] = []
    cursor = 0
    while len(selected) < limit:
        progressed = False
        for generator_id in FAKE_GENERATORS:
            if cursor < len(groups[generator_id]) and len(selected) < limit:
                selected.append(groups[generator_id][cursor])
                progressed = True
        if not progressed:
            break
        cursor += 1
    return selected


def select_attack_rows(rows: Sequence[ManifestRow], limit: int, seed: int) -> dict[str, list[ManifestRow]]:
    test_rows = split_rows(rows, "test")
    rng = random.Random(seed)
    real_rows = [row for row in test_rows if row.generator_id == "real"]
    rng.shuffle(real_rows)
    return {
        "real_to_fake": real_rows[:limit],
        "fake_to_real": stratified_fake_rows(test_rows, limit, seed + 1),
    }


def surrogate_target_score(
    clip_model: nn.Module,
    surrogate: ScoreMLP,
    images: torch.Tensor,
    target_label: int,
) -> torch.Tensor:
    fake_score = torch.sigmoid(surrogate(clip_features(clip_model, images)))
    return fake_score if target_label == 1 else 1.0 - fake_score


def targeted_pgd_surrogate(
    *,
    originals: torch.Tensor,
    clip_model: nn.Module,
    surrogate: ScoreMLP,
    target_label: int,
    eps: float,
    steps: int,
    restarts: int,
    seed: int,
) -> torch.Tensor:
    if eps <= 0 or steps <= 0 or restarts <= 0:
        raise ValueError("Attack epsilon, steps, and restarts must be positive")
    step_size = eps / 4.0
    generator = torch.Generator(device=originals.device)
    generator.manual_seed(seed)
    best_images = originals.detach().clone()
    best_scores = torch.full((originals.shape[0],), -math.inf, device=originals.device)
    for restart in range(restarts):
        noise = torch.empty_like(originals).uniform_(-eps, eps, generator=generator)
        current = (originals + noise).clamp(0.0, 1.0).detach()
        for _ in range(steps):
            current.requires_grad_(True)
            target_scores = surrogate_target_score(clip_model, surrogate, current, target_label)
            gradient = torch.autograd.grad(target_scores.sum(), current, only_inputs=True)[0]
            current = current.detach() + step_size * gradient.sign()
            current = torch.maximum(torch.minimum(current, originals + eps), originals - eps)
            current = current.clamp(0.0, 1.0).detach()
        with torch.no_grad():
            scores = surrogate_target_score(clip_model, surrogate, current, target_label)
        improved = scores > best_scores
        best_scores = torch.where(improved, scores, best_scores)
        best_images[improved] = current[improved]
        log(
            f"attack_restart_complete restart={restart + 1}/{restarts} "
            f"mean_surrogate_target_score={scores.mean().item():.6f}"
        )
    return best_images.detach()


def psnr_per_image(originals: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
    mse = (originals - adversarial).square().flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def save_attack_example(
    output_dir: Path,
    *,
    direction: str,
    rank: int,
    original: torch.Tensor,
    adversarial: torch.Tensor,
) -> dict[str, str]:
    directory = output_dir / "examples" / direction
    directory.mkdir(parents=True, exist_ok=True)
    clean_path = directory / f"{rank:03d}_clean.png"
    adversarial_path = directory / f"{rank:03d}_adversarial.png"
    delta_path = directory / f"{rank:03d}_delta_8x.png"
    tvf.to_pil_image(original.clamp(0.0, 1.0)).save(clean_path)
    tvf.to_pil_image(adversarial.clamp(0.0, 1.0)).save(adversarial_path)
    delta = (0.5 + 8.0 * (adversarial - original)).clamp(0.0, 1.0)
    tvf.to_pil_image(delta).save(delta_path)
    return {"clean": str(clean_path), "adversarial": str(adversarial_path), "delta_8x": str(delta_path)}


def summarize_attack_direction(
    *,
    direction: str,
    cell: Sequence[dict[str, object]],
    clean_target_metrics: dict[str, object],
) -> dict[str, object]:
    """Build a direction summary with an explicit target-score sign."""
    if direction not in ("real_to_fake", "fake_to_real"):
        raise ValueError(f"Unknown attack direction: {direction}")
    if not cell:
        raise RuntimeError(f"No attack rows to summarize for direction={direction}")
    target_label = 1 if direction == "real_to_fake" else 0
    correct_cell = [row for row in cell if int(row["initially_correct"]) == 1]
    targeted_asr_all = float(np.mean([float(row["success"]) for row in cell]))
    targeted_asr_initially_correct = (
        float(np.mean([float(row["success"]) for row in correct_cell])) if correct_cell else None
    )
    adversarial_target_scores = [
        float(
            target_acceptance_scores(
                torch.tensor(float(row["adversarial_fake_score"])),
                target_label,
            ).item()
        )
        for row in cell
    ]
    clean_target_acceptance_mean = float(clean_target_metrics["clean_target_acceptance_mean"])
    adversarial_source_target_score_mean = float(np.mean(adversarial_target_scores))
    return {
        "n": len(cell),
        "initially_correct_n": len(correct_cell),
        # The unqualified ASR stays comparable with the submitted paper's all-source definition.
        "targeted_asr": targeted_asr_all,
        "targeted_asr_population": "all_attacked_sources",
        "targeted_asr_initially_correct": targeted_asr_initially_correct,
        "targeted_asr_all": targeted_asr_all,
        "clean_target_acceptance_mean": clean_target_acceptance_mean,
        "adversarial_source_target_score_mean": adversarial_source_target_score_mean,
        "empirical_gap_upper_bound": (
            clean_target_acceptance_mean - adversarial_source_target_score_mean
        ),
        "empirical_gap_population": "all_attacked_sources_vs_full_clean_test_target_distribution",
        "clean_target_distribution_n": int(clean_target_metrics["clean_target_distribution_n"]),
        "clean_target_distribution_source_generators": clean_target_metrics[
            "clean_target_distribution_source_generators"
        ],
        "clean_fake_score_mean": float(np.mean([float(row["clean_fake_score"]) for row in cell])),
        "adversarial_fake_score_mean": float(
            np.mean([float(row["adversarial_fake_score"]) for row in cell])
        ),
        "linf_max": float(max(float(row["linf"]) for row in cell)),
        "psnr_median": float(np.median([float(row["psnr"]) for row in cell])),
        "source_generators": dict(sorted(Counter(str(row["generator_id"]) for row in cell).items())),
    }


def run_targeted_attacks(
    *,
    args: argparse.Namespace,
    rows: Sequence[ManifestRow],
    victim: nn.Module,
    clip_model: nn.Module,
    surrogate: ScoreMLP,
    threshold: float,
    run_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    selected = select_attack_rows(rows, args.attack_samples_per_direction, args.seed + 100)
    clean_target_metrics_by_label = {
        target_label: clean_target_distribution_metrics(
            rows=rows,
            victim=victim,
            target_label=target_label,
            batch_size=args.eval_batch_size,
            device=device,
        )
        for target_label in (0, 1)
    }
    csv_rows: list[dict[str, object]] = []
    direction_summaries: dict[str, object] = {}
    saved_examples: dict[str, list[dict[str, str]]] = {}
    for direction, direction_rows in selected.items():
        target_label = 1 if direction == "real_to_fake" else 0
        clean_source_label = 0 if target_label == 1 else 1
        saved_examples[direction] = []
        log(
            f"attack_direction_start direction={direction} n={len(direction_rows)} eps={args.attack_eps:.8f} "
            f"steps={args.attack_steps} restarts={args.attack_restarts}"
        )
        for batch_index, start in enumerate(range(0, len(direction_rows), args.attack_batch_size), start=1):
            batch_rows = direction_rows[start : start + args.attack_batch_size]
            originals = load_raw_batch(batch_rows, device)
            with torch.no_grad():
                clean_scores = victim_fake_scores(victim, originals)
            adversarial = targeted_pgd_surrogate(
                originals=originals,
                clip_model=clip_model,
                surrogate=surrogate,
                target_label=target_label,
                eps=args.attack_eps,
                steps=args.attack_steps,
                restarts=args.attack_restarts,
                seed=args.seed + 1000 + start,
            )
            with torch.no_grad():
                adversarial_scores = victim_fake_scores(victim, adversarial)
                linf = (adversarial - originals).abs().flatten(1).max(dim=1).values
                psnr = psnr_per_image(originals, adversarial)
            clean_predictions = (clean_scores >= threshold).to(torch.int64)
            adversarial_predictions = (adversarial_scores >= threshold).to(torch.int64)
            successes = adversarial_predictions == target_label
            for index, row in enumerate(batch_rows):
                initially_correct = bool(clean_predictions[index].item() == clean_source_label)
                success = bool(successes[index].item())
                result_row = {
                    "direction": direction,
                    "prompt_id": row.prompt_id,
                    "image_id": row.image_id,
                    "generator_id": row.generator_id,
                    "image_path": row.image_path,
                    "target_label": target_label,
                    "eps": args.attack_eps,
                    "steps": args.attack_steps,
                    "restarts": args.attack_restarts,
                    "clean_fake_score": float(clean_scores[index].item()),
                    "adversarial_fake_score": float(adversarial_scores[index].item()),
                    "adversarial_target_score": float(
                        target_acceptance_scores(adversarial_scores[index], target_label).item()
                    ),
                    "clean_prediction": int(clean_predictions[index].item()),
                    "adversarial_prediction": int(adversarial_predictions[index].item()),
                    "initially_correct": int(initially_correct),
                    "success": int(success),
                    "success_on_initially_correct": int(success and initially_correct),
                    "linf": float(linf[index].item()),
                    "psnr": float(psnr[index].item()),
                }
                csv_rows.append(result_row)
                if len(saved_examples[direction]) < args.save_examples_per_direction and success:
                    saved_examples[direction].append(
                        save_attack_example(
                            run_dir,
                            direction=direction,
                            rank=len(saved_examples[direction]),
                            original=originals[index].detach().cpu(),
                            adversarial=adversarial[index].detach().cpu(),
                        )
                    )
            log(
                f"attack_batch_complete direction={direction} batch={batch_index}/"
                f"{math.ceil(len(direction_rows) / args.attack_batch_size)} "
                f"victim_target_success={successes.float().mean().item():.4f}"
            )
        cell = [row for row in csv_rows if row["direction"] == direction]
        direction_summaries[direction] = summarize_attack_direction(
            direction=direction,
            cell=cell,
            clean_target_metrics=clean_target_metrics_by_label[target_label],
        )
        log(f"attack_direction_complete direction={direction} summary={json.dumps(direction_summaries[direction])}")
    csv_path = run_dir / "attack_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = {
        "victim_family": "raw_pixel_imagenet_resnet",
        "surrogate_family": "frozen_clip_features_mlp_score_query",
        "attack": "targeted_linf_pgd_on_surrogate_evaluated_on_victim",
        "victim_threshold": threshold,
        "eps": args.attack_eps,
        "steps": args.attack_steps,
        "restarts": args.attack_restarts,
        "directions": direction_summaries,
        "saved_examples": saved_examples,
        "results_csv": str(csv_path),
    }
    atomic_write_json(run_dir / "attack_summary.json", summary)
    return summary


def fit_surrogate_and_attack(
    args: argparse.Namespace,
    rows: Sequence[ManifestRow],
    run_dir: Path,
    device: torch.device,
    manifest_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    victim, clean_metrics = load_victim(
        run_dir,
        device,
        args=args,
        manifest_sha256=manifest_sha256,
    )
    gate_value = float(clean_metrics["val"]["auc"])
    if gate_value < args.clean_auc_gate:
        raise RuntimeError(
            f"CLEAN_VALIDATION_AUC_GATE_FAILED val_auc={gate_value:.6f} required={args.clean_auc_gate:.6f}"
        )
    log(
        f"clean_validation_auc_gate_passed val_auc={gate_value:.6f} "
        f"required={args.clean_auc_gate:.6f} test_auc_reporting_only={clean_metrics['test']['auc']} "
        f"claim_eligible={clean_metrics['claim_eligible']}"
    )
    clip_model, feature_dim = load_clip(args, device)
    log(
        f"surrogate_query_start victim=raw_pixel_{args.arch} surrogate=CLIP-{args.clip_model}-MLP "
        f"query_variants={args.query_variants} query_eps={args.query_eps:.8f}"
    )
    train_rows = split_rows(rows, "train")
    val_rows = split_rows(rows, "val")
    train_features, train_targets = collect_score_query_bank(
        rows=train_rows,
        victim=victim,
        clip_model=clip_model,
        variants=args.query_variants,
        eps=args.query_eps,
        batch_size=args.query_batch_size,
        seed=args.seed + 10,
        device=device,
        split="train",
    )
    val_features, val_targets = collect_score_query_bank(
        rows=val_rows,
        victim=victim,
        clip_model=clip_model,
        variants=args.query_variants,
        eps=args.query_eps,
        batch_size=args.query_batch_size,
        seed=args.seed + 20,
        device=device,
        split="val",
    )
    query_cache_path = run_dir / "score_query_features.pt"
    torch.save(
        {
            "train_features": train_features,
            "train_targets": train_targets,
            "val_features": val_features,
            "val_targets": val_targets,
            "clip_model": args.clip_model,
            "clip_pretrained": args.clip_pretrained,
            "query_eps": args.query_eps,
            "query_variants": args.query_variants,
        },
        query_cache_path,
    )
    surrogate, surrogate_metrics = fit_score_surrogate(
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        feature_dim=feature_dim,
        hidden_dim=args.surrogate_hidden_dim,
        epochs=args.surrogate_epochs,
        learning_rate=args.surrogate_lr,
        batch_size=max(32, args.query_batch_size * 4),
        seed=args.seed + 30,
        device=device,
    )
    surrogate_checkpoint_path = run_dir / "surrogate_checkpoint.pt"
    torch.save(
        {
            "family": "frozen_clip_features_mlp_score_query",
            "clip_model": args.clip_model,
            "clip_pretrained": args.clip_pretrained,
            "feature_dim": feature_dim,
            "hidden_dim": args.surrogate_hidden_dim,
            "state_dict": {key: value.detach().cpu() for key, value in surrogate.state_dict().items()},
            "victim_family": "raw_pixel_imagenet_resnet",
            "victim_architecture": args.arch,
        },
        surrogate_checkpoint_path,
    )
    surrogate_metrics.update(
        {
            "family": "frozen_clip_features_mlp_score_query",
            "victim_family": "raw_pixel_imagenet_resnet",
            "diagnostic_scope": (
                "Validation-query diagnostics only; val_mae, val_eps95, and val_max_error are not "
                "uniform approximation bounds."
            ),
            "clip_model": args.clip_model,
            "clip_pretrained": args.clip_pretrained,
            "query_variants": args.query_variants,
            "query_eps": args.query_eps,
            "checkpoint": str(surrogate_checkpoint_path),
            "checkpoint_sha256": file_sha256(surrogate_checkpoint_path),
            "query_cache": str(query_cache_path),
            "query_cache_sha256": file_sha256(query_cache_path),
        }
    )
    atomic_write_json(run_dir / "surrogate_metrics.json", surrogate_metrics)
    log(
        f"surrogate_complete val_mae={surrogate_metrics['val_mae']:.6f} "
        f"val_eps95={surrogate_metrics['val_eps95']:.6f} val_pearson={surrogate_metrics['val_pearson']:.6f}"
    )
    threshold = float(clean_metrics["val"]["threshold"])
    attack_summary = run_targeted_attacks(
        args=args,
        rows=rows,
        victim=victim,
        clip_model=clip_model,
        surrogate=surrogate,
        threshold=threshold,
        run_dir=run_dir,
        device=device,
    )
    return surrogate_metrics, attack_summary


def environment_snapshot(args: argparse.Namespace, manifest: Path, manifest_validation: dict[str, object]) -> dict[str, object]:
    script_path = Path(__file__).resolve()
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "device_count": torch.cuda.device_count(),
        }
    return {
        "created_at_utc": utc_now(),
        "arguments": vars(args),
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "manifest_validation": manifest_validation,
        "script": str(script_path),
        "script_sha256": file_sha256(script_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "container_image": args.container_image,
        "storage": {
            "hf_home": os.environ.get("HF_HOME"),
            "torch_home": os.environ.get("TORCH_HOME"),
            "open_clip_cache": os.environ.get("PP_OPEN_CLIP_CACHE"),
            "output_root": args.output_root,
        },
        "scientific_contract": {
            "victim": "Fixed ImageNet-pretrained raw-pixel ResNet-18; no CLIP features in victim",
            "surrogate": "MLP over frozen CLIP image features fitted only to victim score queries",
            "surrogate_diagnostics": "Validation-query diagnostics only; not uniform approximation bounds",
            "attack": "targeted Linf PGD through surrogate, evaluated only on the raw-pixel victim",
            "attack_domain": "deterministic victim evaluation crop: Resize(256)+CenterCrop(224)+ToTensor",
            "split": "B1 prompt-disjoint train/validation/test",
            "execution_gate": "validation AUC >= configured threshold",
            "claim_gate": "validation and test AUC >= configured threshold",
            "test_policy": "report test result without architecture switching or test-guided tuning/rerun",
            "dire_claim": "not DIRE and must not be described as DIRE",
        },
    }


def main() -> None:
    args = parse_args()
    if not (0.0 < args.clean_auc_gate <= 1.0):
        raise ValueError("clean AUC gate must be in (0, 1]")
    if args.attack_eps > 8.0 / 255.0 + 1e-12:
        raise ValueError("This experiment caps the targeted Linf attack at 8/255")
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA-capable GPU.")
    device = torch.device("cuda:0")
    manifest = Path(args.manifest).resolve()
    validate_full_scientific_contract(args, manifest)
    rows = read_manifest(manifest, Path(args.image_root) if args.image_root else None)
    manifest_validation = validate_manifest(rows)
    run_dir = Path(args.output_root).resolve() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = environment_snapshot(args, manifest, manifest_validation)
    atomic_write_json(run_dir / "run_snapshot.json", snapshot)
    atomic_write_json(
        run_dir / "status.json",
        {"stage": "starting", "updated_at_utc": utc_now(), "run_id": args.run_id},
    )
    log(
        f"run_start run_id={args.run_id} stage={args.stage} subset={args.subset} arch={args.arch} "
        f"manifest_rows={manifest_validation['rows']} gpu={torch.cuda.get_device_name(0)}"
    )
    try:
        clean_metrics: dict[str, object] | None = None
        surrogate_metrics: dict[str, object] | None = None
        attack_summary: dict[str, object] | None = None
        if args.stage in {"train-victim", "all"}:
            atomic_write_json(
                run_dir / "status.json",
                {"stage": "training_victim", "updated_at_utc": utc_now(), "run_id": args.run_id},
            )
            clean_metrics = train_victim(
                args,
                rows,
                run_dir,
                device,
                manifest_sha256=file_sha256(manifest),
            )
        if args.stage in {"surrogate-attack", "all"}:
            atomic_write_json(
                run_dir / "status.json",
                {"stage": "fitting_surrogate_and_attacking", "updated_at_utc": utc_now(), "run_id": args.run_id},
            )
            surrogate_metrics, attack_summary = fit_surrogate_and_attack(
                args,
                rows,
                run_dir,
                device,
                manifest_sha256=file_sha256(manifest),
            )
        decision = {
            "run_id": args.run_id,
            "subset": args.subset,
            "architecture": args.arch,
            "clean_auc_gate": args.clean_auc_gate,
            "clean_metrics": clean_metrics,
            "surrogate_metrics": surrogate_metrics,
            "attack_summary": attack_summary,
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(run_dir / "decision.json", decision)
        atomic_write_json(
            run_dir / "status.json",
            {"stage": "completed", "updated_at_utc": utc_now(), "run_id": args.run_id},
        )
        log(f"run_complete run_id={args.run_id} decision={run_dir / 'decision.json'}")
    except Exception as exc:
        atomic_write_json(
            run_dir / "status.json",
            {
                "stage": "failed",
                "updated_at_utc": utc_now(),
                "run_id": args.run_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        log(f"run_failed run_id={args.run_id} error_type={type(exc).__name__} error={exc}")
        raise


if __name__ == "__main__":
    main()
