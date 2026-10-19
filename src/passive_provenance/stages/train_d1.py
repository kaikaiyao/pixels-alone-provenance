from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from passive_provenance.attacks.pgd import pgd_maximize_cross_entropy
from passive_provenance.config import load_config, resolve_artifact_path, resolve_artifact_relative
from passive_provenance.datasets import load_triplet_manifest
from passive_provenance.models.d1 import FrozenClipVerifier
from passive_provenance.utils.io import write_json


class TripletTorchDataset(Dataset):
    def __init__(self, records, split: str):
        self.samples = []
        for record in records:
            if record.split != split:
                continue
            self.samples.extend(
                [
                    (record.real_path, 0, record.image_id),
                    (record.sd15_path, 1, record.image_id),
                    (record.sdxl_path, 2, record.image_id),
                ]
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, image_id = self.samples[index]
        array = np.asarray(Image.open(path).convert("RGB")).astype("float32") / 255.0
        return torch.from_numpy(array).permute(2, 0, 1), label, image_id


def extract_features(model: FrozenClipVerifier, loader: DataLoader, device: torch.device):
    features = []
    labels = []
    with torch.no_grad():
        for images, batch_labels, _batch_image_ids in tqdm(loader, desc="extract-features", leave=False):
            images = images.to(device)
            encoded = model.encode(images).cpu()
            features.append(encoded)
            labels.append(batch_labels)
    return torch.cat(features), torch.cat(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the D1 CLIP + linear head verifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--subset", choices=["full", "smoke"], default="smoke")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--adversarial-training", action="store_true")
    return parser.parse_args()


def output_checkpoint_path(config: dict[str, object], override: str | None) -> Path:
    if override:
        return resolve_artifact_relative(config, override)
    return resolve_artifact_path(config, "d1_checkpoint")


def evaluate_accuracy(model: FrozenClipVerifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _image_ids in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = model.forward_probs(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return float(correct / max(total, 1))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest_key = "smoke_manifest" if args.subset == "smoke" else "dataset_manifest"
    records = load_triplet_manifest(resolve_artifact_path(config, manifest_key))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrozenClipVerifier(
        clip_model_name=config["d1"]["clip_model_name"],
        clip_pretrained=config["d1"]["clip_pretrained"],
        input_size=int(config["data"]["clip_input_size"]),
        mean=tuple(config["data"]["clip_mean"]),
        std=tuple(config["data"]["clip_std"]),
    ).to(device)
    adv_cfg = dict(config["d1"].get("adversarial_training", {}))
    adv_enabled = bool(args.adversarial_training or adv_cfg.get("enabled", False))
    batch_size = int(adv_cfg.get("batch_size", config["d1"]["batch_size"])) if adv_enabled else int(config["d1"]["batch_size"])
    train_loader = DataLoader(
        TripletTorchDataset(records, "train"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TripletTorchDataset(records, "val"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=float(config["d1"]["learning_rate"]), weight_decay=float(config["d1"]["weight_decay"]))
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_val = -1.0
    epochs = int(config["d1"]["epochs"])

    if adv_enabled:
        eps = float(adv_cfg["eps"])
        steps = int(adv_cfg["steps"])
        step_size = float(adv_cfg["step_size"])
        restarts = int(adv_cfg.get("restarts", 1))
        clean_weight = float(adv_cfg.get("clean_weight", 0.5))
        adv_weight = float(adv_cfg.get("adv_weight", 0.5))
        for epoch in range(epochs):
            model.train()
            model.encoder.eval()
            running_loss = 0.0
            running_clean = 0
            running_adv = 0
            total = 0
            for images, labels, _image_ids in tqdm(train_loader, desc=f"adv-train-epoch-{epoch + 1}", leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                adversarial = pgd_maximize_cross_entropy(
                    model.forward_logits,
                    images,
                    labels,
                    eps=eps,
                    steps=steps,
                    step_size=step_size,
                    restarts=restarts,
                ).adversarial
                clean_logits = model.forward_logits(images)
                adv_logits = model.forward_logits(adversarial)
                loss = clean_weight * criterion(clean_logits, labels) + adv_weight * criterion(adv_logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_size_now = int(labels.numel())
                running_loss += float(loss.item()) * batch_size_now
                running_clean += int((clean_logits.argmax(dim=1) == labels).sum().item())
                running_adv += int((adv_logits.argmax(dim=1) == labels).sum().item())
                total += batch_size_now
            val_acc = evaluate_accuracy(model, val_loader, device)
            train_loss = running_loss / max(total, 1)
            train_clean_acc = running_clean / max(total, 1)
            train_adv_acc = running_adv / max(total, 1)
            print(
                "[train-d1] "
                f"mode=adv epoch={epoch + 1}/{epochs} "
                f"train_loss={train_loss:.4f} train_clean_acc={train_clean_acc:.4f} "
                f"train_adv_acc={train_adv_acc:.4f} val_acc={val_acc:.4f}",
                flush=True,
            )
            if val_acc > best_val:
                best_val = val_acc
                best_state = {key: value.detach().cpu() for key, value in model.head.state_dict().items()}
    else:
        train_features, train_labels = extract_features(model, train_loader, device)
        val_features, val_labels = extract_features(model, val_loader, device)
        for epoch in range(epochs):
            logits = model.head(train_features.to(device))
            loss = criterion(logits, train_labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                val_probs = torch.softmax(model.head(val_features.to(device)), dim=1)
                val_pred = val_probs.argmax(dim=1)
                val_acc = float((val_pred == val_labels.to(device)).float().mean().item())
            print(
                f"[train-d1] mode=clean epoch={epoch + 1}/{epochs} loss={float(loss.item()):.4f} val_acc={val_acc:.4f}",
                flush=True,
            )
            if val_acc > best_val:
                best_val = val_acc
                best_state = {key: value.detach().cpu() for key, value in model.head.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No validation checkpoint captured.")
    model.head.load_state_dict(best_state)
    checkpoint_path = output_checkpoint_path(config, args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "clip_model_name": config["d1"]["clip_model_name"],
            "clip_pretrained": config["d1"]["clip_pretrained"],
            "input_size": config["data"]["clip_input_size"],
            "mean": config["data"]["clip_mean"],
            "std": config["data"]["clip_std"],
            "class_names": ("real", "sd15", "sdxl"),
            "head_state_dict": model.head.state_dict(),
            "training_mode": "adversarial" if adv_enabled else "clean",
            "adversarial_training": adv_cfg if adv_enabled else {},
            "best_val_accuracy": best_val,
        },
        checkpoint_path,
    )
    write_json(
        checkpoint_path.with_suffix(".json"),
        {
            "subset": args.subset,
            "best_val_accuracy": best_val,
            "training_mode": "adversarial" if adv_enabled else "clean",
            "adversarial_training": adv_cfg if adv_enabled else {},
        },
    )


if __name__ == "__main__":
    main()
