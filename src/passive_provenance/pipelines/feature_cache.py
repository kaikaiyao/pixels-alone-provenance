from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from passive_provenance.datasets import SampleManifestDataset, SampleRecord


class TorchSampleDataset(Dataset):
    def __init__(self, records: list[SampleRecord], split: str, class_names: tuple[str, ...]) -> None:
        self.samples = SampleManifestDataset(records, split=split, class_names=class_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        array, label, record, path = self.samples[index]
        image = torch.from_numpy(array).permute(2, 0, 1)
        return image, label, record.image_id, record.generator_id, path


def build_feature_cache(
    model,
    records: list[SampleRecord],
    *,
    class_names: tuple[str, ...],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
) -> dict[str, dict[str, object]]:
    cache: dict[str, dict[str, object]] = {}
    for split in ("train", "val", "test"):
        loader = DataLoader(
            TorchSampleDataset(records, split, class_names),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        features = []
        labels = []
        image_ids = []
        generator_ids: list[str] = []
        paths: list[str] = []
        with torch.no_grad():
            for images, batch_labels, batch_ids, batch_generators, batch_paths in tqdm(loader, desc=f"extract-{split}", leave=False):
                encoded = model.encode(images.to(device)).detach().cpu()
                features.append(encoded)
                labels.append(batch_labels.detach().cpu())
                image_ids.append(batch_ids.detach().cpu())
                generator_ids.extend(list(batch_generators))
                paths.extend(list(batch_paths))
        cache[split] = {
            "features": torch.cat(features, dim=0) if features else torch.empty(0, model.backbone.feature_dim),
            "labels": torch.cat(labels, dim=0) if labels else torch.empty(0, dtype=torch.long),
            "image_ids": torch.cat(image_ids, dim=0) if image_ids else torch.empty(0, dtype=torch.long),
            "generator_ids": generator_ids,
            "paths": paths,
        }
    return cache


def load_image_tensor(path: str | Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image).astype("float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)
