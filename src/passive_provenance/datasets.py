from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from passive_provenance.constants import CLASS_NAMES
from passive_provenance.utils.image_ops import normalize_real_image
from passive_provenance.utils.io import stable_sha256, write_csv


@dataclass(frozen=True)
class SampleRecord:
    benchmark_id: str
    prompt_id: int
    image_id: int
    caption: str
    split: str
    source_kind: str
    generator_id: str
    seed: int
    image_path: str
    width: int
    height: int
    preprocess_hash: str


@dataclass(frozen=True)
class TripletRecord:
    image_id: int
    caption: str
    split: str
    seed: int
    real_path: str
    sd15_path: str
    sdxl_path: str
    width: int
    height: int
    preprocess_hash: str


def load_coco_caption_annotations(coco_root: str | Path) -> tuple[dict[int, str], dict[int, str]]:
    coco_root = Path(coco_root)
    captions_path = coco_root / "annotations" / "captions_val2017.json"
    instances = json.loads(captions_path.read_text(encoding="utf-8"))
    image_file_map = {item["id"]: item["file_name"] for item in instances["images"]}
    grouped: dict[int, list[tuple[int, str]]] = {}
    for ann in instances["annotations"]:
        grouped.setdefault(ann["image_id"], []).append((ann["id"], ann["caption"].strip()))
    captions = {image_id: min(entries, key=lambda item: item[0])[1] for image_id, entries in grouped.items()}
    return captions, image_file_map


def deterministic_split(image_ids: Iterable[int], split_counts: dict[str, int], seed: int) -> dict[int, str]:
    ordered = np.array(sorted(image_ids), dtype=np.int64)
    rng = np.random.default_rng(seed)
    shuffled = ordered.copy()
    rng.shuffle(shuffled)
    total = sum(split_counts.values())
    if len(shuffled) < total:
        raise ValueError(f"Requested {total} images but only found {len(shuffled)} candidates.")
    selected = shuffled[:total]
    split_map: dict[int, str] = {}
    cursor = 0
    for split_name in ("train", "val", "test"):
        count = split_counts[split_name]
        for image_id in selected[cursor : cursor + count]:
            split_map[int(image_id)] = split_name
        cursor += count
    return split_map


def build_preprocess_hash(size: int, caption_policy: str) -> str:
    return stable_sha256(f"size={size}|caption_policy={caption_policy}|format=png|mode=rgb|exif=stripped")


def _sample_manifest_fieldnames() -> list[str]:
    return list(SampleRecord.__dataclass_fields__.keys())


def _triplet_manifest_fieldnames() -> list[str]:
    return list(TripletRecord.__dataclass_fields__.keys())


def classify_sample(record: SampleRecord, class_names: tuple[str, ...] | list[str]) -> str | None:
    names = tuple(class_names)
    if record.generator_id in names:
        return record.generator_id
    if record.generator_id == "real" and "real" in names:
        return "real"
    fake_alias = "fake" if "fake" in names else "ai" if "ai" in names else None
    if fake_alias is not None and record.generator_id != "real":
        return fake_alias
    return None


def prepare_sample_manifest(
    *,
    coco_root: str | Path,
    output_manifest_path: str | Path,
    smoke_manifest_path: str | Path,
    benchmark_id: str,
    real_output_dir: str | Path,
    generator_output_dirs: dict[str, str | Path],
    generator_ids: Iterable[str],
    split_counts: dict[str, int],
    smoke_counts: dict[str, int],
    image_size: int,
    seed: int,
    caption_policy: str,
) -> list[SampleRecord]:
    captions, image_file_map = load_coco_caption_annotations(coco_root)
    split_map = deterministic_split(captions.keys(), split_counts, seed)
    preprocess_hash = build_preprocess_hash(image_size, caption_policy)
    coco_root = Path(coco_root)
    real_output_dir = Path(real_output_dir)
    generator_output_dirs = {generator_id: Path(path) for generator_id, path in generator_output_dirs.items()}
    sample_records: list[SampleRecord] = []
    by_split: dict[str, list[SampleRecord]] = {"train": [], "val": [], "test": []}
    generator_ids = tuple(generator_ids)

    for image_id in sorted(split_map):
        split = split_map[image_id]
        file_name = image_file_map[image_id]
        source_path = coco_root / "val2017" / file_name
        dest_real_path = real_output_dir / split / f"{image_id:012d}.png"
        normalize_real_image(source_path, dest_real_path, image_size)
        seed_value = int(seed + image_id)
        entries = [
            SampleRecord(
                benchmark_id=benchmark_id,
                prompt_id=image_id,
                image_id=image_id,
                caption=captions[image_id],
                split=split,
                source_kind="real",
                generator_id="real",
                seed=seed_value,
                image_path=str(dest_real_path),
                width=image_size,
                height=image_size,
                preprocess_hash=preprocess_hash,
            )
        ]
        for generator_id in generator_ids:
            generated_output_dir = generator_output_dirs[generator_id]
            entries.append(
                SampleRecord(
                    benchmark_id=benchmark_id,
                    prompt_id=image_id,
                    image_id=image_id,
                    caption=captions[image_id],
                    split=split,
                    source_kind="generated",
                    generator_id=generator_id,
                    seed=seed_value,
                    image_path=str(generated_output_dir / split / f"{image_id:012d}.png"),
                    width=image_size,
                    height=image_size,
                    preprocess_hash=preprocess_hash,
                )
            )
        sample_records.extend(entries)
        by_split[split].extend(entries)

    write_csv(Path(output_manifest_path), [asdict(record) for record in sample_records], _sample_manifest_fieldnames())

    smoke_records: list[SampleRecord] = []
    for split_name in ("train", "val", "test"):
        per_prompt: dict[int, list[SampleRecord]] = {}
        for record in by_split[split_name]:
            per_prompt.setdefault(record.prompt_id, []).append(record)
        for prompt_id in sorted(per_prompt)[: smoke_counts[split_name]]:
            smoke_records.extend(sorted(per_prompt[prompt_id], key=lambda item: item.generator_id))
    write_csv(Path(smoke_manifest_path), [asdict(record) for record in smoke_records], _sample_manifest_fieldnames())
    return sample_records


def export_triplet_manifest(
    sample_records: list[SampleRecord],
    output_manifest_path: str | Path,
    smoke_manifest_path: str | Path,
    smoke_counts: dict[str, int],
) -> list[TripletRecord]:
    grouped: dict[tuple[int, str], dict[str, SampleRecord]] = {}
    for record in sample_records:
        key = (record.prompt_id, record.split)
        grouped.setdefault(key, {})[record.generator_id] = record
    triplets: list[TripletRecord] = []
    per_split: dict[str, list[TripletRecord]] = {"train": [], "val": [], "test": []}
    for (_prompt_id, split), entries in sorted(grouped.items()):
        required = {"real", "sd15", "sdxl"}
        if not required.issubset(entries):
            continue
        real_entry = entries["real"]
        triplet = TripletRecord(
            image_id=real_entry.image_id,
            caption=real_entry.caption,
            split=split,
            seed=real_entry.seed,
            real_path=entries["real"].image_path,
            sd15_path=entries["sd15"].image_path,
            sdxl_path=entries["sdxl"].image_path,
            width=real_entry.width,
            height=real_entry.height,
            preprocess_hash=real_entry.preprocess_hash,
        )
        triplets.append(triplet)
        per_split[split].append(triplet)

    write_csv(Path(output_manifest_path), [asdict(record) for record in triplets], _triplet_manifest_fieldnames())

    smoke_records: list[TripletRecord] = []
    for split_name in ("train", "val", "test"):
        smoke_records.extend(per_split[split_name][: smoke_counts[split_name]])
    write_csv(Path(smoke_manifest_path), [asdict(record) for record in smoke_records], _triplet_manifest_fieldnames())
    return triplets


def prepare_triplet_manifest(
    coco_root: str | Path,
    output_manifest_path: str | Path,
    smoke_manifest_path: str | Path,
    real_output_dir: str | Path,
    generated_output_dir: str | Path,
    split_counts: dict[str, int],
    smoke_counts: dict[str, int],
    image_size: int,
    seed: int,
    caption_policy: str,
) -> list[TripletRecord]:
    captions, image_file_map = load_coco_caption_annotations(coco_root)
    split_map = deterministic_split(captions.keys(), split_counts, seed)
    preprocess_hash = build_preprocess_hash(image_size, caption_policy)
    coco_root = Path(coco_root)
    real_output_dir = Path(real_output_dir)
    generated_output_dir = Path(generated_output_dir)
    records: list[TripletRecord] = []
    by_split: dict[str, list[TripletRecord]] = {"train": [], "val": [], "test": []}

    for image_id in sorted(split_map):
        split = split_map[image_id]
        file_name = image_file_map[image_id]
        source_path = coco_root / "val2017" / file_name
        dest_real_path = real_output_dir / split / f"{image_id:012d}.png"
        normalize_real_image(source_path, dest_real_path, image_size)
        record = TripletRecord(
            image_id=image_id,
            caption=captions[image_id],
            split=split,
            seed=int(seed + image_id),
            real_path=str(dest_real_path),
            sd15_path=str(generated_output_dir / "sd15" / split / f"{image_id:012d}.png"),
            sdxl_path=str(generated_output_dir / "sdxl" / split / f"{image_id:012d}.png"),
            width=image_size,
            height=image_size,
            preprocess_hash=preprocess_hash,
        )
        records.append(record)
        by_split[split].append(record)

    write_csv(Path(output_manifest_path), [asdict(record) for record in records], list(TripletRecord.__dataclass_fields__.keys()))

    smoke_records: list[TripletRecord] = []
    for split_name in ("train", "val", "test"):
        smoke_records.extend(by_split[split_name][: smoke_counts[split_name]])
    write_csv(Path(smoke_manifest_path), [asdict(record) for record in smoke_records], list(TripletRecord.__dataclass_fields__.keys()))
    return records


def load_triplet_manifest(path: str | Path) -> list[TripletRecord]:
    import csv

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        TripletRecord(
            image_id=int(row["image_id"]),
            caption=row["caption"],
            split=row["split"],
            seed=int(row["seed"]),
            real_path=row["real_path"],
            sd15_path=row["sd15_path"],
            sdxl_path=row["sdxl_path"],
            width=int(row["width"]),
            height=int(row["height"]),
            preprocess_hash=row["preprocess_hash"],
        )
        for row in rows
    ]


def load_sample_manifest(path: str | Path) -> list[SampleRecord]:
    import csv

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        SampleRecord(
            benchmark_id=row["benchmark_id"],
            prompt_id=int(row["prompt_id"]),
            image_id=int(row["image_id"]),
            caption=row["caption"],
            split=row["split"],
            source_kind=row["source_kind"],
            generator_id=row["generator_id"],
            seed=int(row["seed"]),
            image_path=row["image_path"],
            width=int(row["width"]),
            height=int(row["height"]),
            preprocess_hash=row["preprocess_hash"],
        )
        for row in rows
    ]


class ManifestImageDataset(Dataset):
    def __init__(self, records: list[TripletRecord], split: str, label_mode: str) -> None:
        self.samples: list[tuple[str, int, TripletRecord]] = []
        label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        for record in records:
            if record.split != split:
                continue
            self.samples.append((record.real_path, label_map["real"], record))
            self.samples.append((record.sd15_path, label_map["sd15"], record))
            self.samples.append((record.sdxl_path, label_map["sdxl"], record))
        if label_mode == "smoke":
            self.samples = self.samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, TripletRecord, str]:
        path, label, record = self.samples[index]
        image = Image.open(path).convert("RGB")
        array = np.asarray(image).astype("float32") / 255.0
        return array, label, record, path


class SampleManifestDataset(Dataset):
    def __init__(self, records: list[SampleRecord], split: str, class_names: Iterable[str]) -> None:
        self.class_names = tuple(class_names)
        self.label_map = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples: list[tuple[str, int, SampleRecord]] = []
        for record in records:
            if record.split != split:
                continue
            label_name = classify_sample(record, self.class_names)
            if label_name is None:
                continue
            self.samples.append((record.image_path, self.label_map[label_name], record))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, SampleRecord, str]:
        path, label, record = self.samples[index]
        image = Image.open(path).convert("RGB")
        array = np.asarray(image).astype("float32") / 255.0
        return array, label, record, path
