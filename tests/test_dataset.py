from __future__ import annotations

import json

from PIL import Image

from passive_provenance.datasets import build_preprocess_hash, deterministic_split, prepare_sample_manifest


def test_deterministic_split_is_reproducible():
    image_ids = list(range(2000))
    counts = {"train": 600, "val": 300, "test": 300}
    split_a = deterministic_split(image_ids, counts, 1234)
    split_b = deterministic_split(image_ids, counts, 1234)
    assert split_a == split_b
    assert sum(1 for split in split_a.values() if split == "train") == 600
    assert sum(1 for split in split_a.values() if split == "val") == 300
    assert sum(1 for split in split_a.values() if split == "test") == 300


def test_preprocess_hash_changes_with_policy():
    first = build_preprocess_hash(512, "lowest_annotation_id")
    second = build_preprocess_hash(512, "different_policy")
    assert first != second


def test_prepare_sample_manifest_uses_generator_output_dir_map(tmp_path):
    coco_root = tmp_path / "coco"
    (coco_root / "annotations").mkdir(parents=True)
    (coco_root / "val2017").mkdir()
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(coco_root / "val2017" / "000000000001.jpg")
    annotations = {
        "images": [{"id": 1, "file_name": "000000000001.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "caption": "a test caption"}],
    }
    (coco_root / "annotations" / "captions_val2017.json").write_text(json.dumps(annotations), encoding="utf-8")

    output_manifest = tmp_path / "sample_manifest.csv"
    smoke_manifest = tmp_path / "sample_manifest_smoke.csv"
    records = prepare_sample_manifest(
        coco_root=coco_root,
        output_manifest_path=output_manifest,
        smoke_manifest_path=smoke_manifest,
        benchmark_id="b1",
        real_output_dir=tmp_path / "real",
        generator_output_dirs={
            "sd15": tmp_path / "shared" / "sd15",
            "sd21": tmp_path / "generated" / "sd21",
        },
        generator_ids=["sd15", "sd21"],
        split_counts={"train": 1, "val": 0, "test": 0},
        smoke_counts={"train": 1, "val": 0, "test": 0},
        image_size=16,
        seed=7,
        caption_policy="lowest_annotation_id",
    )
    by_generator = {record.generator_id: record for record in records if record.source_kind == "generated"}
    assert by_generator["sd15"].image_path.endswith("shared/sd15/train/000000000001.png")
    assert by_generator["sd21"].image_path.endswith("generated/sd21/train/000000000001.png")
