from __future__ import annotations

from pathlib import Path

from passive_provenance.datasets import SampleRecord, classify_sample, export_triplet_manifest
from passive_provenance.matrix import (
    benchmark_generated_root_for_generator,
    feature_cache_path,
    get_benchmark_spec,
    get_interface_spec,
    get_verifier_spec,
    sample_manifest_path,
)


def test_b1_benchmark_registry_has_expected_generator_set():
    from passive_provenance.config import load_config

    config = load_config("configs/default.yaml")
    spec = get_benchmark_spec(config, "b1")
    assert spec.label_space == ("real", "sd15", "sdxl", "sd21", "pixart")
    assert spec.generator_ids == ("sd15", "sdxl", "sd21", "pixart")
    assert spec.generator_benchmark_overrides == {"sd15": "b0", "sdxl": "b0"}


def test_feature_cache_and_manifest_paths_are_subset_scoped():
    from passive_provenance.config import load_config

    config = load_config("configs/default.yaml")
    assert feature_cache_path(config, "b1", "smoke").name == "feature_cache_smoke.pt"
    assert feature_cache_path(config, "b1", "full").name == "feature_cache_full.pt"
    assert sample_manifest_path(config, "b1", "full").name == "sample_manifest_full.csv"


def test_b1_shared_generators_reuse_b0_paths():
    from passive_provenance.config import load_config

    config = load_config("configs/default.yaml")
    assert str(benchmark_generated_root_for_generator(config, "b1", "sd15")).endswith("artifacts/datasets/triplets/generated/sd15")
    assert str(benchmark_generated_root_for_generator(config, "b1", "sd21")).endswith("artifacts/datasets/benchmarks/b1/generated/sd21")


def test_sample_classification_supports_fake_collapse():
    record = SampleRecord(
        benchmark_id="b1",
        prompt_id=1,
        image_id=1,
        caption="caption",
        split="test",
        source_kind="generated",
        generator_id="pixart",
        seed=7,
        image_path="/tmp/image.png",
        width=512,
        height=512,
        preprocess_hash="hash",
    )
    assert classify_sample(record, ("real", "fake")) == "fake"


def test_export_triplet_manifest_uses_b0_records(tmp_path: Path):
    rows = [
        SampleRecord("b0", 1, 1, "caption", "train", "real", "real", 7, "/tmp/real.png", 512, 512, "hash"),
        SampleRecord("b0", 1, 1, "caption", "train", "generated", "sd15", 7, "/tmp/sd15.png", 512, 512, "hash"),
        SampleRecord("b0", 1, 1, "caption", "train", "generated", "sdxl", 7, "/tmp/sdxl.png", 512, 512, "hash"),
    ]
    full_path = tmp_path / "dataset_manifest.csv"
    smoke_path = tmp_path / "dataset_manifest_smoke.csv"
    triplets = export_triplet_manifest(rows, full_path, smoke_path, {"train": 1, "val": 0, "test": 0})
    assert len(triplets) == 1
    assert full_path.exists()


def test_interface_and_verifier_registries_load():
    from passive_provenance.config import load_config

    config = load_config("configs/default.yaml")
    verifier = get_verifier_spec(config, "b0_mlp")
    interface = get_interface_spec(config, "binary_budget_512_abstain")
    assert verifier.head_type == "mlp_gelu"
    assert interface.rate_limit == 512
    assert interface.abstention is True
