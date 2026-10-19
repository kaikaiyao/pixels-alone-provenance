from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from PIL import Image

from passive_provenance.config import load_config
from passive_provenance.datasets import SampleRecord
from passive_provenance.external_detectors import (
    ExternalDetectorAdapter,
    ExternalDetectorMetadata,
    build_external_detector,
    canonical_download_url,
    is_html_like_file,
)
from passive_provenance.matrix import sample_manifest_path, verifier_metrics_path, verifier_predictions_path
from passive_provenance.matrix import interface_summary_path, query_transcript_path
from passive_provenance.stages.collect_interface_queries import collect_external_detector_queries, resolve_requested_splits
from passive_provenance.stages.eval_clean import external_clean_eval
from passive_provenance.utils.io import read_csv, read_json, write_csv


class DummyExternalDetector(ExternalDetectorAdapter):
    def __init__(self) -> None:
        super().__init__("clip_detector", ("real", "fake"))

    def predict_probabilities(self, image_paths: list[str]) -> torch.Tensor:
        rows = []
        for path in image_paths:
            if "real" in path:
                rows.append([0.9, 0.1])
            else:
                rows.append([0.1, 0.9])
        return torch.tensor(rows, dtype=torch.float32)

    def predict_tensor_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        rows = []
        for image in images:
            fake_score = float(image.mean().item())
            rows.append([1.0 - fake_score, fake_score])
        return torch.tensor(rows, dtype=torch.float32)

    def metadata(self) -> ExternalDetectorMetadata:
        return ExternalDetectorMetadata(
            backend_name="dummy_external_detector",
            class_names=self.class_names,
            extras={"test_stub": True},
        )


def test_external_clean_eval_writes_clean_metrics_and_predictions(tmp_path, monkeypatch):
    config = load_config("configs/default.yaml")
    config["paths"]["workspace_root"] = str(tmp_path)

    manifest = sample_manifest_path(config, "b1", "full")
    manifest.parent.mkdir(parents=True, exist_ok=True)

    real_path = tmp_path / "images" / "real_test.png"
    fake_path = tmp_path / "images" / "sd15_test.png"
    real_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(real_path)
    Image.new("RGB", (8, 8), color=(78, 90, 12)).save(fake_path)

    records = [
        SampleRecord("b1", 1, 1, "caption", "test", "real", "real", 11, str(real_path), 8, 8, "hash"),
        SampleRecord("b1", 1, 1, "caption", "test", "generated", "sd15", 11, str(fake_path), 8, 8, "hash"),
    ]
    write_csv(manifest, [asdict(record) for record in records], list(SampleRecord.__dataclass_fields__.keys()))

    monkeypatch.setattr(
        "passive_provenance.stages.eval_clean.build_external_detector",
        lambda config, verifier_id, device=None: DummyExternalDetector(),
    )

    external_clean_eval(config, "full", "clip_detector")

    metrics = read_json(verifier_metrics_path(config, "b1", "clip_detector"))
    assert metrics["status"] == "clean_evaluated"
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["fake_auc"] == pytest.approx(1.0)
    assert metrics["detector_backend"] == "dummy_external_detector"

    rows = read_csv(verifier_predictions_path(config, "b1", "clip_detector"))
    assert len(rows) == 2
    assert {row["source"] for row in rows} == {"real", "sd15"}


def test_collect_external_detector_queries_writes_transcript_and_summary(tmp_path, monkeypatch):
    config = load_config("configs/default.yaml")
    config["paths"]["workspace_root"] = str(tmp_path)

    manifest = sample_manifest_path(config, "b1", "full")
    manifest.parent.mkdir(parents=True, exist_ok=True)

    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    split_specs = [
        ("train", 1, "real"),
        ("train", 2, "sd15"),
        ("val", 3, "real"),
        ("val", 4, "sd15"),
        ("test", 5, "real"),
        ("test", 6, "sd15"),
    ]
    for split, image_id, generator_id in split_specs:
        image_path = image_dir / f"{split}_{generator_id}.png"
        Image.new("RGB", (8, 8), color=(12 + image_id, 34, 56)).save(image_path)
        rows.append(
            asdict(
                SampleRecord(
                    "b1",
                    image_id,
                    image_id,
                    "caption",
                    split,
                    "real" if generator_id == "real" else "generated",
                    generator_id,
                    image_id,
                    str(image_path),
                    8,
                    8,
                    "hash",
                )
            )
        )
    write_csv(manifest, rows, list(SampleRecord.__dataclass_fields__.keys()))

    monkeypatch.setattr(
        "passive_provenance.stages.collect_interface_queries.build_external_detector",
        lambda config, verifier_id, device=None: DummyExternalDetector(),
    )

    collect_external_detector_queries(
        config,
        verifier_id="clip_detector",
        interface_id="binary",
        target_label="fake",
        subset="full",
        splits=("train", "val", "test"),
    )

    transcript = read_csv(query_transcript_path(config, "b1", "clip_detector", "binary", "full"))
    assert len(transcript) == 6
    assert {row["split"] for row in transcript} == {"train", "val", "test"}

    summary = read_json(interface_summary_path(config, "b1", "clip_detector", "binary", "full"))
    assert summary["verifier_id"] == "clip_detector"
    assert summary["interface_id"] == "binary"
    assert summary["splits"] == ["train", "val", "test"]
    assert summary["rows"] == 6


def test_collect_external_detector_queries_respects_requested_splits(tmp_path, monkeypatch):
    config = load_config("configs/default.yaml")
    config["paths"]["workspace_root"] = str(tmp_path)

    manifest = sample_manifest_path(config, "b1", "full")
    manifest.parent.mkdir(parents=True, exist_ok=True)

    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    split_specs = [
        ("train", 1, "real"),
        ("train", 2, "sd15"),
        ("val", 3, "real"),
        ("val", 4, "sd15"),
        ("test", 5, "real"),
        ("test", 6, "sd15"),
    ]
    for split, image_id, generator_id in split_specs:
        image_path = image_dir / f"{split}_{generator_id}.png"
        Image.new("RGB", (8, 8), color=(12 + image_id, 34, 56)).save(image_path)
        rows.append(
            asdict(
                SampleRecord(
                    "b1",
                    image_id,
                    image_id,
                    "caption",
                    split,
                    "real" if generator_id == "real" else "generated",
                    generator_id,
                    image_id,
                    str(image_path),
                    8,
                    8,
                    "hash",
                )
            )
        )
    write_csv(manifest, rows, list(SampleRecord.__dataclass_fields__.keys()))

    monkeypatch.setattr(
        "passive_provenance.stages.collect_interface_queries.build_external_detector",
        lambda config, verifier_id, device=None: DummyExternalDetector(),
    )

    collect_external_detector_queries(
        config,
        verifier_id="clip_detector",
        interface_id="binary",
        target_label="fake",
        subset="full",
        splits=("train", "val"),
    )

    transcript = read_csv(query_transcript_path(config, "b1", "clip_detector", "binary", "full"))
    assert len(transcript) == 4
    assert {row["split"] for row in transcript} == {"train", "val"}

    summary = read_json(interface_summary_path(config, "b1", "clip_detector", "binary", "full"))
    assert summary["splits"] == ["train", "val"]
    assert summary["rows"] == 4


def test_resolve_requested_splits_validates_inputs():
    assert resolve_requested_splits("train,val") == ("train", "val")
    with pytest.raises(ValueError):
        resolve_requested_splits("")
    with pytest.raises(ValueError):
        resolve_requested_splits("train,dev")


def test_build_external_detector_constructs_cnnspot_from_official_weight(monkeypatch, tmp_path):
    config = load_config("configs/default.yaml")
    config["paths"]["workspace_root"] = str(tmp_path)
    captured = {}

    class DummyCnnAdapter:
        def __init__(self, verifier_id, class_names, *, weight_path, weight_url, batch_size, device, resize_size, crop_size):
            captured["verifier_id"] = verifier_id
            captured["class_names"] = class_names
            captured["weight_path"] = weight_path
            captured["weight_url"] = weight_url
            captured["batch_size"] = batch_size
            captured["device"] = device
            captured["resize_size"] = resize_size
            captured["crop_size"] = crop_size

    monkeypatch.setattr("passive_provenance.external_detectors.CnnDetectionAdapter", DummyCnnAdapter)

    build_external_detector(config, "cnnspot", device="cpu")

    assert captured["verifier_id"] == "cnnspot"
    assert captured["class_names"] == ("real", "fake")
    assert str(captured["weight_path"]).endswith("artifacts/external_models/cnnspot/blur_jpg_prob0.5.pth")
    assert "dropbox.com" in captured["weight_url"]
    assert captured["batch_size"] == 32
    assert captured["resize_size"] == 256
    assert captured["crop_size"] == 224


def test_build_external_detector_constructs_dire_from_config(monkeypatch, tmp_path):
    config = load_config("configs/default.yaml")
    config["paths"]["workspace_root"] = str(tmp_path)
    captured = {}

    class DummyDireAdapter:
        def __init__(
            self,
            verifier_id,
            class_names,
            *,
            weight_path,
            weight_url,
            arch,
            batch_size,
            device,
            resize_size,
            crop_size,
            aug_norm,
        ):
            captured["verifier_id"] = verifier_id
            captured["class_names"] = class_names
            captured["weight_path"] = weight_path
            captured["weight_url"] = weight_url
            captured["arch"] = arch
            captured["batch_size"] = batch_size
            captured["device"] = device
            captured["resize_size"] = resize_size
            captured["crop_size"] = crop_size
            captured["aug_norm"] = aug_norm

    monkeypatch.setattr("passive_provenance.external_detectors.DireDetectionAdapter", DummyDireAdapter)

    build_external_detector(config, "dire", device="cpu")

    assert captured["verifier_id"] == "dire"
    assert captured["class_names"] == ("real", "fake")
    assert str(captured["weight_path"]).endswith("artifacts/external_models/dire/lsun_adm.pth")
    assert captured["weight_url"] == ""
    assert captured["arch"] == "resnet50"
    assert captured["batch_size"] == 32
    assert captured["resize_size"] == 256
    assert captured["crop_size"] == 224
    assert captured["aug_norm"] is True


def test_dropbox_weight_urls_are_canonicalized_and_bad_cache_is_detected(tmp_path):
    assert canonical_download_url("https://www.dropbox.com/s/foo/bar.pth?dl=0").endswith("?dl=1")
    path = tmp_path / "bad_download.pth"
    path.write_text("<html>not a checkpoint</html>", encoding="utf-8")
    assert is_html_like_file(path) is True


def test_dire_adapter_fails_honestly_without_weights(tmp_path):
    config = load_config("configs/default.yaml")
    config["paths"]["workspace_root"] = str(tmp_path)
    with pytest.raises(RuntimeError, match="Missing external-detector weights"):
        build_external_detector(config, "dire", device="cpu")
