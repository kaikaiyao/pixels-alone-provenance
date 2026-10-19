from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image
from sklearn import metrics as sk_metrics

from passive_provenance.config import load_config, resolve_artifact_path, resolve_artifact_relative
from passive_provenance.constants import RESULT_COLUMNS
from passive_provenance.datasets import classify_sample, load_sample_manifest, load_triplet_manifest
from passive_provenance.external_detectors import build_external_detector
from passive_provenance.matrix import (
    feature_cache_path,
    get_verifier_spec,
    sample_manifest_path,
    verifier_checkpoint_path,
    verifier_metrics_path,
    verifier_predictions_path,
)
from passive_provenance.models.d1 import FrozenClipVerifier
from passive_provenance.models.generic import load_generic_verifier_checkpoint
from passive_provenance.utils.io import write_csv, write_json
from passive_provenance.utils.metrics import compute_multiclass_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate clean D1 metrics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--subset", choices=["full", "smoke"], default="smoke")
    parser.add_argument("--verifier-id")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--metrics-path")
    parser.add_argument("--predictions-path")
    parser.add_argument("--detector-name", default="d1_clip3")
    return parser.parse_args()


def image_tensor(path: str) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def output_path(config: dict[str, object], override: str | None, default_key: str):
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


def select_binary_threshold(labels: np.ndarray, fake_scores: np.ndarray) -> float:
    fpr, tpr, thresholds = sk_metrics.roc_curve(labels, fake_scores)
    best_index = int(np.argmax(tpr - fpr))
    return float(thresholds[best_index])


def tpr_at_fpr(labels: np.ndarray, fake_scores: np.ndarray, max_fpr: float) -> float:
    fpr, tpr, _ = sk_metrics.roc_curve(labels, fake_scores)
    eligible = np.where(fpr <= max_fpr)[0]
    if eligible.size == 0:
        return 0.0
    return float(tpr[eligible[-1]])


def equal_error_rate(labels: np.ndarray, fake_scores: np.ndarray) -> float:
    fpr, tpr, _ = sk_metrics.roc_curve(labels, fake_scores)
    fnr = 1.0 - tpr
    best_index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[best_index] + fnr[best_index]) / 2.0)


def external_clean_eval(config: dict[str, object], subset: str, verifier_id: str) -> None:
    verifier_spec = get_verifier_spec(config, verifier_id)
    records = load_sample_manifest(sample_manifest_path(config, verifier_spec.benchmark_id, subset))
    split_records = [record for record in records if record.split == "test" and classify_sample(record, verifier_spec.class_names) is not None]
    if not split_records:
        raise RuntimeError(f"No test records found for external detector {verifier_id}.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[eval-clean] verifier_id={verifier_id} benchmark_id={verifier_spec.benchmark_id} "
        f"subset={subset} test_images={len(split_records)} device={device.type}",
        flush=True,
    )
    detector = build_external_detector(config, verifier_id, device=device)
    probabilities = detector.predict_probabilities([record.image_path for record in split_records]).detach().cpu().numpy()
    label_map = {name: idx for idx, name in enumerate(verifier_spec.class_names)}
    labels = []
    rows = []
    for index, record in enumerate(split_records):
        label_name = classify_sample(record, verifier_spec.class_names)
        if label_name is None:
            continue
        label_index = label_map[label_name]
        labels.append(label_index)
        prediction = int(np.argmax(probabilities[index]))
        rows.append(
            {
                "benchmark_id": verifier_spec.benchmark_id,
                "verifier_id": verifier_id,
                "interface_id": "full_vector",
                "image_id": int(record.image_id),
                "split": "test",
                "source": record.generator_id,
                "target": "",
                "detector": verifier_id,
                "attack": "clean",
                "eps": "",
                "pi": "",
                "seed": record.seed,
                "clean_target_score": "",
                "adv_target_score": "",
                "clean_pred": prediction,
                "adv_pred": prediction,
                "success": int(prediction == label_index),
                "linf": 0.0,
                "psnr": "",
                "lpips": "",
                "emulator_eps95": "",
                "emulator_max_error": "",
                "alpha_estimate": "",
                "recovered_whitebox_gap": "",
                "query_count": "",
                "query_budget": "",
                "optimizer": "",
                "abstained": "",
                "rate_limited": "",
            }
        )
    labels_array = np.asarray(labels, dtype=np.int32)
    predictions = np.asarray([int(np.argmax(item)) for item in probabilities], dtype=np.int32)
    metrics = compute_multiclass_metrics(labels_array, predictions, probabilities)
    fake_labels = (labels_array != 0).astype(np.int32)
    fake_scores = probabilities[:, 1:].sum(axis=1)
    metrics["binary_threshold"] = select_binary_threshold(fake_labels, fake_scores)
    metrics["binary_tpr_at_5_fpr"] = tpr_at_fpr(fake_labels, fake_scores, max_fpr=0.05)
    metrics["binary_eer"] = equal_error_rate(fake_labels, fake_scores)
    metrics["benchmark_id"] = verifier_spec.benchmark_id
    metrics["verifier_id"] = verifier_id
    metrics["family"] = verifier_spec.family
    metrics["head_type"] = verifier_spec.head_type
    metrics["status"] = "clean_evaluated"
    detector_metadata = detector.metadata()
    metrics["detector_backend"] = detector_metadata.backend_name
    metrics.update(detector_metadata.extras)
    write_json(verifier_metrics_path(config, verifier_spec.benchmark_id, verifier_id), metrics)
    write_csv(verifier_predictions_path(config, verifier_spec.benchmark_id, verifier_id), rows, RESULT_COLUMNS)


def generic_clean_eval(config: dict[str, object], subset: str, verifier_id: str) -> None:
    verifier_spec = get_verifier_spec(config, verifier_id)
    if verifier_spec.family == "external_detector":
        external_clean_eval(config, subset, verifier_id)
        return
    cache = torch.load(feature_cache_path(config, verifier_spec.benchmark_id, subset), map_location="cpu")
    model = load_generic_verifier_checkpoint(str(verifier_checkpoint_path(config, verifier_spec.benchmark_id, verifier_id)))
    test_features = cache["splits"]["test"]["features"]
    generator_ids = cache["splits"]["test"]["generator_ids"]
    image_ids = cache["splits"]["test"]["image_ids"].tolist()
    with torch.no_grad():
        probabilities = model.head_probabilities(test_features).detach().cpu().numpy()
    label_map = {name: idx for idx, name in enumerate(model.class_names)}
    labels = []
    rows = []
    for index, generator_id in enumerate(generator_ids):
        if generator_id in label_map:
            label_index = label_map[generator_id]
            source_name = generator_id
        elif generator_id == "real" and "real" in label_map:
            label_index = label_map["real"]
            source_name = "real"
        elif generator_id != "real" and "fake" in label_map:
            label_index = label_map["fake"]
            source_name = generator_id
        else:
            raise ValueError(f"Unable to map generator_id={generator_id} into {model.class_names}")
        labels.append(label_index)
        prediction = int(np.argmax(probabilities[index]))
        rows.append(
            {
                "benchmark_id": verifier_spec.benchmark_id,
                "verifier_id": verifier_id,
                "interface_id": "full_vector",
                "image_id": int(image_ids[index]),
                "split": "test",
                "source": source_name,
                "target": "",
                "detector": verifier_id,
                "attack": "clean",
                "eps": "",
                "pi": "",
                "seed": "",
                "clean_target_score": "",
                "adv_target_score": "",
                "clean_pred": prediction,
                "adv_pred": prediction,
                "success": int(prediction == label_index),
                "linf": 0.0,
                "psnr": "",
                "lpips": "",
                "emulator_eps95": "",
                "emulator_max_error": "",
                "alpha_estimate": "",
                "recovered_whitebox_gap": "",
                "query_count": "",
                "query_budget": "",
                "optimizer": "",
                "abstained": "",
                "rate_limited": "",
            }
        )
    labels_array = np.asarray(labels, dtype=np.int32)
    predictions = np.asarray([int(np.argmax(item)) for item in probabilities], dtype=np.int32)
    metrics = compute_multiclass_metrics(labels_array, predictions, probabilities)
    fake_labels = (labels_array != 0).astype(np.int32)
    fake_scores = probabilities[:, 1:].sum(axis=1)
    metrics["binary_threshold"] = select_binary_threshold(fake_labels, fake_scores)
    metrics["binary_tpr_at_5_fpr"] = tpr_at_fpr(fake_labels, fake_scores, max_fpr=0.05)
    metrics["binary_eer"] = equal_error_rate(fake_labels, fake_scores)
    metrics["benchmark_id"] = verifier_spec.benchmark_id
    metrics["verifier_id"] = verifier_id
    write_json(verifier_metrics_path(config, verifier_spec.benchmark_id, verifier_id), metrics)
    write_csv(verifier_predictions_path(config, verifier_spec.benchmark_id, verifier_id), rows, RESULT_COLUMNS)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.verifier_id:
        generic_clean_eval(config, args.subset, args.verifier_id)
        return
    manifest_key = "smoke_manifest" if args.subset == "smoke" else "dataset_manifest"
    records = load_triplet_manifest(resolve_artifact_path(config, manifest_key))
    model = load_checkpoint(config, args.checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    rows = []
    split_payload: dict[str, dict[str, list[object]]] = {
        "val": {"labels": [], "probabilities": [], "image_ids": []},
        "test": {"labels": [], "probabilities": [], "image_ids": []},
    }
    for record in records:
        if record.split not in {"val", "test"}:
            continue
        for label_name, label_index, image_path in [("real", 0, record.real_path), ("sd15", 1, record.sd15_path), ("sdxl", 2, record.sdxl_path)]:
            image = image_tensor(image_path).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = model.forward_probs(image)[0].detach().cpu().numpy()
            prediction = int(np.argmax(probs))
            split_payload[record.split]["labels"].append(label_index)
            split_payload[record.split]["probabilities"].append(probs)
            split_payload[record.split]["image_ids"].append(record.image_id)
            if record.split == "test":
                rows.append(
                    {
                        "image_id": record.image_id,
                        "split": "test",
                        "source": label_name,
                        "target": "",
                        "detector": args.detector_name,
                        "attack": "clean",
                        "eps": "",
                        "pi": "",
                        "seed": record.seed,
                        "clean_target_score": "",
                        "adv_target_score": "",
                        "clean_pred": prediction,
                        "adv_pred": prediction,
                        "success": int(prediction == label_index),
                        "linf": 0.0,
                        "psnr": "",
                        "lpips": "",
                        "emulator_eps95": "",
                        "alpha_estimate": "",
                        "query_count": "",
                    }
                )
    val_labels = np.asarray(split_payload["val"]["labels"], dtype=np.int32)
    val_probabilities = np.asarray(split_payload["val"]["probabilities"], dtype=np.float32)
    val_fake_labels = (val_labels != 0).astype(np.int32)
    val_fake_scores = val_probabilities[:, 1] + val_probabilities[:, 2]
    binary_threshold = select_binary_threshold(val_fake_labels, val_fake_scores)

    test_labels = np.asarray(split_payload["test"]["labels"], dtype=np.int32)
    test_probabilities = np.asarray(split_payload["test"]["probabilities"], dtype=np.float32)
    test_predictions = np.asarray([int(np.argmax(item)) for item in test_probabilities], dtype=np.int32)
    metrics = compute_multiclass_metrics(test_labels, test_predictions, test_probabilities)
    test_fake_labels = (test_labels != 0).astype(np.int32)
    test_fake_scores = test_probabilities[:, 1] + test_probabilities[:, 2]
    metrics["binary_threshold"] = binary_threshold
    metrics["binary_tpr_at_5_fpr"] = tpr_at_fpr(test_fake_labels, test_fake_scores, max_fpr=0.05)
    metrics["binary_eer"] = equal_error_rate(test_fake_labels, test_fake_scores)
    metrics["detector"] = args.detector_name
    write_json(output_path(config, args.metrics_path, "clean_metrics"), metrics)
    write_csv(output_path(config, args.predictions_path, "clean_predictions"), rows, RESULT_COLUMNS)


if __name__ == "__main__":
    main()
