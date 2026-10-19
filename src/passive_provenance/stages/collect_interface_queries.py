from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from passive_provenance.config import load_config
from passive_provenance.datasets import classify_sample, load_sample_manifest
from passive_provenance.external_detectors import build_external_detector
from passive_provenance.interfaces import build_interface_wrapper, calibrate_abstention_margin, calibrate_binary_threshold
from passive_provenance.matrix import (
    feature_cache_path,
    get_interface_spec,
    get_verifier_spec,
    interface_summary_path,
    query_transcript_path,
    sample_manifest_path,
    verifier_checkpoint_path,
)
from passive_provenance.models.generic import load_generic_verifier_checkpoint
from passive_provenance.utils.io import write_csv, write_json


QUERY_COLUMNS = [
    "split",
    "index_in_split",
    "image_id",
    "generator_id",
    "true_label",
    "payload",
    "target_score",
    "abstained",
    "budget_exhausted",
]

VALID_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect interface transcripts for a verifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--verifier-id", required=True)
    parser.add_argument("--interface-id", required=True)
    parser.add_argument("--target-label", default="fake")
    parser.add_argument("--subset", choices=["full", "smoke"], default="full")
    parser.add_argument("--splits", default="train,val,test")
    return parser.parse_args()


def resolve_requested_splits(raw_splits: str) -> tuple[str, ...]:
    requested = tuple(split.strip() for split in raw_splits.split(",") if split.strip())
    if not requested:
        raise ValueError("At least one split must be provided.")
    invalid = sorted(set(requested) - set(VALID_SPLITS))
    if invalid:
        raise ValueError(f"Unsupported splits: {', '.join(invalid)}")
    return requested


def map_generator_ids(generator_ids: list[str], class_names: tuple[str, ...], target_label: str) -> np.ndarray:
    if target_label in class_names:
        return np.asarray([1 if generator_id == target_label else 0 for generator_id in generator_ids], dtype=np.int32)
    if target_label in {"fake", "ai", "coarse_ai"}:
        return np.asarray([0 if generator_id == "real" else 1 for generator_id in generator_ids], dtype=np.int32)
    raise ValueError(f"Unsupported target_label={target_label}")


def target_scores_from_probabilities(probabilities: np.ndarray, class_names: tuple[str, ...], target_label: str) -> np.ndarray:
    if target_label in class_names:
        return probabilities[:, class_names.index(target_label)]
    if len(class_names) == 2:
        target_index = 0 if target_label == "real" else 1
        return probabilities[:, target_index]
    return probabilities[:, 1:].sum(axis=1)


def collect_external_detector_queries(
    config: dict[str, object],
    *,
    verifier_id: str,
    interface_id: str,
    target_label: str,
    subset: str,
    splits: tuple[str, ...],
) -> None:
    verifier_spec = get_verifier_spec(config, verifier_id)
    interface_spec = get_interface_spec(config, interface_id)
    records = load_sample_manifest(sample_manifest_path(config, verifier_spec.benchmark_id, subset))
    split_records = {
        split: [
            record
            for record in records
            if record.split == split and classify_sample(record, verifier_spec.class_names) is not None
        ]
        for split in splits
    }
    if not split_records["val"]:
        raise RuntimeError(f"No validation rows available to calibrate interface={interface_id} for verifier={verifier_id}.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[collect-interface] verifier_id={verifier_id} benchmark_id={verifier_spec.benchmark_id} "
        f"interface_id={interface_id} subset={subset} device={device.type}",
        flush=True,
    )
    detector = build_external_detector(config, verifier_id, device=device)
    split_probabilities: dict[str, np.ndarray] = {}
    for split, rows in split_records.items():
        if not rows:
            split_probabilities[split] = np.zeros((0, len(verifier_spec.class_names)), dtype=np.float32)
            continue
        split_probabilities[split] = detector.predict_probabilities([record.image_path for record in rows]).detach().cpu().numpy()
    val_generator_ids = [record.generator_id for record in split_records["val"]]
    val_labels = map_generator_ids(val_generator_ids, verifier_spec.class_names, target_label)
    binary_threshold = calibrate_binary_threshold(split_probabilities["val"], val_labels, verifier_spec.class_names, target_label)
    val_target_scores = target_scores_from_probabilities(split_probabilities["val"], verifier_spec.class_names, target_label)
    abstention_margin = calibrate_abstention_margin(val_target_scores, binary_threshold, fraction=0.1) if interface_spec.abstention else 0.0
    wrapper = build_interface_wrapper(
        interface_spec,
        verifier_spec.class_names,
        target_label=target_label,
        binary_threshold=binary_threshold,
        abstention_margin=abstention_margin,
    )

    rows = []
    for split in splits:
        wrapper.reset()
        probabilities = split_probabilities[split]
        for index, record in enumerate(split_records[split]):
            reply = wrapper.query(torch.from_numpy(probabilities[index]))
            true_label = classify_sample(record, verifier_spec.class_names)
            rows.append(
                {
                    "split": split,
                    "index_in_split": index,
                    "image_id": int(record.image_id),
                    "generator_id": record.generator_id,
                    "true_label": true_label,
                    "payload": json.dumps(reply.payload),
                    "target_score": reply.target_score,
                    "abstained": int(reply.abstained),
                    "budget_exhausted": int(reply.budget_exhausted),
                }
            )
    transcript_path = query_transcript_path(config, verifier_spec.benchmark_id, verifier_spec.verifier_id, interface_spec.interface_id, subset)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(transcript_path, rows, QUERY_COLUMNS)
    write_json(
        interface_summary_path(config, verifier_spec.benchmark_id, verifier_spec.verifier_id, interface_spec.interface_id, subset),
        {
            "benchmark_id": verifier_spec.benchmark_id,
            "verifier_id": verifier_spec.verifier_id,
            "interface_id": interface_spec.interface_id,
            "target_label": target_label,
            "splits": list(splits),
            "binary_threshold": binary_threshold,
            "abstention_margin": abstention_margin,
            "rows": len(rows),
        },
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    verifier_spec = get_verifier_spec(config, args.verifier_id)
    interface_spec = get_interface_spec(config, args.interface_id)
    requested_splits = resolve_requested_splits(args.splits)
    if "val" not in requested_splits:
        raise ValueError("Validation split is required to calibrate the interface wrapper.")
    if verifier_spec.family == "external_detector":
        collect_external_detector_queries(
            config,
            verifier_id=args.verifier_id,
            interface_id=args.interface_id,
            target_label=args.target_label,
            subset=args.subset,
            splits=requested_splits,
        )
        return
    cache = torch.load(feature_cache_path(config, verifier_spec.benchmark_id, args.subset), map_location="cpu")
    verifier = load_generic_verifier_checkpoint(str(verifier_checkpoint_path(config, verifier_spec.benchmark_id, verifier_spec.verifier_id)))
    verifier.eval()

    val_features = cache["splits"]["val"]["features"]
    with torch.no_grad():
        val_probs = verifier.head_probabilities(val_features).detach().cpu().numpy()
    val_labels = map_generator_ids(cache["splits"]["val"]["generator_ids"], verifier.class_names, args.target_label)
    binary_threshold = calibrate_binary_threshold(val_probs, val_labels, verifier.class_names, args.target_label)
    if args.target_label in verifier.class_names:
        target_scores = val_probs[:, verifier.class_names.index(args.target_label)]
    else:
        target_index = 1 if len(verifier.class_names) == 2 else None
        if target_index is None:
            target_scores = val_probs[:, 1:].sum(axis=1)
        else:
            target_scores = val_probs[:, target_index]
    abstention_margin = calibrate_abstention_margin(target_scores, binary_threshold, fraction=0.1) if interface_spec.abstention else 0.0
    wrapper = build_interface_wrapper(
        interface_spec,
        verifier.class_names,
        target_label=args.target_label,
        binary_threshold=binary_threshold,
        abstention_margin=abstention_margin,
    )

    rows = []
    for split in requested_splits:
        wrapper.reset()
        features = cache["splits"][split]["features"]
        generator_ids = cache["splits"][split]["generator_ids"]
        image_ids = cache["splits"][split]["image_ids"].tolist()
        with torch.no_grad():
            probabilities = verifier.head_probabilities(features)
        for index, probs in enumerate(probabilities):
            reply = wrapper.query(probs.detach().cpu())
            true_label = generator_ids[index] if generator_ids[index] in verifier.class_names else "fake" if generator_ids[index] != "real" else "real"
            rows.append(
                {
                    "split": split,
                    "index_in_split": index,
                    "image_id": int(image_ids[index]),
                    "generator_id": generator_ids[index],
                    "true_label": true_label,
                    "payload": json.dumps(reply.payload),
                    "target_score": reply.target_score,
                    "abstained": int(reply.abstained),
                    "budget_exhausted": int(reply.budget_exhausted),
                }
            )
    transcript_path = query_transcript_path(config, verifier_spec.benchmark_id, verifier_spec.verifier_id, interface_spec.interface_id, args.subset)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(transcript_path, rows, QUERY_COLUMNS)
    write_json(
        interface_summary_path(config, verifier_spec.benchmark_id, verifier_spec.verifier_id, interface_spec.interface_id, args.subset),
        {
            "benchmark_id": verifier_spec.benchmark_id,
            "verifier_id": verifier_spec.verifier_id,
            "interface_id": interface_spec.interface_id,
            "target_label": args.target_label,
            "splits": list(requested_splits),
            "binary_threshold": binary_threshold,
            "abstention_margin": abstention_margin,
            "rows": len(rows),
        },
    )


if __name__ == "__main__":
    main()
