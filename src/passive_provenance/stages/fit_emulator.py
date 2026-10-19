from __future__ import annotations

import argparse
import json

import torch

from passive_provenance.config import load_config
from passive_provenance.matrix import (
    emulator_checkpoint_path,
    emulator_metrics_path,
    feature_cache_path,
    get_interface_spec,
    get_verifier_spec,
    interface_summary_path,
    query_transcript_path,
)
from passive_provenance.pipelines.emulator import fit_emulator
from passive_provenance.utils.io import read_csv, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a feature-side emulator from interface transcripts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--verifier-id", required=True)
    parser.add_argument("--interface-id", required=True)
    parser.add_argument("--subset", choices=["full", "smoke"], default="full")
    return parser.parse_args()


def parse_payload(raw_payload: str) -> torch.Tensor:
    payload = json.loads(raw_payload)
    if isinstance(payload, list):
        return torch.tensor(payload, dtype=torch.float32)
    if isinstance(payload, (int, float)):
        return torch.tensor([float(payload)], dtype=torch.float32)
    if payload == "abstain":
        return torch.tensor([2], dtype=torch.long)
    return torch.tensor([0], dtype=torch.float32)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    verifier_spec = get_verifier_spec(config, args.verifier_id)
    interface_spec = get_interface_spec(config, args.interface_id)
    transcript = read_csv(query_transcript_path(config, verifier_spec.benchmark_id, verifier_spec.verifier_id, interface_spec.interface_id, args.subset))
    interface_summary = read_json(interface_summary_path(config, verifier_spec.benchmark_id, verifier_spec.verifier_id, interface_spec.interface_id, args.subset))
    cache = torch.load(feature_cache_path(config, verifier_spec.benchmark_id, args.subset), map_location="cpu")
    usable_rows = [row for row in transcript if row["budget_exhausted"] == "0"]
    split_rows = {
        split: [row for row in usable_rows if row["split"] == split]
        for split in ("train", "val")
    }
    train_indices = [int(row["index_in_split"]) for row in split_rows["train"]]
    val_indices = [int(row["index_in_split"]) for row in split_rows["val"]]
    train_features = cache["splits"]["train"]["features"][train_indices]
    val_features = cache["splits"]["val"]["features"][val_indices]

    mode = "scalar"
    train_targets: list[torch.Tensor] = []
    val_targets: list[torch.Tensor] = []
    if interface_spec.kind == "full_probability_vector":
        mode = "vector"
    elif interface_spec.kind == "top1_label":
        mode = "label"
    elif interface_spec.kind == "binary_accept_reject" and interface_spec.abstention:
        mode = "ternary"
    elif interface_spec.kind == "binary_accept_reject":
        mode = "binary"

    for row in split_rows["train"]:
        payload = parse_payload(row["payload"])
        train_targets.append(payload.float() if mode not in {"label", "ternary"} else payload.long())
    for row in split_rows["val"]:
        payload = parse_payload(row["payload"])
        val_targets.append(payload.float() if mode not in {"label", "ternary"} else payload.long())
    if not train_targets or not val_targets:
        raise RuntimeError("No transcript rows available to fit the emulator.")
    train_targets_tensor = torch.stack(train_targets)
    val_targets_tensor = torch.stack(val_targets)
    emulator, metrics = fit_emulator(
        train_features=train_features,
        train_targets=train_targets_tensor,
        val_features=val_features,
        val_targets=val_targets_tensor,
        mode=mode,
        hidden_dim=int(config["emulator"]["hidden_dim"]),
        epochs=int(config["emulator"]["epochs"]),
        learning_rate=float(config["emulator"]["learning_rate"]),
    )
    checkpoint_path = emulator_checkpoint_path(config, verifier_spec.benchmark_id, verifier_spec.verifier_id, interface_spec.interface_id)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "verifier_id": verifier_spec.verifier_id,
            "benchmark_id": verifier_spec.benchmark_id,
            "interface_id": interface_spec.interface_id,
            "target_label": interface_summary["target_label"],
            "mode": mode,
            "state_dict": emulator.state_dict(),
            "input_dim": train_features.shape[1],
            "output_dim": emulator.output_dim,
            "hidden_dim": int(config["emulator"]["hidden_dim"]),
        },
        checkpoint_path,
    )
    write_json(
        emulator_metrics_path(config, verifier_spec.benchmark_id, verifier_spec.verifier_id, interface_spec.interface_id),
        {
            "benchmark_id": verifier_spec.benchmark_id,
            "verifier_id": verifier_spec.verifier_id,
            "interface_id": interface_spec.interface_id,
            "target_label": interface_summary["target_label"],
            "mode": mode,
            "emulator_eps95": metrics.emulator_eps95,
            "emulator_max_error": metrics.emulator_max_error,
            "train_loss": metrics.train_loss,
            "val_loss": metrics.val_loss,
        },
    )


if __name__ == "__main__":
    main()
