from __future__ import annotations

import argparse

import torch
from torch import nn

from passive_provenance.config import load_config
from passive_provenance.matrix import feature_cache_path, get_verifier_spec, verifier_checkpoint_path, verifier_metrics_path
from passive_provenance.models.generic import ClipFeatureVerifier, save_generic_verifier_checkpoint
from passive_provenance.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CLIP-family verifier head from cached features.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--verifier-id", required=True)
    parser.add_argument("--subset", choices=["full", "smoke"], default="full")
    return parser.parse_args()


def map_generator_ids(generator_ids: list[str], class_names: tuple[str, ...]) -> torch.Tensor:
    label_map = {name: idx for idx, name in enumerate(class_names)}
    labels = []
    for generator_id in generator_ids:
        if generator_id in label_map:
            labels.append(label_map[generator_id])
        elif generator_id == "real" and "real" in label_map:
            labels.append(label_map["real"])
        elif generator_id != "real" and "fake" in label_map:
            labels.append(label_map["fake"])
        else:
            raise ValueError(f"Unable to map generator_id={generator_id} to class_names={class_names}")
    return torch.tensor(labels, dtype=torch.long)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    spec = get_verifier_spec(config, args.verifier_id)
    if spec.family == "external_detector":
        write_json(
            verifier_metrics_path(config, spec.benchmark_id, spec.verifier_id),
            {
                "status": "external_inference_ready",
                "benchmark_id": spec.benchmark_id,
                "verifier_id": spec.verifier_id,
                "family": spec.family,
                "head_type": spec.head_type,
                "note": "External detectors are inference-only in this pipeline. Run eval-clean to materialize clean metrics.",
            },
        )
        return
    cache = torch.load(feature_cache_path(config, spec.benchmark_id, args.subset), map_location="cpu")
    train_features = cache["splits"]["train"]["features"]
    val_features = cache["splits"]["val"]["features"]
    train_labels = map_generator_ids(cache["splits"]["train"]["generator_ids"], spec.class_names)
    val_labels = map_generator_ids(cache["splits"]["val"]["generator_ids"], spec.class_names)

    model = ClipFeatureVerifier(
        verifier_id=spec.verifier_id,
        class_names=spec.class_names,
        clip_model_name=config["d1"]["clip_model_name"],
        clip_pretrained=config["d1"]["clip_pretrained"],
        input_size=int(config["data"]["clip_input_size"]),
        mean=tuple(config["data"]["clip_mean"]),
        std=tuple(config["data"]["clip_std"]),
        head_type=spec.head_type,
        hidden_dim=int(config["emulator"]["hidden_dim"]),
    )
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=float(config["d1"]["learning_rate"]), weight_decay=float(config["d1"]["weight_decay"]))
    criterion = nn.CrossEntropyLoss()
    if spec.head_type == "binary_logistic":
        criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_val = -1.0
    epochs = int(config["d1"]["epochs"])
    for _ in range(epochs):
        logits = model.head(train_features)
        if spec.head_type == "binary_logistic":
            loss = criterion(logits, train_labels.float())
        else:
            loss = criterion(logits, train_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            probabilities = model.head_probabilities(val_features)
            predictions = probabilities.argmax(dim=1)
            val_acc = float((predictions == val_labels).float().mean().item())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu() for key, value in model.head.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Verifier training did not produce a validation checkpoint.")
    model.head.load_state_dict(best_state)
    checkpoint_path = verifier_checkpoint_path(config, spec.benchmark_id, spec.verifier_id)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_generic_verifier_checkpoint(
        model,
        str(checkpoint_path),
        benchmark_id=spec.benchmark_id,
        family=spec.family,
        hidden_dim=int(config["emulator"]["hidden_dim"]),
    )
    write_json(
        verifier_metrics_path(config, spec.benchmark_id, spec.verifier_id),
        {
            "status": "trained",
            "benchmark_id": spec.benchmark_id,
            "verifier_id": spec.verifier_id,
            "best_val_accuracy": best_val,
        },
    )


if __name__ == "__main__":
    main()
