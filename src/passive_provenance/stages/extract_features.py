from __future__ import annotations

import argparse

import torch

from passive_provenance.config import load_config
from passive_provenance.datasets import load_sample_manifest
from passive_provenance.matrix import feature_cache_path, get_benchmark_spec, sample_manifest_path
from passive_provenance.models.generic import ClipFeatureVerifier
from passive_provenance.pipelines.feature_cache import build_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and cache frozen CLIP features for a benchmark.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", choices=["b0", "b1"], required=True)
    parser.add_argument("--subset", choices=["full", "smoke"], default="smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    benchmark = get_benchmark_spec(config, args.benchmark)
    records = load_sample_manifest(sample_manifest_path(config, benchmark.benchmark_id, args.subset))
    model = ClipFeatureVerifier(
        verifier_id=f"{benchmark.benchmark_id}_feature_backbone",
        class_names=benchmark.label_space,
        clip_model_name=config["d1"]["clip_model_name"],
        clip_pretrained=config["d1"]["clip_pretrained"],
        input_size=int(config["data"]["clip_input_size"]),
        mean=tuple(config["data"]["clip_mean"]),
        std=tuple(config["data"]["clip_std"]),
        head_type="softmax_linear",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    cache = build_feature_cache(
        model,
        records,
        class_names=benchmark.label_space,
        batch_size=int(config["feature_cache"]["batch_size"]),
        num_workers=int(config["feature_cache"].get("num_workers", 0)),
        pin_memory=bool(config["feature_cache"].get("pin_memory", False)),
        device=device,
    )
    payload = {
        "benchmark_id": benchmark.benchmark_id,
        "class_names": benchmark.label_space,
        "clip_model_name": config["d1"]["clip_model_name"],
        "clip_pretrained": config["d1"]["clip_pretrained"],
        "splits": cache,
    }
    path = feature_cache_path(config, benchmark.benchmark_id, args.subset)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


if __name__ == "__main__":
    main()
