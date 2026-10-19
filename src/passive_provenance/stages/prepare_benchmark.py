from __future__ import annotations

import argparse
import os

from passive_provenance.config import load_config
from passive_provenance.datasets import export_triplet_manifest, prepare_sample_manifest
from passive_provenance.matrix import (
    benchmark_generated_root,
    benchmark_generated_root_for_generator,
    benchmark_real_root,
    get_benchmark_spec,
    legacy_triplet_manifest_path,
    namespace_root,
    paper_matrix_root,
    sample_manifest_path,
)
from passive_provenance.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare benchmark manifests for B0/B1.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", choices=["b0", "b1"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    benchmark = get_benchmark_spec(config, args.benchmark)
    coco_root = os.environ.get("PP_COCO_ROOT", config["data"]["coco_root"])
    full_manifest = sample_manifest_path(config, benchmark.benchmark_id, "full")
    smoke_manifest = sample_manifest_path(config, benchmark.benchmark_id, "smoke")
    ensure_dir(full_manifest.parent)
    records = prepare_sample_manifest(
        coco_root=coco_root,
        output_manifest_path=full_manifest,
        smoke_manifest_path=smoke_manifest,
        benchmark_id=benchmark.benchmark_id,
        real_output_dir=benchmark_real_root(config, benchmark.benchmark_id),
        generator_output_dirs={
            generator_id: benchmark_generated_root_for_generator(config, benchmark.benchmark_id, generator_id)
            for generator_id in benchmark.generator_ids
        },
        generator_ids=benchmark.generator_ids,
        split_counts=config["experiment"]["split_counts"],
        smoke_counts=config["experiment"]["smoke_counts"],
        image_size=int(config["data"]["image_size"]),
        seed=int(config["experiment"]["seed"]),
        caption_policy=config["data"]["caption_policy"],
    )
    legacy_paths = {}
    if benchmark.legacy_triplet_export:
        legacy_full = legacy_triplet_manifest_path(config, benchmark.benchmark_id, "full")
        legacy_smoke = legacy_triplet_manifest_path(config, benchmark.benchmark_id, "smoke")
        export_triplet_manifest(records, legacy_full, legacy_smoke, config["experiment"]["smoke_counts"])
        legacy_paths = {
            "legacy_full_triplet_manifest": str(legacy_full),
            "legacy_smoke_triplet_manifest": str(legacy_smoke),
        }
    summary = {
        "benchmark_id": benchmark.benchmark_id,
        "label_space": list(benchmark.label_space),
        "generator_ids": list(benchmark.generator_ids),
        "full_sample_manifest": str(full_manifest),
        "smoke_sample_manifest": str(smoke_manifest),
        "num_samples": len(records),
        **legacy_paths,
    }
    write_json(namespace_root(config, benchmark.benchmark_id) / "prepare_benchmark_summary.json", summary)
    write_json(paper_matrix_root(config) / f"prepare_{benchmark.benchmark_id}.json", summary)


if __name__ == "__main__":
    main()
