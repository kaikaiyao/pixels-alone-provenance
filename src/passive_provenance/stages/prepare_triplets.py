from __future__ import annotations

import argparse
import os
from pathlib import Path

from passive_provenance.config import load_config, resolve_artifact_path
from passive_provenance.datasets import prepare_triplet_manifest
from passive_provenance.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare matched COCO triplet manifests.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    artifacts_root = resolve_artifact_path(config, "dataset_manifest").parents[1]
    ensure_dir(artifacts_root)
    coco_root = os.environ.get("PP_COCO_ROOT", config["data"]["coco_root"])
    records = prepare_triplet_manifest(
        coco_root=coco_root,
        output_manifest_path=resolve_artifact_path(config, "dataset_manifest"),
        smoke_manifest_path=resolve_artifact_path(config, "smoke_manifest"),
        real_output_dir=Path(artifacts_root) / "datasets" / "triplets" / "real",
        generated_output_dir=Path(artifacts_root) / "datasets" / "triplets" / "generated",
        split_counts=config["experiment"]["split_counts"],
        smoke_counts=config["experiment"]["smoke_counts"],
        image_size=int(config["data"]["image_size"]),
        seed=int(config["experiment"]["seed"]),
        caption_policy=config["data"]["caption_policy"],
    )
    summary = {
        "num_records": len(records),
        "manifest_path": str(resolve_artifact_path(config, "dataset_manifest")),
        "smoke_manifest_path": str(resolve_artifact_path(config, "smoke_manifest")),
    }
    write_json(resolve_artifact_path(config, "dataset_manifest").parent / "prepare_triplets_summary.json", summary)


if __name__ == "__main__":
    main()
