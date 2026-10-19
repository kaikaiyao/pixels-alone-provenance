from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
from tqdm import tqdm

from passive_provenance.config import load_config, resolve_artifact_path
from passive_provenance.datasets import load_triplet_manifest
from passive_provenance.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SD1.5 or SDXL images for the triplet manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["sd15", "sdxl"], required=True)
    parser.add_argument("--subset", choices=["full", "smoke"], default="smoke")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    return parser.parse_args()


def load_pipeline(config: dict[str, object], model_key: str) -> AutoPipelineForText2Image:
    model_id = os.environ.get(f"PP_{model_key.upper()}_MODEL_ID", config["generation"][f"{model_key}_model_id"])
    dtype_name = config["generation"]["torch_dtype"]
    dtype = getattr(torch, dtype_name)
    use_safetensors = bool(config["generation"].get(f"{model_key}_use_safetensors", True))
    load_kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": use_safetensors,
    }
    cache_dir = os.environ.get("HF_HOME")
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    variant = config["generation"].get(f"{model_key}_variant")
    if variant:
        load_kwargs["variant"] = variant
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, **load_kwargs)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.set_progress_bar_config(disable=True)
    # Safety-checker black images create shortcut leakage for detector experiments.
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    if hasattr(pipe, "watermarker"):
        pipe.watermarker = None
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    return pipe


def is_near_black_image(path: Path) -> bool:
    if not path.exists():
        return False
    image = Image.open(path).convert("RGB")
    extrema = image.getextrema()
    channel_max = max(bounds[1] for bounds in extrema)
    if channel_max == 0:
        return True
    return False


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest_key = "smoke_manifest" if args.subset == "smoke" else "dataset_manifest"
    records = load_triplet_manifest(resolve_artifact_path(config, manifest_key))
    if args.split != "all":
        records = [record for record in records if record.split == args.split]
    pipe = load_pipeline(config, args.model)
    generated = 0
    failures: list[dict[str, object]] = []
    steps = int(config["generation"][f"{args.model}_steps"])
    guidance_scale = float(config["generation"][f"{args.model}_guidance_scale"])
    width = int(config["generation"]["width"])
    height = int(config["generation"]["height"])
    for record in tqdm(records, desc=f"generate-{args.model}"):
        output_path = Path(getattr(record, f"{args.model}_path"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not is_near_black_image(output_path):
            continue
        if output_path.exists():
            output_path.unlink()
        generator = torch.Generator(device=pipe.device).manual_seed(record.seed)
        try:
            result = pipe(
                prompt=record.caption,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            image = result.images[0].convert("RGB")
            clean = Image.new("RGB", image.size)
            clean.putdata(list(image.getdata()))
            if max(bounds[1] for bounds in clean.getextrema()) == 0:
                raise RuntimeError("pipeline produced an all-black image")
            clean.save(output_path, format="PNG")
            generated += 1
        except Exception as exc:
            failures.append({"image_id": record.image_id, "error": str(exc)})
    write_json(
        resolve_artifact_path(config, "dataset_manifest").parent / f"generation_{args.model}_{args.subset}.json",
        {"generated": generated, "failed": failures},
    )


if __name__ == "__main__":
    main()
