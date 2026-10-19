from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
from tqdm import tqdm

from passive_provenance.config import load_config
from passive_provenance.datasets import load_sample_manifest
from passive_provenance.matrix import get_generation_model_spec, namespace_root, sample_manifest_path
from passive_provenance.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark images for a specific generator.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", choices=["b0", "b1"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--subset", choices=["full", "smoke"], default="smoke")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    return parser.parse_args()


def load_pipeline(config: dict[str, object], model_key: str) -> AutoPipelineForText2Image:
    spec = get_generation_model_spec(config, model_key)
    model_id = os.environ.get(f"PP_{model_key.upper()}_MODEL_ID", spec.model_id)
    dtype_name = config["generation"]["torch_dtype"]
    dtype = getattr(torch, dtype_name)
    load_kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": spec.use_safetensors,
    }
    if config["generation"]["models"].get(model_key, {}).get("local_files_only"):
        load_kwargs["local_files_only"] = True
    cache_dir = os.environ.get("HF_HOME")
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    if spec.variant:
        load_kwargs["variant"] = spec.variant
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, **load_kwargs)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.set_progress_bar_config(disable=True)
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
    return max(bounds[1] for bounds in extrema) == 0


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    records = load_sample_manifest(sample_manifest_path(config, args.benchmark, args.subset))
    records = [record for record in records if record.generator_id == args.model and record.source_kind == "generated"]
    if args.split != "all":
        records = [record for record in records if record.split == args.split]
    pipe = load_pipeline(config, args.model)
    spec = get_generation_model_spec(config, args.model)
    generated = 0
    failures: list[dict[str, object]] = []
    for record in tqdm(records, desc=f"generate-{args.benchmark}-{args.model}"):
        output_path = Path(record.image_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not is_near_black_image(output_path):
            continue
        if output_path.exists():
            output_path.unlink()
        generator = torch.Generator(device=pipe.device).manual_seed(record.seed)
        try:
            result = pipe(
                prompt=record.caption,
                width=spec.width,
                height=spec.height,
                num_inference_steps=spec.steps,
                guidance_scale=spec.guidance_scale,
                generator=generator,
            )
            image = result.images[0].convert("RGB")
            if max(bounds[1] for bounds in image.getextrema()) == 0:
                raise RuntimeError("pipeline produced an all-black image")
            image.save(output_path, format="PNG")
            generated += 1
        except Exception as exc:
            failures.append({"image_id": record.image_id, "split": record.split, "error": str(exc)})
    write_json(
        namespace_root(config, args.benchmark) / f"generation_{args.model}_{args.subset}_{args.split}.json",
        {"benchmark_id": args.benchmark, "model": args.model, "generated": generated, "failed": failures},
    )


if __name__ == "__main__":
    main()
