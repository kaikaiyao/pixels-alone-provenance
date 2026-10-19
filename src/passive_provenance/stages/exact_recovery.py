from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from passive_provenance.attacks.pgd import pgd_targeted, score_from_probs
from passive_provenance.config import load_config, resolve_artifact_path
from passive_provenance.constants import RESULT_COLUMNS
from passive_provenance.datasets import load_triplet_manifest
from passive_provenance.models.d1 import FrozenClipVerifier
from passive_provenance.pipelines.exact_recovery import recover_softmax_relative_head, relative_parameter_error, score_error
from passive_provenance.utils.io import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover the D1 linear head from score queries and compare attacks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--subset", choices=["full", "smoke"], default="smoke")
    return parser.parse_args()


def load_checkpoint(config: dict[str, object]) -> FrozenClipVerifier:
    checkpoint_path = resolve_artifact_path(config, "d1_checkpoint")
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


def image_tensor(path: str) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def build_probe_bank(records) -> list[str]:
    paths = []
    for record in records:
        paths.extend([record.real_path, record.sd15_path, record.sdxl_path])
    return paths


def encode_probe_bank(
    model: FrozenClipVerifier,
    probe_paths: list[str],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    names: list[str] = []
    feature_chunks: list[torch.Tensor] = []
    probability_chunks: list[torch.Tensor] = []
    for start in tqdm(range(0, len(probe_paths), batch_size), desc="encoding probe bank"):
        batch_paths = probe_paths[start : start + batch_size]
        images = torch.stack([image_tensor(path) for path in batch_paths]).to(device)
        variants = [
            (batch_paths, images),
            ([f"{path}#flip" for path in batch_paths], torch.flip(images, dims=[3])),
        ]
        for variant_names, variant_images in variants:
            with torch.no_grad():
                features = model.encode(variant_images)
                probabilities = model.head.probabilities(features)
            names.extend(variant_names)
            feature_chunks.append(features.detach().cpu())
            probability_chunks.append(probabilities.detach().cpu())
        del images
    return names, torch.cat(feature_chunks, dim=0), torch.cat(probability_chunks, dim=0)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest_key = "smoke_manifest" if args.subset == "smoke" else "dataset_manifest"
    records = load_triplet_manifest(resolve_artifact_path(config, manifest_key))
    model = load_checkpoint(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    batch_size = int(config["d1"].get("batch_size", 32))

    probe_paths = build_probe_bank(records)
    probe_names, probe_features, probe_probs = encode_probe_bank(
        model,
        probe_paths,
        device=device,
        batch_size=batch_size,
    )

    test_records = [record for record in records if record.split == "test"]
    sd15_images = torch.stack([image_tensor(record.sd15_path) for record in test_records]).to(device)
    with torch.no_grad():
        test_features = model.encode(sd15_images)
        native_probs = model.head.probabilities(test_features)

    eps = max(float(item) for item in config["attacks"]["epsilons"])
    native_attack = pgd_targeted(
        model.forward_probs,
        sd15_images,
        eps=eps,
        steps=int(config["attacks"]["steps"]),
        step_size=eps / 4.0,
        restarts=int(config["attacks"]["restarts"]),
        target="sdxl",
    )
    with torch.no_grad():
        native_adv_probs = model.forward_probs(native_attack.adversarial)
    native_scores = score_from_probs(native_adv_probs, "sdxl")

    requested_counts = [int(item) for item in config["exact_recovery"].get("query_counts", [])]
    if not requested_counts:
        requested_counts = [128, 256, 512, 1024, 2048]
    available_queries = int(probe_features.shape[0])
    query_counts = sorted({min(count, available_queries) for count in requested_counts if count > 0} | {available_queries})
    true_weights = model.head.linear.weight.detach()
    true_bias = model.head.linear.bias.detach()
    curve_points = []
    best_payload = None
    for query_count in query_counts:
        selected_features = probe_features[:query_count]
        selected_probs = probe_probs[:query_count]
        recovered = recover_softmax_relative_head(
            selected_features,
            selected_probs,
            anchor_index=int(config["exact_recovery"]["anchor_index"]),
        )
        recovered = recovered.to(device)
        recovered_test_probs = recovered.probabilities(test_features)
        parameter_metrics = relative_parameter_error(true_weights, true_bias, recovered)

        def recovered_forward(images: torch.Tensor, recovered_head=recovered) -> torch.Tensor:
            features = model.encode(images)
            return recovered_head.probabilities(features)

        recovered_attack = pgd_targeted(
            recovered_forward,
            sd15_images,
            eps=eps,
            steps=int(config["attacks"]["steps"]),
            step_size=eps / 4.0,
            restarts=int(config["attacks"]["restarts"]),
            target="sdxl",
        )
        with torch.no_grad():
            recovered_adv_probs = model.forward_probs(recovered_attack.adversarial)
        recovered_scores = score_from_probs(recovered_adv_probs, "sdxl")
        point = {
            "query_count": query_count,
            "weight_error": parameter_metrics["weight_error"],
            "bias_error": parameter_metrics["bias_error"],
            "score_error": score_error(native_probs, recovered_test_probs),
            "whitebox_recovered_gap": float((native_scores - recovered_scores).abs().mean().item()),
            "recovered_attack_asr": float((recovered_adv_probs.argmax(dim=1) == 2).float().mean().item()),
        }
        curve_points.append(point)
        if query_count == available_queries:
            best_payload = {
                "recovered": recovered,
                "recovered_adv_probs": recovered_adv_probs,
                "recovered_scores": recovered_scores,
                "metrics": point,
                "recovered_attack": recovered_attack,
            }

    if best_payload is None:
        raise RuntimeError("Exact recovery sweep produced no terminal point.")

    rows = []
    for idx, record in enumerate(test_records):
        rows.append(
            {
                "image_id": record.image_id,
                "split": "test",
                "source": "sd15",
                "target": "sdxl",
                "detector": "d1_clip3_recovered",
                "attack": "exact_recovery",
                "eps": eps,
                "pi": "",
                "seed": record.seed,
                "clean_target_score": float(score_from_probs(native_probs, "sdxl")[idx].item()),
                "adv_target_score": float(best_payload["recovered_scores"][idx].item()),
                "clean_pred": int(native_probs[idx].argmax().item()),
                "adv_pred": int(best_payload["recovered_adv_probs"][idx].argmax().item()),
                "success": int(best_payload["recovered_adv_probs"][idx].argmax().item() == 2),
                "linf": float((best_payload["recovered_attack"].adversarial[idx] - sd15_images[idx]).abs().max().item()),
                "psnr": "",
                "lpips": "",
                "emulator_eps95": "",
                "alpha_estimate": "",
                "query_count": available_queries,
            }
        )

    write_json(
        resolve_artifact_path(config, "exact_recovery_metrics"),
        {
            "available_queries": available_queries,
            "probe_bank_size": len(probe_names),
            "points": curve_points,
            "best": best_payload["metrics"],
        },
    )
    write_csv(resolve_artifact_path(config, "exact_recovery_results"), rows, RESULT_COLUMNS)


if __name__ == "__main__":
    main()
