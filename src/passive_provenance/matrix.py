from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from passive_provenance.config import resolve_artifact_path, resolve_artifact_relative


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    label_space: tuple[str, ...]
    generator_ids: tuple[str, ...]
    generator_benchmark_overrides: dict[str, str]
    legacy_triplet_export: bool


@dataclass(frozen=True)
class GenerationModelSpec:
    model_id: str
    steps: int
    guidance_scale: float
    variant: str
    use_safetensors: bool
    width: int
    height: int


@dataclass(frozen=True)
class VerifierSpec:
    verifier_id: str
    benchmark_id: str
    family: str
    head_type: str
    class_names: tuple[str, ...]
    differentiable: bool
    score_revealing: bool
    public_encoder: bool


@dataclass(frozen=True)
class InterfaceSpec:
    interface_id: str
    kind: str
    quantization_bits: int | None = None
    rate_limit: int | None = None
    abstention: bool = False


def _tuple(values: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(values)


def get_benchmark_spec(config: dict[str, Any], benchmark_id: str) -> BenchmarkSpec:
    raw = config["benchmarks"][benchmark_id]
    return BenchmarkSpec(
        benchmark_id=benchmark_id,
        label_space=_tuple(raw["label_space"]),
        generator_ids=_tuple(raw["generator_ids"]),
        generator_benchmark_overrides=dict(raw.get("generator_benchmark_overrides", {})),
        legacy_triplet_export=bool(raw.get("legacy_triplet_export", False)),
    )


def get_generation_model_spec(config: dict[str, Any], model_id: str) -> GenerationModelSpec:
    models = config["generation"].get("models", {})
    raw = models[model_id]
    return GenerationModelSpec(
        model_id=raw["model_id"],
        steps=int(raw["steps"]),
        guidance_scale=float(raw["guidance_scale"]),
        variant=str(raw.get("variant", "")),
        use_safetensors=bool(raw.get("use_safetensors", True)),
        width=int(raw.get("width", config["generation"]["width"])),
        height=int(raw.get("height", config["generation"]["height"])),
    )


def get_verifier_spec(config: dict[str, Any], verifier_id: str) -> VerifierSpec:
    raw = config["verifiers"][verifier_id]
    return VerifierSpec(
        verifier_id=verifier_id,
        benchmark_id=raw["benchmark_id"],
        family=raw["family"],
        head_type=raw["head_type"],
        class_names=_tuple(raw["class_names"]),
        differentiable=bool(raw.get("differentiable", True)),
        score_revealing=bool(raw.get("score_revealing", False)),
        public_encoder=bool(raw.get("public_encoder", True)),
    )


def get_interface_spec(config: dict[str, Any], interface_id: str) -> InterfaceSpec:
    raw = config["interfaces"][interface_id]
    return InterfaceSpec(
        interface_id=interface_id,
        kind=raw["kind"],
        quantization_bits=int(raw["quantization_bits"]) if raw.get("quantization_bits") is not None else None,
        rate_limit=int(raw["rate_limit"]) if raw.get("rate_limit") is not None else None,
        abstention=bool(raw.get("abstention", False)),
    )


def artifacts_root(config: dict[str, Any]) -> Path:
    return resolve_artifact_path(config, "artifacts_root")


def benchmark_generated_root(config: dict[str, Any], benchmark_id: str) -> Path:
    if benchmark_id == "b0":
        return resolve_artifact_relative(config, config["data"]["generated_output_dir"])
    return resolve_artifact_relative(config, f"artifacts/datasets/benchmarks/{benchmark_id}/generated")


def benchmark_generated_root_for_generator(config: dict[str, Any], benchmark_id: str, generator_id: str) -> Path:
    benchmark = get_benchmark_spec(config, benchmark_id)
    source_benchmark = benchmark.generator_benchmark_overrides.get(generator_id, benchmark_id)
    return benchmark_generated_root(config, source_benchmark) / generator_id


def benchmark_real_root(config: dict[str, Any], benchmark_id: str) -> Path:
    if benchmark_id == "b0":
        return resolve_artifact_relative(config, config["data"]["real_output_dir"])
    return resolve_artifact_relative(config, f"artifacts/datasets/benchmarks/{benchmark_id}/real")


def sample_manifest_path(config: dict[str, Any], benchmark_id: str, subset: str) -> Path:
    suffix = "smoke" if subset == "smoke" else "full"
    return resolve_artifact_relative(config, f"artifacts/generated/benchmarks/{benchmark_id}/sample_manifest_{suffix}.csv")


def legacy_triplet_manifest_path(config: dict[str, Any], benchmark_id: str, subset: str) -> Path:
    suffix = "smoke" if subset == "smoke" else "full"
    if benchmark_id == "b0":
        key = "smoke_manifest" if subset == "smoke" else "dataset_manifest"
        return resolve_artifact_path(config, key)
    return resolve_artifact_relative(config, f"artifacts/generated/benchmarks/{benchmark_id}/triplet_manifest_{suffix}.csv")


def namespace_root(config: dict[str, Any], benchmark_id: str, verifier_id: str | None = None, interface_id: str | None = None) -> Path:
    root = resolve_artifact_relative(config, f"artifacts/generated/{benchmark_id}")
    if verifier_id is not None:
        root = root / verifier_id
    if interface_id is not None:
        root = root / interface_id
    return root


def feature_cache_path(config: dict[str, Any], benchmark_id: str, subset: str = "full") -> Path:
    return namespace_root(config, benchmark_id) / f"feature_cache_{subset}.pt"


def verifier_checkpoint_path(config: dict[str, Any], benchmark_id: str, verifier_id: str) -> Path:
    return namespace_root(config, benchmark_id, verifier_id) / "checkpoint.pt"


def verifier_metrics_path(config: dict[str, Any], benchmark_id: str, verifier_id: str) -> Path:
    return namespace_root(config, benchmark_id, verifier_id) / "clean_metrics.json"


def verifier_predictions_path(config: dict[str, Any], benchmark_id: str, verifier_id: str) -> Path:
    return namespace_root(config, benchmark_id, verifier_id) / "clean_predictions.csv"


def query_transcript_path(config: dict[str, Any], benchmark_id: str, verifier_id: str, interface_id: str, subset: str) -> Path:
    return namespace_root(config, benchmark_id, verifier_id, interface_id) / f"query_transcript_{subset}.csv"


def interface_summary_path(config: dict[str, Any], benchmark_id: str, verifier_id: str, interface_id: str, subset: str) -> Path:
    return namespace_root(config, benchmark_id, verifier_id, interface_id) / f"query_summary_{subset}.json"


def emulator_checkpoint_path(config: dict[str, Any], benchmark_id: str, verifier_id: str, interface_id: str) -> Path:
    return namespace_root(config, benchmark_id, verifier_id, interface_id) / "emulator.pt"


def emulator_metrics_path(config: dict[str, Any], benchmark_id: str, verifier_id: str, interface_id: str) -> Path:
    return namespace_root(config, benchmark_id, verifier_id, interface_id) / "emulator_metrics.json"


def attack_results_path(config: dict[str, Any], benchmark_id: str, verifier_id: str, attack_id: str, interface_id: str | None = None) -> Path:
    base = namespace_root(config, benchmark_id, verifier_id, interface_id)
    return base / f"{attack_id}_results.csv"


def attack_summary_path(config: dict[str, Any], benchmark_id: str, verifier_id: str, attack_id: str, interface_id: str | None = None) -> Path:
    base = namespace_root(config, benchmark_id, verifier_id, interface_id)
    return base / f"{attack_id}_summary.json"


def paper_matrix_root(config: dict[str, Any]) -> Path:
    return resolve_artifact_relative(config, "artifacts/generated/paper_matrix")


def ai_class_names(class_names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(name for name in class_names if name not in {"real", "human"})


def target_weights(
    class_names: tuple[str, ...] | list[str],
    target: str,
    *,
    uniform_ai: bool = False,
) -> dict[str, float]:
    names = tuple(class_names)
    if target in names:
        return {target: 1.0}
    if target in {"fake", "ai", "coarse_ai"}:
        ai_names = ai_class_names(names)
        if not ai_names:
            raise ValueError(f"No AI classes available for target {target}.")
        if uniform_ai:
            weight = 1.0 / float(len(ai_names))
            return {name: weight for name in ai_names}
        return {name: 1.0 for name in ai_names}
    raise ValueError(f"Unsupported target {target}.")
