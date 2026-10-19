from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import scipy

from barrier_i_lp import (
    DEFAULT_DIMENSION,
    DEFAULT_RADII,
    Scenario,
    build_scenarios,
    hamming_edges,
    point_to_uniform_closed_form,
    sample_empirical_distribution,
    solve_attackability_radius,
    solve_minimax_robust_gap,
    state_bitstrings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the controlled finite-state Barrier I LP experiment on X={0,1}^6."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--plugin-sizes", default="250,1000,5000")
    parser.add_argument("--plugin-repeats", type=int, default=10)
    parser.add_argument("--skip-plugin", action="store_true")
    return parser.parse_args()


def parse_integer_list(raw_value: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw_value.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("--plugin-sizes must contain positive comma-separated integers.")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_number(value: float) -> str:
    return f"{value:.8f}"


def validate_monotonic(rows: list[dict[str, Any]], value_key: str, tolerance: float) -> None:
    ordered = sorted(rows, key=lambda row: int(row["radius"]))
    for previous, current in zip(ordered, ordered[1:]):
        if float(current[value_key]) > float(previous[value_key]) + tolerance:
            raise AssertionError(
                f"{value_key} is not monotone as Gamma expands: "
                f"radius {previous['radius']} -> {current['radius']} gives "
                f"{previous[value_key]} -> {current[value_key]}."
            )


def run_population(
    scenarios: list[Scenario],
    *,
    dimension: int,
    tolerance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_rows = []
        for radius in DEFAULT_RADII:
            attack_solution = solve_attackability_radius(
                scenario.source,
                scenario.target,
                dimension=dimension,
                radius=radius,
            )
            verifier_solution = solve_minimax_robust_gap(
                scenario.source,
                scenario.target,
                dimension=dimension,
                radius=radius,
            )
            residual = abs(attack_solution.value - verifier_solution.value)
            if residual > tolerance:
                raise AssertionError(
                    f"Barrier I identity failed for {scenario.name}, radius={radius}: "
                    f"Delta={attack_solution.value}, gap={verifier_solution.value}, residual={residual}."
                )
            analytic_delta: float | str = ""
            analytic_error: float | str = ""
            if scenario.name == "point_to_uniform":
                analytic_delta = point_to_uniform_closed_form(dimension, radius)
                analytic_error = abs(attack_solution.value - analytic_delta)
                if analytic_error > tolerance:
                    raise AssertionError(
                        f"Closed-form sanity check failed at radius={radius}: "
                        f"LP={attack_solution.value}, analytic={analytic_delta}."
                    )
            row = {
                "scenario": scenario.name,
                "radius": radius,
                "state_count": 2**dimension,
                "edge_count": len(hamming_edges(dimension, radius)),
                "delta": attack_solution.value,
                "minimax_robust_gap": verifier_solution.value,
                "absolute_residual": residual,
                "analytic_delta": analytic_delta,
                "analytic_error": analytic_error,
                "transport_objective_consistency": attack_solution.consistency_error,
                "verifier_objective_consistency": verifier_solution.consistency_error,
                "transport_runtime_seconds": attack_solution.runtime_seconds,
                "verifier_runtime_seconds": verifier_solution.runtime_seconds,
                "transport_iterations": attack_solution.iterations,
                "verifier_iterations": verifier_solution.iterations,
            }
            rows.append(row)
            scenario_rows.append(row)
        validate_monotonic(scenario_rows, "delta", tolerance)
        validate_monotonic(scenario_rows, "minimax_robust_gap", tolerance)
    return rows


def run_plugin(
    scenarios: list[Scenario],
    *,
    dimension: int,
    sizes: tuple[int, ...],
    repeats: int,
    seed: int,
    tolerance: float,
    population_lookup: dict[tuple[str, int], float],
) -> list[dict[str, Any]]:
    if repeats <= 0:
        raise ValueError("--plugin-repeats must be positive when the plugin experiment is enabled.")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for sample_size in sizes:
            for replicate in range(repeats):
                empirical_source = sample_empirical_distribution(scenario.source, sample_size, rng)
                empirical_target = sample_empirical_distribution(scenario.target, sample_size, rng)
                replicate_rows = []
                for radius in DEFAULT_RADII:
                    attack_solution = solve_attackability_radius(
                        empirical_source,
                        empirical_target,
                        dimension=dimension,
                        radius=radius,
                    )
                    verifier_solution = solve_minimax_robust_gap(
                        empirical_source,
                        empirical_target,
                        dimension=dimension,
                        radius=radius,
                    )
                    residual = abs(attack_solution.value - verifier_solution.value)
                    if residual > tolerance:
                        raise AssertionError(
                            f"Plug-in Barrier I identity failed for {scenario.name}, n={sample_size}, "
                            f"replicate={replicate}, radius={radius}, residual={residual}."
                        )
                    population_delta = population_lookup[(scenario.name, radius)]
                    row = {
                        "scenario": scenario.name,
                        "sample_size": sample_size,
                        "replicate": replicate,
                        "radius": radius,
                        "delta_hat": attack_solution.value,
                        "minimax_robust_gap_hat": verifier_solution.value,
                        "absolute_residual": residual,
                        "population_delta": population_delta,
                        "signed_error": attack_solution.value - population_delta,
                        "absolute_error": abs(attack_solution.value - population_delta),
                        "transport_runtime_seconds": attack_solution.runtime_seconds,
                        "verifier_runtime_seconds": verifier_solution.runtime_seconds,
                    }
                    rows.append(row)
                    replicate_rows.append(row)
                validate_monotonic(replicate_rows, "delta_hat", tolerance)
                validate_monotonic(replicate_rows, "minimax_robust_gap_hat", tolerance)
    return rows


def summarize_plugin(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scenario"]), int(row["radius"]), int(row["sample_size"]))].append(row)
    summaries = []
    for (scenario, radius, sample_size), group_rows in sorted(grouped.items()):
        estimates = np.asarray([float(row["delta_hat"]) for row in group_rows], dtype=np.float64)
        errors = np.asarray([float(row["signed_error"]) for row in group_rows], dtype=np.float64)
        residuals = np.asarray([float(row["absolute_residual"]) for row in group_rows], dtype=np.float64)
        summaries.append(
            {
                "scenario": scenario,
                "radius": radius,
                "sample_size": sample_size,
                "repeats": len(group_rows),
                "population_delta": float(group_rows[0]["population_delta"]),
                "mean_delta_hat": float(estimates.mean()),
                "bias": float(errors.mean()),
                "mae": float(np.abs(errors).mean()),
                "std_delta_hat": float(estimates.std(ddof=1)) if estimates.size > 1 else 0.0,
                "q025_delta_hat": float(np.quantile(estimates, 0.025)),
                "q975_delta_hat": float(np.quantile(estimates, 0.975)),
                "max_identity_residual": float(residuals.max()),
            }
        )
    return summaries


def summarize_plugin_overall(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["sample_size"])].append(row)
    summaries = []
    for sample_size, group_rows in sorted(grouped.items()):
        absolute_errors = np.asarray(
            [float(row["absolute_error"]) for row in group_rows],
            dtype=np.float64,
        )
        residuals = np.asarray(
            [float(row["absolute_residual"]) for row in group_rows],
            dtype=np.float64,
        )
        summaries.append(
            {
                "sample_size": sample_size,
                "estimate_count": len(group_rows),
                "overall_mae": float(absolute_errors.mean()),
                "max_absolute_error": float(absolute_errors.max()),
                "max_identity_residual": float(residuals.max()),
            }
        )
    return summaries


def make_markdown_summary(
    *,
    population_rows: list[dict[str, Any]],
    plugin_summary_rows: list[dict[str, Any]],
    plugin_overall_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    lines = [
        "# Barrier I Finite-State LP Results",
        "",
        f"- State space: `X={{0,1}}^{metadata['dimension']}` ({metadata['state_count']} states)",
        f"- Hamming radii: `{metadata['radii']}`",
        f"- Equality tolerance: `{metadata['tolerance']}`",
        f"- Maximum population identity residual: `{metadata['max_population_identity_residual']:.3e}`",
        f"- Total wall time: `{metadata['wall_time_seconds']:.3f}` seconds",
        "",
        "## Population LPs",
        "",
        "| Scenario | Radius | Delta | Minimax robust gap | Absolute residual |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in population_rows:
        lines.append(
            "| {scenario} | {radius} | {delta} | {gap} | {residual} |".format(
                scenario=row["scenario"],
                radius=row["radius"],
                delta=format_number(float(row["delta"])),
                gap=format_number(float(row["minimax_robust_gap"])),
                residual=f"{float(row['absolute_residual']):.3e}",
            )
        )
    if plugin_summary_rows:
        lines.extend(
            [
                "",
                "## Plug-in Estimates",
                "",
                "Each row aggregates independent empirical source/target histograms. Quantiles are across Monte Carlo replicates, not confidence-interval coverage claims.",
                "",
                "### Overall error by sample size",
                "",
                "| n | Estimates | Overall MAE | Maximum absolute error |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for row in plugin_overall_rows:
            lines.append(
                "| {sample_size} | {count} | {mae} | {maximum} |".format(
                    sample_size=row["sample_size"],
                    count=row["estimate_count"],
                    mae=format_number(float(row["overall_mae"])),
                    maximum=format_number(float(row["max_absolute_error"])),
                )
            )
        lines.extend(
            [
                "",
                "### Per-scenario and radius",
                "",
                "| Scenario | Radius | n | Repeats | Population Delta | Mean Delta-hat | MAE |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in plugin_summary_rows:
            lines.append(
                "| {scenario} | {radius} | {sample_size} | {repeats} | {population} | {estimate} | {mae} |".format(
                    scenario=row["scenario"],
                    radius=row["radius"],
                    sample_size=row["sample_size"],
                    repeats=row["repeats"],
                    population=format_number(float(row["population_delta"])),
                    estimate=format_number(float(row["mean_delta_hat"])),
                    mae=format_number(float(row["mae"])),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This controlled experiment numerically checks the two independent LP formulations and illustrates finite-sample plug-in behavior. It does not estimate the attackability radius of real high-dimensional image distributions.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dimension = DEFAULT_DIMENSION
    tolerance = float(args.tolerance)
    plugin_sizes = parse_integer_list(args.plugin_sizes)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    scenarios = build_scenarios(dimension)
    population_rows = run_population(scenarios, dimension=dimension, tolerance=tolerance)
    population_lookup = {
        (str(row["scenario"]), int(row["radius"])): float(row["delta"])
        for row in population_rows
    }
    plugin_rows: list[dict[str, Any]] = []
    if not args.skip_plugin:
        plugin_rows = run_plugin(
            scenarios,
            dimension=dimension,
            sizes=plugin_sizes,
            repeats=int(args.plugin_repeats),
            seed=int(args.seed),
            tolerance=tolerance,
            population_lookup=population_lookup,
        )
    plugin_summary_rows = summarize_plugin(plugin_rows)
    plugin_overall_rows = summarize_plugin_overall(plugin_rows)
    wall_time = perf_counter() - start

    max_population_residual = max(float(row["absolute_residual"]) for row in population_rows)
    max_plugin_residual = max((float(row["absolute_residual"]) for row in plugin_rows), default=0.0)
    metadata = {
        "all_checks_passed": max(max_population_residual, max_plugin_residual) <= tolerance,
        "dimension": dimension,
        "state_count": 2**dimension,
        "radii": list(DEFAULT_RADII),
        "scenario_count": len(scenarios),
        "population_lp_pairs": len(population_rows),
        "plugin_lp_pairs": len(plugin_rows),
        "plugin_sizes": list(plugin_sizes) if plugin_rows else [],
        "plugin_repeats": int(args.plugin_repeats) if plugin_rows else 0,
        "seed": int(args.seed),
        "tolerance": tolerance,
        "max_population_identity_residual": max_population_residual,
        "max_plugin_identity_residual": max_plugin_residual,
        "wall_time_seconds": wall_time,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
    }

    population_fields = list(population_rows[0].keys())
    write_csv(output_dir / "population_results.csv", population_rows, population_fields)
    if plugin_rows:
        write_csv(output_dir / "plugin_results.csv", plugin_rows, list(plugin_rows[0].keys()))
        write_csv(output_dir / "plugin_summary.csv", plugin_summary_rows, list(plugin_summary_rows[0].keys()))
        write_csv(
            output_dir / "plugin_overall_summary.csv",
            plugin_overall_rows,
            list(plugin_overall_rows[0].keys()),
        )
    scenario_payload = {
        "dimension": dimension,
        "states": state_bitstrings(dimension),
        "scenarios": [
            {
                "name": scenario.name,
                "description": scenario.description,
                "source": scenario.source.tolist(),
                "target": scenario.target.tolist(),
            }
            for scenario in scenarios
        ],
    }
    write_json(output_dir / "scenario_distributions.json", scenario_payload)
    write_json(
        output_dir / "summary.json",
        {
            "metadata": metadata,
            "population_results": population_rows,
            "plugin_summary": plugin_summary_rows,
            "plugin_overall_summary": plugin_overall_rows,
        },
    )
    (output_dir / "summary.md").write_text(
        make_markdown_summary(
            population_rows=population_rows,
            plugin_summary_rows=plugin_summary_rows,
            plugin_overall_rows=plugin_overall_rows,
            metadata=metadata,
        ),
        encoding="utf-8",
    )

    if not metadata["all_checks_passed"]:
        raise AssertionError(f"At least one identity residual exceeded tolerance={tolerance}.")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
