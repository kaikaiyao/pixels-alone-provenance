from __future__ import annotations

from dataclasses import dataclass
from math import comb
from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


DEFAULT_DIMENSION = 6
DEFAULT_RADII = (0, 1, 2, 3)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    source: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class LPSolution:
    value: float
    objective_value: float
    consistency_error: float
    runtime_seconds: float
    iterations: int
    status: str


def normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("A categorical distribution must be one-dimensional.")
    if np.any(values < 0.0):
        raise ValueError("A categorical distribution cannot contain negative mass.")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("A categorical distribution must have positive total mass.")
    values = values / total
    if not np.isclose(values.sum(), 1.0, atol=1e-12):
        raise RuntimeError("Distribution normalization failed.")
    return values


def binary_state_matrix(dimension: int = DEFAULT_DIMENSION) -> np.ndarray:
    state_ids = np.arange(2**dimension, dtype=np.int64)
    bit_positions = np.arange(dimension, dtype=np.int64)
    return ((state_ids[:, None] >> bit_positions[None, :]) & 1).astype(np.int8)


def state_bitstrings(dimension: int = DEFAULT_DIMENSION) -> list[str]:
    return [format(state_id, f"0{dimension}b") for state_id in range(2**dimension)]


def hamming_edges(dimension: int, radius: int) -> list[tuple[int, int]]:
    if radius < 0 or radius > dimension:
        raise ValueError(f"radius must be in [0, {dimension}], got {radius}.")
    state_count = 2**dimension
    return [
        (source, destination)
        for source in range(state_count)
        for destination in range(state_count)
        if bin(source ^ destination).count("1") <= radius
    ]


def product_bernoulli(bit_probabilities: list[float] | tuple[float, ...]) -> np.ndarray:
    probabilities = np.asarray(bit_probabilities, dtype=np.float64)
    if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
        raise ValueError("Product-Bernoulli probabilities must lie strictly between zero and one.")
    states = binary_state_matrix(len(probabilities)).astype(np.float64)
    masses = np.prod(
        np.where(states == 1.0, probabilities[None, :], 1.0 - probabilities[None, :]),
        axis=1,
    )
    return normalize(masses)


def point_mixture(dimension: int, weighted_states: list[tuple[int, float]]) -> np.ndarray:
    masses = np.zeros(2**dimension, dtype=np.float64)
    for state_id, weight in weighted_states:
        if state_id < 0 or state_id >= 2**dimension:
            raise ValueError(f"Invalid state id {state_id} for dimension={dimension}.")
        masses[state_id] += float(weight)
    return normalize(masses)


def build_scenarios(dimension: int = DEFAULT_DIMENSION) -> list[Scenario]:
    if dimension != DEFAULT_DIMENSION:
        raise ValueError(
            f"This controlled experiment fixes dimension={DEFAULT_DIMENSION}; got {dimension}."
        )
    uniform = np.full(2**dimension, 1.0 / (2**dimension), dtype=np.float64)
    source_alternating = [0.15, 0.75, 0.15, 0.75, 0.15, 0.75]
    target_alternating = [0.75, 0.15, 0.75, 0.15, 0.75, 0.15]
    bimodal_source = 0.5 * product_bernoulli([0.10] * dimension) + 0.5 * product_bernoulli([0.90] * dimension)
    return [
        Scenario(
            name="point_to_uniform",
            description="A point-mass source at 000000 and a uniform target; Delta has a closed form.",
            source=point_mixture(dimension, [(0, 1.0)]),
            target=uniform,
        ),
        Scenario(
            name="bernoulli_density_shift",
            description="Independent bits shift from sparse ones (p=0.20) to dense ones (p=0.65).",
            source=product_bernoulli([0.20] * dimension),
            target=product_bernoulli([0.65] * dimension),
        ),
        Scenario(
            name="alternating_bit_bias",
            description="Source and target reverse alternating bit-wise Bernoulli biases.",
            source=product_bernoulli(source_alternating),
            target=product_bernoulli(target_alternating),
        ),
        Scenario(
            name="translated_two_modes",
            description="Two discrete source modes are translated toward two differently weighted target modes.",
            source=point_mixture(dimension, [(0b000000, 0.60), (0b111111, 0.40)]),
            target=point_mixture(dimension, [(0b001111, 0.60), (0b110000, 0.40)]),
        ),
        Scenario(
            name="bimodal_to_central",
            description="A smoothed antipodal mixture is compared with an unbiased product target.",
            source=normalize(bimodal_source),
            target=product_bernoulli([0.50] * dimension),
        ),
    ]


def point_to_uniform_closed_form(dimension: int, radius: int) -> float:
    reachable_count = sum(comb(dimension, weight) for weight in range(radius + 1))
    return 1.0 - reachable_count / float(2**dimension)


def validate_distributions(source: np.ndarray, target: np.ndarray, dimension: int) -> None:
    expected_size = 2**dimension
    for label, distribution in (("source", source), ("target", target)):
        if distribution.shape != (expected_size,):
            raise ValueError(f"{label} has shape {distribution.shape}; expected {(expected_size,)}.")
        if np.any(distribution < -1e-12):
            raise ValueError(f"{label} contains negative mass.")
        if not np.isclose(distribution.sum(), 1.0, atol=1e-10):
            raise ValueError(f"{label} does not sum to one: {distribution.sum()}.")


def solve_attackability_radius(
    source: np.ndarray,
    target: np.ndarray,
    *,
    dimension: int,
    radius: int,
) -> LPSolution:
    """Solve Delta = min_Q TV(P_target, Q) over all Hamming-bounded transport kernels."""

    validate_distributions(source, target, dimension)
    state_count = 2**dimension
    edges = hamming_edges(dimension, radius)
    edge_count = len(edges)
    variable_count = edge_count + state_count

    objective = np.zeros(variable_count, dtype=np.float64)
    objective[edge_count:] = 0.5

    equality_rows: list[int] = []
    equality_columns: list[int] = []
    equality_data: list[float] = []
    for edge_index, (source_state, _destination_state) in enumerate(edges):
        equality_rows.append(source_state)
        equality_columns.append(edge_index)
        equality_data.append(1.0)
    equality_matrix = coo_matrix(
        (equality_data, (equality_rows, equality_columns)),
        shape=(state_count, variable_count),
    ).tocsr()

    upper_rows: list[int] = []
    upper_columns: list[int] = []
    upper_data: list[float] = []
    incoming_edges: list[list[int]] = [[] for _ in range(state_count)]
    for edge_index, (_source_state, destination_state) in enumerate(edges):
        incoming_edges[destination_state].append(edge_index)
    for destination_state, indices in enumerate(incoming_edges):
        positive_row = 2 * destination_state
        negative_row = positive_row + 1
        for edge_index in indices:
            upper_rows.extend((positive_row, negative_row))
            upper_columns.extend((edge_index, edge_index))
            upper_data.extend((1.0, -1.0))
        deviation_index = edge_count + destination_state
        upper_rows.extend((positive_row, negative_row))
        upper_columns.extend((deviation_index, deviation_index))
        upper_data.extend((-1.0, -1.0))
    upper_matrix = coo_matrix(
        (upper_data, (upper_rows, upper_columns)),
        shape=(2 * state_count, variable_count),
    ).tocsr()
    upper_rhs = np.empty(2 * state_count, dtype=np.float64)
    upper_rhs[0::2] = target
    upper_rhs[1::2] = -target

    start = perf_counter()
    result = linprog(
        objective,
        A_ub=upper_matrix,
        b_ub=upper_rhs,
        A_eq=equality_matrix,
        b_eq=source,
        bounds=(0.0, None),
        method="highs",
        options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9},
    )
    runtime = perf_counter() - start
    if not result.success:
        raise RuntimeError(
            f"Attackability-radius LP failed for radius={radius}: status={result.status}, message={result.message}"
        )

    transport = result.x[:edge_count]
    attacked_distribution = np.zeros(state_count, dtype=np.float64)
    for edge_index, (_source_state, destination_state) in enumerate(edges):
        attacked_distribution[destination_state] += transport[edge_index]
    direct_tv = 0.5 * float(np.abs(target - attacked_distribution).sum())
    objective_value = float(result.fun)
    return LPSolution(
        value=direct_tv,
        objective_value=objective_value,
        consistency_error=abs(direct_tv - objective_value),
        runtime_seconds=runtime,
        iterations=int(result.nit),
        status=str(result.message),
    )


def solve_minimax_robust_gap(
    source: np.ndarray,
    target: np.ndarray,
    *,
    dimension: int,
    radius: int,
) -> LPSolution:
    """Solve sup_f E_target[f] - E_source[max_{z in Gamma(x)} f(z)] as an independent LP."""

    validate_distributions(source, target, dimension)
    state_count = 2**dimension
    edges = hamming_edges(dimension, radius)
    variable_count = 2 * state_count

    # scipy minimizes, so negate the verifier term and retain the source max term.
    objective = np.concatenate((-target, source)).astype(np.float64)
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for row_index, (source_state, destination_state) in enumerate(edges):
        # f(destination) <= m(source), where m(source) represents max_{z in Gamma(source)} f(z).
        rows.extend((row_index, row_index))
        columns.extend((destination_state, state_count + source_state))
        data.extend((1.0, -1.0))
    upper_matrix = coo_matrix(
        (data, (rows, columns)),
        shape=(len(edges), variable_count),
    ).tocsr()

    start = perf_counter()
    result = linprog(
        objective,
        A_ub=upper_matrix,
        b_ub=np.zeros(len(edges), dtype=np.float64),
        bounds=[(0.0, 1.0)] * variable_count,
        method="highs",
        options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9},
    )
    runtime = perf_counter() - start
    if not result.success:
        raise RuntimeError(
            f"Verifier minimax LP failed for radius={radius}: status={result.status}, message={result.message}"
        )

    verifier = result.x[:state_count]
    worst_source_scores = np.asarray(
        [
            max(verifier[destination] for candidate_source, destination in edges if candidate_source == source_state)
            for source_state in range(state_count)
        ],
        dtype=np.float64,
    )
    direct_gap = float(target @ verifier - source @ worst_source_scores)
    objective_gap = -float(result.fun)
    return LPSolution(
        value=direct_gap,
        objective_value=objective_gap,
        consistency_error=abs(direct_gap - objective_gap),
        runtime_seconds=runtime,
        iterations=int(result.nit),
        status=str(result.message),
    )


def sample_empirical_distribution(
    distribution: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    counts = rng.multinomial(sample_size, distribution)
    return counts.astype(np.float64) / float(sample_size)
