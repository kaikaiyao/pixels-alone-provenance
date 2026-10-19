from __future__ import annotations

import unittest

from barrier_i_lp import (
    DEFAULT_DIMENSION,
    DEFAULT_RADII,
    build_scenarios,
    point_to_uniform_closed_form,
    solve_attackability_radius,
    solve_minimax_robust_gap,
)


TOLERANCE = 1e-6


class BarrierIFiniteStateLPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = build_scenarios(DEFAULT_DIMENSION)

    def test_point_to_uniform_matches_closed_form(self) -> None:
        scenario = next(item for item in self.scenarios if item.name == "point_to_uniform")
        for radius in DEFAULT_RADII:
            solution = solve_attackability_radius(
                scenario.source,
                scenario.target,
                dimension=DEFAULT_DIMENSION,
                radius=radius,
            )
            expected = point_to_uniform_closed_form(DEFAULT_DIMENSION, radius)
            self.assertAlmostEqual(solution.value, expected, delta=TOLERANCE)

    def test_independent_lp_values_match(self) -> None:
        for scenario in self.scenarios:
            for radius in DEFAULT_RADII:
                delta = solve_attackability_radius(
                    scenario.source,
                    scenario.target,
                    dimension=DEFAULT_DIMENSION,
                    radius=radius,
                ).value
                minimax_gap = solve_minimax_robust_gap(
                    scenario.source,
                    scenario.target,
                    dimension=DEFAULT_DIMENSION,
                    radius=radius,
                ).value
                self.assertAlmostEqual(
                    delta,
                    minimax_gap,
                    delta=TOLERANCE,
                    msg=f"scenario={scenario.name}, radius={radius}",
                )

    def test_values_are_monotone_as_edit_set_expands(self) -> None:
        for scenario in self.scenarios:
            values = [
                solve_attackability_radius(
                    scenario.source,
                    scenario.target,
                    dimension=DEFAULT_DIMENSION,
                    radius=radius,
                ).value
                for radius in DEFAULT_RADII
            ]
            for previous, current in zip(values, values[1:]):
                self.assertLessEqual(current, previous + TOLERANCE)


if __name__ == "__main__":
    unittest.main()
