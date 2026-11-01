"""Reliability comparisons, masking, convergence gates and resumable results."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from electrostatic_bvp.cases import build_parallel_plate
from electrostatic_bvp.config import load_config
from electrostatic_bvp.fields import parallel_plate_field
from electrostatic_bvp.solver import solve_parallel_plate
from scripts.validate_m2 import (
    CaseData, algebraic_bounds, compare_cases, load_case, meets_tolerance, save_case,
)
from scripts.plot_m2_grid_comparison import save_figure


ROOT = Path(__file__).resolve().parents[1]


def manufactured(h):
    config = load_config(ROOT / "configs/parallel_plate.json")
    config["geometry"] = {"plate_length": 0.4, "gap": 0.4,
                          "domain_half_width": 0.8, "domain_half_height": 0.8}
    config["grid"]["h"] = h
    grid = build_parallel_plate(config)
    xg, yg = np.meshgrid(grid.x, grid.y)
    potential = yg + 0.25 * xg
    field = parallel_plate_field(grid.x, grid.y, potential, grid.positive_plate_mask,
                                 grid.negative_plate_mask, h=h)
    arrays = {"x": grid.x, "y": grid.y, "potential": potential,
              "electric_field_x": field.electric_field_x,
              "electric_field_y": field.electric_field_y,
              "analysis_valid_mask": field.analysis_valid_mask}
    meta = {"converged": True, "max_abs_laplace_residual_free_nodes": 0.0,
            "endpoint_exclusion_radius": field.endpoint_exclusion_radius,
            "endpoint_exclusion_has_two_intervals": True}
    return CaseData(config, arrays, meta)


class ValidationM2Tests(unittest.TestCase):
    def test_common_nodes_and_fixed_physical_mask(self):
        coarse, fine = manufactured(0.1), manufactured(0.05)
        target = (-0.4, 0.4, -0.3, 0.3)
        result = compare_cases(coarse, fine, target)
        self.assertFalse(result["interpolation"])
        self.assertGreater(result["common_valid_nodes"], 0)
        self.assertLess(result["max_potential_difference_over_abs_v0"], 1e-14)
        self.assertLess(result["max_field_vector_difference_over_eref"], 1e-14)
        self.assertTrue(meets_tolerance(result, 0.01, require_iteration_bound=True))
        # Perturbing a conductor endpoint is excluded from the physical region.
        xidx = int(np.flatnonzero(np.isclose(fine.arrays["x"], 0.2))[0])
        yidx = int(np.flatnonzero(np.isclose(fine.arrays["y"], 0.2))[0])
        fine.arrays["potential"][yidx, xidx] += 100
        self.assertLess(compare_cases(coarse, fine, target)["max_potential_difference_over_abs_v0"], 1e-14)
        # A genuine common interior node must be counted.
        xidx = len(fine.arrays["x"]) // 2
        yidx = len(fine.arrays["y"]) // 2
        fine.arrays["potential"][yidx, xidx] += 0.03
        result = compare_cases(coarse, fine, target)
        self.assertAlmostEqual(result["max_potential_difference_over_abs_v0"], 0.03)
        self.assertFalse(meets_tolerance(result, 0.01, require_iteration_bound=True))

    def test_no_acceptance_for_nonconvergence_or_large_iteration_bound(self):
        first, second = manufactured(0.1), manufactured(0.05)
        target = (-0.4, 0.4, -0.3, 0.3)
        second.metadata["converged"] = False
        comparison = compare_cases(first, second, target)
        self.assertFalse(meets_tolerance(comparison, 0.01, require_iteration_bound=False))
        second.metadata["converged"] = True
        second.metadata["max_abs_laplace_residual_free_nodes"] = 0.1
        comparison = compare_cases(first, second, target)
        self.assertFalse(meets_tolerance(comparison, 0.01, require_iteration_bound=True))

    def test_reject_unmatched_masks_grid_and_target(self):
        first, second = manufactured(0.1), manufactured(0.05)
        target = (-0.4, 0.4, -0.3, 0.3)
        second.arrays["analysis_valid_mask"][len(second.arrays["y"]) // 2, len(second.arrays["x"]) // 2] = False
        with self.assertRaisesRegex(ValueError, "masks"):
            compare_cases(first, second, target)
        second = manufactured(0.05)
        with self.assertRaisesRegex(ValueError, "target region"):
            compare_cases(first, second, (-2, 2, -0.3, 0.3))
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            compare_cases(first, second, (-0.8, 0.8, -0.3, 0.3))
        second.config["grid"]["h"] = 0.06
        with self.assertRaisesRegex(ValueError, "nested"):
            compare_cases(first, second, target)

    def test_barrier_bound_covers_actual_discrete_iteration_error(self):
        config = load_config(ROOT / "configs/parallel_plate.json")
        config["geometry"] = {"plate_length": 0.4, "gap": 0.4,
                              "domain_half_width": 0.6, "domain_half_height": 0.6}
        config["grid"]["h"] = 0.1
        config["solver"]["Nmax"] = 3
        approximate = solve_parallel_plate(config)
        accurate_config = deepcopy(config)
        accurate_config["solver"] = {"method": "jacobi", "tol": 1e-14, "Nmax": 10000}
        exact = solve_parallel_plate(accurate_config)
        self.assertTrue(exact.converged)
        case = CaseData(config, {}, {"max_abs_laplace_residual_free_nodes": approximate.max_abs_residual})
        bounds = algebraic_bounds(case)
        self.assertLessEqual(np.max(np.abs(approximate.potential - exact.potential)),
                             bounds["potential_over_abs_v0"] + 1e-12)
        interior = approximate.field.field_valid_mask[1:-1, 1:-1]
        error = np.hypot(approximate.field.electric_field_x[1:-1, 1:-1] - exact.field.electric_field_x[1:-1, 1:-1],
                         approximate.field.electric_field_y[1:-1, 1:-1] - exact.field.electric_field_y[1:-1, 1:-1])
        actual = np.max(error[interior]) / (1 / config["geometry"]["gap"])
        self.assertLessEqual(actual, bounds["field_vector_over_eref"] + 1e-12)

    def test_saved_run_resume_validates_provenance_and_data(self):
        config = load_config(ROOT / "configs/parallel_plate.json")
        config["solver"]["Nmax"] = 1
        result = solve_parallel_plate(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            original = save_case(path, config, result, 0.2, {"source": "example"})
            loaded = load_case(path, config, {"source": "example"})
            np.testing.assert_array_equal(original.arrays["potential"], loaded.arrays["potential"])
            with self.assertRaisesRegex(ValueError, "source"):
                load_case(path, config, {"source": "different"})
            with (path / "solution.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_case(path, config, {"source": "example"})

    def test_grid_figure_shows_valid_common_node_maximum(self):
        coarse, fine = manufactured(0.1), manufactured(0.05)
        target = (-0.4, 0.4, -0.3, 0.3)
        center = (len(fine.arrays["y"]) // 2, len(fine.arrays["x"]) // 2)
        fine.arrays["electric_field_x"][center] += 0.1
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.png"
            result, peak = save_figure(coarse, fine, output, target=target)
            self.assertAlmostEqual(result["max_field_vector_difference_over_eref"], 0.04)
            np.testing.assert_allclose(peak, (0, 0))
            self.assertGreater(output.stat().st_size, 1000)
            coarse.metadata["converged"] = False
            with self.assertRaisesRegex(ValueError, "must converge"):
                save_figure(coarse, fine, output, target=target)


if __name__ == "__main__":
    unittest.main()
