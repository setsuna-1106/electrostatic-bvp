"""Physical, discretization, and stopping-condition checks."""

import unittest
from pathlib import Path

import numpy as np

from electrostatic_bvp.config import ConfigError, load_config
from electrostatic_bvp.solver import analytic_rectangle_sine, electric_field, solve_rectangle_sine


ROOT = Path(__file__).resolve().parents[1]


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/rectangle.json")
        self.config["grid"]["h"] = 0.1
        self.config["solver"]["tol"] = 1e-11

    def test_boundaries_symmetry_maximum_principle_and_residual(self):
        result = solve_rectangle_sine(self.config)
        self.assertTrue(result.converged)
        self.assertLessEqual(result.final_change, self.config["solver"]["tol"])
        u = result.potential
        np.testing.assert_array_equal(u[0], 0)
        np.testing.assert_array_equal(u[:, 0], 0)
        np.testing.assert_array_equal(u[:, -1], 0)
        np.testing.assert_allclose(u[-1, 1:-1], np.sin(np.pi * result.x[1:-1] / 1.5))
        np.testing.assert_allclose(u, u[:, ::-1], atol=1e-15)
        self.assertGreaterEqual(float(u.min()), 0)
        self.assertLessEqual(float(u.max()), 1)
        residual = (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1]
                    + u[2:, 1:-1] - 4 * u[1:-1, 1:-1]) / 0.1**2
        self.assertLess(float(np.max(np.abs(residual))), 4e-9)
        self.assertAlmostEqual(result.max_abs_residual, np.max(np.abs(residual)), places=12)
        error = u[1:-1, 1:-1] - result.analytic_potential[1:-1, 1:-1]
        self.assertAlmostEqual(result.max_abs_error, np.max(np.abs(error)))
        self.assertAlmostEqual(result.rmse, np.sqrt(np.mean(error**2)))

    def test_one_step_is_jacobi_and_nmax_is_not_convergence(self):
        self.config["solver"]["Nmax"] = 1
        result = solve_rectangle_sine(self.config)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        expected = np.zeros_like(result.potential)
        expected[-1, 1:-1] = np.sin(np.pi * result.x[1:-1] / 1.5)
        expected[-2, 1:-1] = expected[-1, 1:-1] / 4
        np.testing.assert_allclose(result.potential, expected, rtol=0, atol=1e-16)

    def test_convergence_at_last_allowed_iteration(self):
        self.config["solver"]["tol"] = 1.0
        self.config["solver"]["Nmax"] = 1
        result = solve_rectangle_sine(self.config)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)

    def test_zero_and_negative_amplitude(self):
        reference = solve_rectangle_sine(self.config)
        self.config["boundary_conditions"]["V0"] = -1
        negative = solve_rectangle_sine(self.config)
        np.testing.assert_array_equal(negative.potential, -reference.potential)
        np.testing.assert_array_equal(negative.electric_field_y, -reference.electric_field_y)
        self.config["boundary_conditions"]["V0"] = 0
        zero = solve_rectangle_sine(self.config)
        self.assertTrue(zero.converged)
        self.assertEqual(zero.iterations, 1)
        self.assertEqual(zero.max_abs_error, 0)
        np.testing.assert_array_equal(zero.electric_field_x, 0)
        np.testing.assert_array_equal(zero.electric_field_y, 0)

    def test_minimum_three_by_three_grid(self):
        self.config["geometry"] = {"Lx": 1, "Ly": 1}
        self.config["grid"]["h"] = 0.5
        result = solve_rectangle_sine(self.config)
        self.assertTrue(result.converged)
        self.assertEqual(result.potential.shape, (3, 3))
        self.assertAlmostEqual(result.potential[1, 1], 0.25)

    def test_field_sign_axes_and_second_order_edges(self):
        x, y = np.meshgrid(np.linspace(0, 2, 5), np.linspace(0, 3, 7))
        potential = 3 * x**2 + 2 * x - 4 * y**2 + 5 * y + 1
        ex, ey = electric_field(potential, dx=0.5, dy=0.5)
        np.testing.assert_allclose(ex, -6 * x - 2, atol=1e-13)
        np.testing.assert_allclose(ey, 8 * y - 5, atol=1e-13)

    def test_stable_analytic_solution_for_tall_rectangle(self):
        x = np.array([0, 0.5, 1])
        y = np.array([0, 1, 299, 300])
        with np.errstate(over="raise", invalid="raise"):
            actual = analytic_rectangle_sine(x, y, lx=1, ly=300, v0=1)
            self.config["geometry"] = {"Lx": 1, "Ly": 300}
            self.config["grid"]["h"] = 0.5
            result = solve_rectangle_sine(self.config)
        self.assertTrue(np.isfinite(result.analytic_potential).all())
        self.assertTrue(np.isfinite(result.electric_field_y).all())
        self.assertEqual(actual[0, 1], 0)
        self.assertEqual(actual[-1, 1], 1)
        self.assertAlmostEqual(actual[-2, 1], np.exp(-np.pi), places=15)

    def test_analytic_solution_matches_direct_formula(self):
        x = np.linspace(0, 1.5, 31)
        y = np.linspace(0, 1, 21)
        expected = np.outer(np.sinh(np.pi * y / 1.5) / np.sinh(np.pi / 1.5),
                            np.sin(np.pi * x / 1.5))
        np.testing.assert_allclose(analytic_rectangle_sine(x, y, lx=1.5, ly=1, v0=1),
                                   expected, rtol=5e-15, atol=1e-15)

    def test_solver_revalidates_input(self):
        self.config["grid"]["h"] = 0.07
        with self.assertRaises(ConfigError):
            solve_rectangle_sine(self.config)


if __name__ == "__main__":
    unittest.main()
