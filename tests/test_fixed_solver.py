"""Jacobi iteration with internal Dirichlet nodes and free-node diagnostics."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from electrostatic_bvp.config import load_config
from electrostatic_bvp.solver import solve_jacobi, solve_parallel_plate


ROOT = Path(__file__).resolve().parents[1]


class FixedSolverTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("numba"), "optional s4 dependency is not installed")
    def test_numba_backend_matches_numpy_jacobi(self):
        fixed = np.zeros((9, 11), dtype=bool)
        fixed[[0, -1], :] = True
        fixed[:, [0, -1]] = True
        fixed[3, 3:8] = True
        fixed[5, 3:8] = True
        values = np.zeros(fixed.shape)
        values[3, 3:8] = 0.5
        values[5, 3:8] = -0.5
        for dx, dy in ((0.1, 0.1), (0.1, 0.2)):
            for tol, nmax in ((1e-12, 1), (1e-12, 43), (1e-11, 10000)):
                with self.subTest(dx=dx, dy=dy, tol=tol, nmax=nmax):
                    numpy = solve_jacobi(fixed, values, dx=dx, dy=dy, tol=tol, nmax=nmax)
                    numba = solve_jacobi(fixed, values, dx=dx, dy=dy, tol=tol, nmax=nmax,
                                         backend="numba")
                    self.assertEqual(numba.iterations, numpy.iterations)
                    self.assertEqual(numba.converged, numpy.converged)
                    np.testing.assert_allclose(numba.potential, numpy.potential, rtol=0, atol=3e-15)
                    np.testing.assert_array_equal(numba.potential[fixed], values[fixed])
                    self.assertAlmostEqual(numba.final_change, numpy.final_change, places=14)
                    self.assertAlmostEqual(numba.max_abs_residual, numpy.max_abs_residual, places=12)

    def test_one_step_and_residual_exclude_fixed_nodes(self):
        fixed = np.zeros((5, 5), dtype=bool)
        fixed[[0, -1], :] = True
        fixed[:, [0, -1]] = True
        fixed[2, 2] = True
        values = np.zeros((5, 5))
        values[2, 2] = 4
        values[1, 1] = 100  # A value at a free node is not a constraint.
        result = solve_jacobi(fixed, values, dx=1, dy=1, tol=1e-12, nmax=1)
        expected = np.zeros((5, 5))
        expected[2, 2] = 4
        expected[1, 2] = expected[2, 1] = expected[2, 3] = expected[3, 2] = 1
        np.testing.assert_array_equal(result.potential, expected)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.final_change, 1)
        # The residual is 2 at the free diagonal nodes; it is -12 on the
        # fixed center, which is not subject to the free-space equation.
        self.assertEqual(result.max_abs_residual, 2)

    def test_fixed_nodes_do_not_drift_and_tolerance_last_step(self):
        fixed = np.zeros((5, 5), dtype=bool)
        fixed[[0, -1], :] = True
        fixed[:, [0, -1]] = True
        fixed[2, 2] = True
        values = np.zeros((5, 5))
        values[2, 2] = -7
        values[0, :] = 3
        result = solve_jacobi(fixed, values, dx=1, dy=1, tol=1e-10, nmax=2000)
        self.assertTrue(result.converged)
        np.testing.assert_array_equal(result.potential[fixed], values[fixed])
        self.assertLessEqual(result.final_change, 1e-10)
        result = solve_jacobi(fixed, values, dx=1, dy=1, tol=2, nmax=1)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)

    def test_invalid_constraints_are_rejected(self):
        fixed = np.ones((3, 3), dtype=bool)
        fixed[1, 1] = False
        values = np.zeros((3, 3))
        for bad_mask, bad_values in ((fixed.astype(int), values),
                                     (fixed, np.zeros((4, 4))),
                                     (np.zeros((3, 3), dtype=bool), values),
                                     (np.ones((3, 3), dtype=bool), values)):
            with self.subTest(mask=bad_mask), self.assertRaises(ValueError):
                solve_jacobi(bad_mask, bad_values, dx=1, dy=1, tol=1e-6, nmax=5)
        for options in ({"dx": 0}, {"dy": float("inf")}, {"tol": 0}, {"nmax": True}):
            kwargs = {"dx": 1, "dy": 1, "tol": 1e-6, "nmax": 5}
            kwargs.update(options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                solve_jacobi(fixed, values, **kwargs)

    def test_parallel_plate_solution_structure_and_voltage(self):
        config = load_config(ROOT / "configs/parallel_plate.json")
        config["geometry"] = {"plate_length": 0.4, "gap": 0.4,
                              "domain_half_width": 0.6, "domain_half_height": 0.6}
        config["solver"]["tol"] = 1e-11
        result = solve_parallel_plate(config)
        self.assertTrue(result.converged)
        np.testing.assert_array_equal(result.potential[result.positive_plate_mask], 0.5)
        np.testing.assert_array_equal(result.potential[result.negative_plate_mask], -0.5)
        np.testing.assert_allclose(result.potential, result.potential[:, ::-1], atol=1e-15)
        np.testing.assert_allclose(result.potential, -result.potential[::-1, :], atol=1e-15)
        self.assertGreaterEqual(result.potential.min(), -0.5)
        self.assertLessEqual(result.potential.max(), 0.5)
        u = result.potential
        residual = (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1]
                    + u[2:, 1:-1] - 4 * u[1:-1, 1:-1]) / 0.1**2
        free = ~result.fixed_mask[1:-1, 1:-1]
        self.assertAlmostEqual(result.max_abs_residual, np.max(np.abs(residual[free])), places=12)
        config["boundary_conditions"]["V0"] = -2
        negative = solve_parallel_plate(config)
        np.testing.assert_allclose(negative.potential, -2 * result.potential, atol=5e-11)
        config["boundary_conditions"]["V0"] = 0
        zero = solve_parallel_plate(config)
        self.assertTrue(zero.converged)
        self.assertEqual(zero.iterations, 1)
        self.assertEqual(zero.max_abs_residual, 0)
        np.testing.assert_array_equal(zero.potential, 0)


if __name__ == "__main__":
    unittest.main()
