"""Independent checks for the optional M2 validation warm start."""

from copy import deepcopy
import importlib.util
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from electrostatic_bvp.cases import build_parallel_plate
from electrostatic_bvp.config import load_config
from electrostatic_bvp.solver import solve_parallel_plate


@unittest.skipUnless(importlib.util.find_spec("numba"), "optional Numba dependency unavailable")
class WarmStartTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/parallel_plate.json")

    def test_padded_fine_mask_and_values_match_the_original_geometry(self):
        from m2_warm_start import _masks_and_initial
        initial, masks, shape = _masks_and_initial(self.config)
        original = build_parallel_plate(self.config)
        ny, nx = shape
        np.testing.assert_array_equal(masks[0][:ny, :nx], original.fixed_mask)
        np.testing.assert_array_equal(initial[:ny, :nx], original.fixed_values)
        self.assertTrue(masks[0][ny:, :].all())
        self.assertTrue(masks[0][:, nx:].all())

    def test_warm_start_matches_zero_start_jacobi_and_residual(self):
        from m2_warm_start import solve_with_warm_start
        cold_config = deepcopy(self.config)
        cold_config["solver"]["tol"] = 1e-11
        cold = solve_parallel_plate(cold_config, backend="numba")
        self.config["solver"]["tol"] = 1e-14
        warm = solve_with_warm_start(self.config)
        self.assertTrue(cold.converged and warm.converged)
        self.assertLessEqual(warm.final_change, self.config["solver"]["tol"])
        # The cold residual's discrete maximum-principle bound covers the
        # different stopping tolerances and is independent of the warm solver.
        radius = min(self.config["geometry"]["domain_half_width"],
                     self.config["geometry"]["domain_half_height"])
        bound = (cold.max_abs_residual + warm.max_abs_residual) * radius**2 / 2
        self.assertLessEqual(np.max(np.abs(cold.potential - warm.potential)), bound + 1e-13)
        h = self.config["grid"]["h"]
        u = warm.potential
        laplacian = ((u[1:-1, :-2] - 2 * u[1:-1, 1:-1] + u[1:-1, 2:]) / h**2
                     + (u[:-2, 1:-1] - 2 * u[1:-1, 1:-1] + u[2:, 1:-1]) / h**2)
        free = ~cold.fixed_mask[1:-1, 1:-1]
        self.assertAlmostEqual(warm.max_abs_residual, np.max(np.abs(laplacian[free])), delta=1e-13)
        np.testing.assert_array_equal(u[cold.fixed_mask], cold.potential[cold.fixed_mask])

    def test_insufficient_warm_start_does_not_silently_accept(self):
        from m2_warm_start import solve_with_warm_start
        with self.assertRaisesRegex(ValueError, "did not reach"):
            solve_with_warm_start(self.config, max_cycles=1)

    def test_target_crop_reconstructs_fields_on_common_nodes(self):
        from refine_m2 import target_case
        grid = build_parallel_plate(self.config)
        x, y = np.meshgrid(grid.x, grid.y)
        potential = 2 * x - 3 * y
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "potential.npy"
            np.save(path, potential)
            case = target_case(self.config, {}, path, (-3., 3., -1.5, 1.5))
        xx = np.flatnonzero(np.isclose(case.arrays["x"], 0))[0]
        yy = np.flatnonzero(np.isclose(case.arrays["y"], 0))[0]
        self.assertAlmostEqual(case.arrays["electric_field_x"][yy, xx], -2.)
        self.assertAlmostEqual(case.arrays["electric_field_y"][yy, xx], 3.)
        self.assertFalse(case.arrays["field_valid_mask"][case.arrays["positive_plate_mask"]].any())


if __name__ == "__main__":
    unittest.main()
