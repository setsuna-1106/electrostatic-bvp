"""Check the open-space model against quadrature and physical constraints."""

import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from electrostatic_bvp.config import load_config
from electrostatic_bvp.open_plate import fit_open_plate, solve_open_plate


ROOT = Path(__file__).resolve().parents[1]


class OpenPlateTests(unittest.TestCase):
    def test_potential_and_field_agree_with_direct_log_kernel_quadrature(self):
        model = fit_open_plate(4.0, 1.0, 1.0)
        x = np.array([-2.5, -0.8, 0, 1.3, 2.2, 4.0])
        y = np.array([0.5, 0.7, 0, -0.4, 0.3, -2.0])
        theta = (np.arange(1024) + 0.5) * np.pi / 1024
        source = 2 * np.cos(theta)
        weight = np.polynomial.chebyshev.chebval(np.cos(theta), model.coefficients)
        dx = x[:, None] - source
        dy1, dy2 = y[:, None] - 0.5, y[:, None] + 0.5
        r1, r2 = dx**2 + dy1**2, dx**2 + dy2**2
        expected_potential = np.mean(weight * 0.5 * np.log(r2 / r1), axis=1)
        expected_ex = np.mean(weight * dx * (1 / r1 - 1 / r2), axis=1)
        expected_ey = np.mean(weight * (dy1 / r1 - dy2 / r2), axis=1)
        np.testing.assert_allclose(model.potential(x, y), expected_potential, rtol=0, atol=2e-13)
        np.testing.assert_allclose(model.field(x, y), (expected_ex, expected_ey), rtol=0, atol=2e-13)

    def test_plates_far_field_and_endpoint_scaling(self):
        for aspect in (2.0, 4.0, 8.0):
            with self.subTest(aspect=aspect):
                model = fit_open_plate(aspect, 1.0, -2.0)
                x = np.linspace(-aspect / 2, aspect / 2, 517)
                np.testing.assert_allclose(model.potential(x, 0.5), -1.0, rtol=0, atol=2e-11)
                np.testing.assert_allclose(model.potential(x, -0.5), 1.0, rtol=0, atol=2e-11)
                near, far = model.potential(0, np.array([100.0, 200.0]))
                self.assertAlmostEqual(near / far, 2.0, delta=0.002)
                # On the plate extension the field grows as distance**(-1/2).
                ex, ey = model.field(aspect / 2 + np.array([1e-5, 4e-5]), 0.5)
                near_tip, far_tip = np.hypot(ex, ey)
                self.assertAlmostEqual(near_tip / far_tip, 2.0, delta=0.002)

    def test_output_window_and_sampling_do_not_change_the_shared_solution(self):
        config = load_config(ROOT / "configs/parallel_plate.json")
        second = deepcopy(config)
        second["geometry"].update(domain_half_width=3.0, domain_half_height=2.0)
        second["grid"]["h"] = 0.05
        first, latter = solve_open_plate(config), solve_open_plate(second)
        np.testing.assert_array_equal(first.fixed_mask, ~first.field.field_valid_mask)
        np.testing.assert_allclose(latter.potential[::2, ::2], first.potential[20:61, 30:91],
                                   rtol=0, atol=2e-14)
        np.testing.assert_allclose(latter.field.electric_field_y[::2, ::2],
                                   first.field.electric_field_y[20:61, 30:91], rtol=0, atol=2e-14)
        self.assertTrue(np.isnan(first.field.electric_field_y[first.fixed_mask]).all())


if __name__ == "__main__":
    unittest.main()
