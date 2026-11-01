"""Field masks, stencils and normalized parallel-plate comparisons."""

import unittest
from pathlib import Path

import numpy as np

from electrostatic_bvp.cases import build_parallel_plate
from electrostatic_bvp.config import load_config
from electrostatic_bvp.fields import parallel_plate_field, parallel_plate_metrics
from electrostatic_bvp.solver import solve_parallel_plate
from electrostatic_bvp.visualize import _arrow_lengths


ROOT = Path(__file__).resolve().parents[1]


class PlateFieldTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/parallel_plate.json")
        self.grid = build_parallel_plate(self.config)
        self.h = self.config["grid"]["h"]

    def test_stencil_uses_correct_side_and_masks_plates(self):
        x, y = np.meshgrid(self.grid.x, self.grid.y)
        # A continuous potential with different gradients on the two sides of
        # the upper plate: crossing it would blend the two physical fields.
        u = np.where(y >= 0.5, y - 0.5, 2 * (y - 0.5))
        field = parallel_plate_field(self.grid.x, self.grid.y, u,
                                     self.grid.positive_plate_mask,
                                     self.grid.negative_plate_mask, h=self.h)
        plate = self.grid.positive_plate_mask | self.grid.negative_plate_mask
        np.testing.assert_array_equal(field.field_valid_mask, ~plate)
        self.assertTrue(np.isnan(field.electric_field_x[plate]).all())
        self.assertTrue(np.isnan(field.electric_field_y[plate]).all())
        self.assertTrue(np.isfinite(field.electric_field_y[~plate]).all())
        self.assertAlmostEqual(field.electric_field_y[46, 60], -1)
        self.assertAlmostEqual(field.electric_field_y[44, 60], -2)
        self.assertFalse(field.analysis_valid_mask[45, 81])  # 0.1 from an endpoint
        self.assertTrue(field.analysis_valid_mask[45, 83])   # 0.3 from an endpoint
        self.assertAlmostEqual(field.endpoint_exclusion_radius, 0.2)
        self.assertTrue(field.endpoint_exclusion_has_two_intervals)

    def test_reference_comparisons_and_negative_voltage(self):
        _, y = np.meshgrid(self.grid.x, self.grid.y)
        for v0 in (1, -2):
            field = parallel_plate_field(self.grid.x, self.grid.y, v0 * y,
                                         self.grid.positive_plate_mask,
                                         self.grid.negative_plate_mask, h=self.h)
            metrics = parallel_plate_metrics(self.grid.x, self.grid.y, field,
                                             self.grid.positive_plate_mask,
                                             self.grid.negative_plate_mask,
                                             v0=v0, h=self.h)
            self.assertAlmostEqual(metrics["center_negative_ey_over_reference"], np.sign(v0))
            self.assertLess(metrics["central_max_vector_deviation_over_reference"], 1e-14)
            self.assertAlmostEqual(metrics["outside_negative_ey_over_reference"], np.sign(v0))
            self.assertEqual(metrics["outside_point_xy"], [2.5, 0])

    def test_outside_comparison_point_may_be_outside_domain(self):
        self.config["geometry"] = {"plate_length": 0.4, "gap": 0.8,
                                   "domain_half_width": 0.4, "domain_half_height": 0.6}
        result = solve_parallel_plate(self.config)
        metrics = parallel_plate_metrics(result.x, result.y, result.field,
                                         result.positive_plate_mask, result.negative_plate_mask,
                                         v0=1, h=self.h)
        self.assertIsNone(metrics["outside_negative_ey_over_reference"])
        self.assertIn("outside", metrics["outside_point_unavailable_reason"])

    def test_solved_field_direction_and_reproducibility(self):
        result = solve_parallel_plate(self.config)
        again = solve_parallel_plate(self.config)
        self.assertTrue(result.converged)
        self.assertLess(result.field.electric_field_y[len(result.y) // 2, len(result.x) // 2], 0)
        np.testing.assert_array_equal(result.field.field_valid_mask, again.field.field_valid_mask)
        np.testing.assert_array_equal(result.field.electric_field_x, again.field.electric_field_x)
        np.testing.assert_array_equal(result.field.electric_field_y, again.field.electric_field_y)

    def test_arrows_cannot_cross_either_conductor(self):
        x = np.array([0.0, 0.0, -2.1, 3.0])
        y = np.array([0.6, 0.4, 0.5, 0.0])
        ux = np.array([0.0, 0.0, 1.0, 1.0])
        uy = np.array([-1.0, 1.0, 0.0, 0.0])
        lengths = _arrow_lengths(x, y, ux, uy, 1.0, -2, 2, (-0.5, 0.5), 0.1)
        np.testing.assert_allclose(lengths, [0.08, 0.08, 0.08, 1.0], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
