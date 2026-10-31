"""Case-specific validation and exact internal plate construction."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from electrostatic_bvp.cases import build_parallel_plate, build_rectangle_sine
from electrostatic_bvp.config import ConfigError, load_config, save_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


class ParallelPlateCaseTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/parallel_plate.json")

    def test_default_geometry_and_fixed_values(self):
        grid = build_parallel_plate(self.config)
        self.assertEqual(grid.fixed_mask.shape, (81, 121))
        np.testing.assert_array_equal(grid.x, np.arange(-60, 61) * 0.1)
        np.testing.assert_array_equal(grid.y, np.arange(-40, 41) * 0.1)
        self.assertEqual(np.count_nonzero(grid.positive_plate_mask), 41)
        self.assertEqual(np.count_nonzero(grid.negative_plate_mask), 41)
        np.testing.assert_array_equal(np.flatnonzero(grid.positive_plate_mask[45]), np.arange(40, 81))
        np.testing.assert_array_equal(np.flatnonzero(grid.negative_plate_mask[35]), np.arange(40, 81))
        self.assertFalse(np.any(grid.positive_plate_mask & grid.negative_plate_mask))
        np.testing.assert_array_equal(grid.fixed_values[grid.positive_plate_mask], 0.5)
        np.testing.assert_array_equal(grid.fixed_values[grid.negative_plate_mask], -0.5)
        self.assertEqual(np.count_nonzero(grid.fixed_mask), 2 * 121 + 2 * 79 + 82)
        np.testing.assert_array_equal(grid.fixed_values[0, :], 0)
        np.testing.assert_array_equal(grid.fixed_values[-1, :], 0)
        np.testing.assert_array_equal(grid.fixed_values[:, [0, -1]], 0)

    def test_minimum_gap_and_frame_clearance_are_allowed(self):
        self.config["geometry"] = {"plate_length": 0.4, "gap": 0.4,
                                   "domain_half_width": 0.4, "domain_half_height": 0.4}
        grid = build_parallel_plate(self.config)
        self.assertEqual(grid.fixed_mask.shape, (9, 9))
        np.testing.assert_array_equal(grid.fixed_values[6, 2:7], 0.5)
        np.testing.assert_array_equal(grid.fixed_values[2, 2:7], -0.5)

    def test_alignment_and_clearance_rejected(self):
        bad = [
            ("plate_length", 4.1), ("gap", 1.1),
            ("domain_half_width", 6.05), ("domain_half_height", 4.05),
            ("plate_length", 12.0), ("plate_length", 11.8),
            ("gap", 8.0), ("gap", 7.8), ("gap", 0.2),
        ]
        for key, value in bad:
            with self.subTest(key=key, value=value):
                config = deepcopy(self.config)
                config["geometry"][key] = value
                with self.assertRaises(ConfigError):
                    build_parallel_plate(config)

    def test_invalid_numbers_keys_and_outer_boundary(self):
        for key in self.config["geometry"]:
            for value in (True, "1", 0, -1, float("nan"), float("inf")):
                with self.subTest(key=key, value=value):
                    config = deepcopy(self.config)
                    config["geometry"][key] = value
                    with self.assertRaises(ConfigError):
                        validate_config(config)
        for key, value in (("V0", True), ("V0", float("nan")),
                           ("outer_potential", True), ("outer_potential", float("inf")),
                           ("outer_potential", 0.1)):
            config = deepcopy(self.config)
            config["boundary_conditions"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ConfigError):
                validate_config(config)
        for section in ("geometry", "boundary_conditions"):
            for mutation in ("missing", "extra"):
                config = deepcopy(self.config)
                if mutation == "missing":
                    config[section].pop(next(iter(config[section])))
                else:
                    config[section]["unknown"] = 1
                with self.subTest(section=section, mutation=mutation), self.assertRaises(ConfigError):
                    validate_config(config)

    def test_round_trip_zero_and_negative_voltage(self):
        for voltage in (0, -2):
            self.config["boundary_conditions"]["V0"] = voltage
            with tempfile.TemporaryDirectory() as directory:
                path = save_config(self.config, Path(directory) / "case.json")
                self.assertEqual(load_config(path), self.config)
            grid = build_parallel_plate(self.config)
            np.testing.assert_array_equal(grid.fixed_values[grid.positive_plate_mask], voltage / 2)
            np.testing.assert_array_equal(grid.fixed_values[grid.negative_plate_mask], -voltage / 2)

    def test_builder_rejects_wrong_case(self):
        with self.assertRaises(ConfigError):
            build_rectangle_sine(self.config)
        with self.assertRaises(ConfigError):
            build_parallel_plate(load_config(ROOT / "configs/rectangle.json"))


if __name__ == "__main__":
    unittest.main()
