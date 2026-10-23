"""Configuration contract shared by the formatter and solver."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from electrostatic_bvp.config import ConfigError, load_config, save_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/rectangle.json")

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_config(self.config, Path(directory) / "nested/case.json")
            self.assertEqual(load_config(path), self.config)

    def test_keys_are_required_and_unknown_keys_rejected(self):
        for section in (None, "geometry", "boundary_conditions", "grid", "solver"):
            for mutation in ("missing", "extra"):
                with self.subTest(section=section, mutation=mutation):
                    config = deepcopy(self.config)
                    target = config if section is None else config[section]
                    if mutation == "missing":
                        target.pop(next(iter(target)))
                    else:
                        target["typo"] = 1
                    with self.assertRaises(ConfigError):
                        validate_config(config)

    def test_invalid_types_and_numeric_ranges(self):
        cases = [
            ("geometry", "Lx", [True, "1.5", 0, -1, float("inf"), float("nan")]),
            ("geometry", "Ly", [None, 0, -1]),
            ("boundary_conditions", "V0", [True, "1", float("inf"), float("nan")]),
            ("grid", "h", [False, 0, -0.1, 0.07, 1, float("inf")]),
            ("solver", "tol", [True, 0, -1, float("nan")]),
            ("solver", "Nmax", [True, 0, -1, 1.5, "100"]),
            ("solver", "method", ["sor", None]),
        ]
        for section, key, values in cases:
            for value in values:
                with self.subTest(section=section, key=key, value=value):
                    config = deepcopy(self.config)
                    config[section][key] = value
                    with self.assertRaises(ConfigError):
                        validate_config(config)

    def test_invalid_objects_and_case(self):
        for value in (None, [], "rectangle_sine", 42):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                validate_config(value)
        for key in ("geometry", "boundary_conditions", "grid", "solver"):
            config = deepcopy(self.config)
            config[key] = []
            with self.subTest(key=key), self.assertRaises(ConfigError):
                validate_config(config)
        self.config["case"] = "parallel_plate"
        with self.assertRaises(ConfigError):
            validate_config(self.config)

    def test_unreadable_and_malformed_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            with self.assertRaises(ConfigError):
                load_config(path)
            path.write_text('{"case":', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "Invalid JSON"):
                load_config(path)
            path.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
