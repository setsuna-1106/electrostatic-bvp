"""Exercise the public commands, saved arrays, metadata, and exit codes."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from electrostatic_bvp.config import load_config, save_config


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_command(self, *arguments):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        return subprocess.run([sys.executable, "-W", "error", *map(str, arguments)],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)

    def test_formatter_and_solver_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            output = Path(directory) / "output"
            formatted = self.run_command("scripts/format_input.py", "--output", config_path)
            self.assertEqual(formatted.returncode, 0, formatted.stderr)
            self.assertEqual(load_config(config_path), load_config(ROOT / "configs/rectangle.json"))
            run = self.run_command("-m", "electrostatic_bvp.main", "--config", config_path,
                                   "--output-dir", output)
            self.assertEqual(run.returncode, 0, run.stderr)
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["solver"]["converged"])
            self.assertEqual(summary["solver"]["stop_reason"], "tolerance_reached")
            self.assertEqual(summary["config"], load_config(config_path))
            self.assertEqual(summary["grid"]["array_shape_yx"], [21, 31])
            with np.load(output / "solution.npz", allow_pickle=False) as arrays:
                self.assertEqual(set(arrays.files), {"x", "y", "potential", "analytic_potential",
                                                    "absolute_error", "electric_field_x", "electric_field_y"})
                for name in arrays.files:
                    self.assertTrue(np.isfinite(arrays[name]).all(), name)
                for name in ("potential", "analytic_potential", "absolute_error",
                             "electric_field_x", "electric_field_y"):
                    self.assertEqual(arrays[name].shape, (21, 31))
                np.testing.assert_array_equal(arrays["absolute_error"],
                                              np.abs(arrays["potential"] - arrays["analytic_potential"]))
                self.assertAlmostEqual(summary["analytic_comparison"]["max_abs_error"],
                                       arrays["absolute_error"][1:-1, 1:-1].max())
                self.assertLess(summary["analytic_comparison"]["max_abs_error"], 3e-4)

    def test_unconverged_run_saves_diagnostics_and_returns_one(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = load_config(ROOT / "configs/rectangle.json")
            config["solver"]["Nmax"] = 1
            save_config(config, path / "config.json")
            run = self.run_command("-m", "electrostatic_bvp.main", "--config", path / "config.json",
                                   "--output-dir", path / "result")
            self.assertEqual(run.returncode, 1, run.stderr)
            summary = json.loads((path / "result/summary.json").read_text())
            self.assertFalse(summary["solver"]["converged"])
            self.assertEqual(summary["solver"]["stop_reason"], "maximum_iterations")
            self.assertTrue((path / "result/solution.npz").is_file())

    def test_bad_configuration_and_output_failure_return_two(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            bad_config = path / "bad.json"
            bad_config.write_text("{}")
            output = path / "output"
            run = self.run_command("-m", "electrostatic_bvp.main", "--config", bad_config,
                                   "--output-dir", output)
            self.assertEqual(run.returncode, 2)
            self.assertFalse(output.exists())
            self.assertNotIn("Traceback", run.stderr)
            output.write_text("occupied")
            run = self.run_command("-m", "electrostatic_bvp.main", "--output-dir", output)
            self.assertEqual(run.returncode, 2)
            self.assertNotIn("Traceback", run.stderr)
            self.assertEqual(output.read_text(), "occupied")

    def test_formatter_rejects_invalid_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bad.json"
            run = self.run_command("scripts/format_input.py", "--h", "0.07", "--output", output)
            self.assertEqual(run.returncode, 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
