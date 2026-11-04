"""Checks for the parameter-scan design and its scientific limitations."""

import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from electrostatic_bvp.cases import build_parallel_plate
from electrostatic_bvp.config import load_config
from electrostatic_bvp.fields import parallel_plate_field
from electrostatic_bvp.solver import solve_parallel_plate
import scripts.scan_m2 as scan_m2
from scripts.scan_m2 import (
    GEOMETRIES, RUNS, SCANS, analyze_geometry, compare_scaled_geometries, configuration,
    copy_seed, target_region,
)
from scripts.validate_m2 import CaseData, ROOT, save_case


class ScanM2Tests(unittest.TestCase):
    def test_five_geometries_cover_both_scans_with_comparable_resolution(self):
        base = load_config(ROOT / "configs/parallel_plate.json")
        self.assertEqual(len(GEOMETRIES), 5)
        self.assertEqual(set(SCANS["plate_length"]) & set(SCANS["gap"]), {"L4-d1"})
        for label, length, gap in GEOMETRIES:
            target = target_region(length, gap)
            self.assertAlmostEqual(target[1], length / 2 + gap, msg=label)
            for run, (h_ratio, a_ratio, b_ratio) in RUNS.items():
                config = configuration(base, length, gap, run, 2000000)
                grid, geom = config["grid"], config["geometry"]
                self.assertAlmostEqual(grid["h"] / gap, h_ratio, msg=label)
                self.assertAlmostEqual(geom["domain_half_width"] / gap, a_ratio, msg=label)
                self.assertAlmostEqual(geom["domain_half_height"] / gap, b_ratio, msg=label)
                self.assertLess(target[1], geom["domain_half_width"], msg=label)
                self.assertLess(target[3], geom["domain_half_height"], msg=label)

    @staticmethod
    def manufactured(h, a, b):
        config = load_config(ROOT / "configs/parallel_plate.json")
        config["geometry"] = {"plate_length": 0.8, "gap": 0.4,
                              "domain_half_width": a, "domain_half_height": b}
        config["grid"]["h"] = h
        grid = build_parallel_plate(config)
        xg, yg = np.meshgrid(grid.x, grid.y)
        potential = yg / config["geometry"]["gap"]
        field = parallel_plate_field(grid.x, grid.y, potential, grid.positive_plate_mask,
                                     grid.negative_plate_mask, h=h)
        arrays = {"x": grid.x, "y": grid.y, "potential": potential,
                  "electric_field_x": field.electric_field_x,
                  "electric_field_y": field.electric_field_y,
                  "positive_plate_mask": grid.positive_plate_mask,
                  "negative_plate_mask": grid.negative_plate_mask,
                  "field_valid_mask": field.field_valid_mask,
                  "analysis_valid_mask": field.analysis_valid_mask}
        metadata = {"converged": True, "max_abs_laplace_residual_free_nodes": 0.0,
                    "endpoint_exclusion_radius": field.endpoint_exclusion_radius,
                    "endpoint_exclusion_has_two_intervals": field.endpoint_exclusion_has_two_intervals}
        return CaseData(config, arrays, metadata)

    def test_manufactured_uniform_field_and_failed_runs_have_limited_status(self):
        cases = {"coarse": self.manufactured(0.04, 1.2, 0.8),
                 "medium": self.manufactured(0.02, 1.2, 0.8),
                 "fine": self.manufactured(0.01, 1.2, 0.8),
                 "expanded_medium": self.manufactured(0.02, 1.6, 1.2)}
        result = analyze_geometry(0.8, 0.4, cases)
        self.assertEqual(result["reliability_status"], "provisional")
        self.assertIn("Only one domain expansion", result["reliability_note"])
        self.assertAlmostEqual(result["metrics"]["center_negative_ey_over_reference"], 1.0)
        self.assertLess(result["metrics"]["central_max_vector_deviation_over_reference"], 1e-12)
        for pair in result["comparisons"].values():
            self.assertTrue(pair["within_suggested_scale"])
            self.assertFalse(pair["interpolation"])
            self.assertLess(pair["max_field_vector_difference_over_eref"], 1e-12)
        cases["fine"].metadata["converged"] = False
        result = analyze_geometry(0.8, 0.4, cases)
        self.assertEqual(result["reliability_status"], "limited")
        self.assertFalse(result["all_converged"])
        self.assertFalse(result["comparisons"]["mesh_medium_fine"]["within_suggested_scale"])
        cases["fine"].metadata["converged"] = True
        fine_arrays = cases["fine"].arrays
        center = (len(fine_arrays["y"]) // 2, len(fine_arrays["x"]) // 2)
        fine_arrays["electric_field_x"][center] += 0.1
        result = analyze_geometry(0.8, 0.4, cases)
        self.assertTrue(result["all_converged"])
        self.assertEqual(result["reliability_status"], "limited")
        self.assertFalse(result["comparisons"]["mesh_medium_fine"]["within_suggested_scale"])
        fine_arrays["electric_field_y"][center] += 0.1
        result = analyze_geometry(0.8, 0.4, cases)
        self.assertAlmostEqual(result["point_metric_absolute_changes"]["mesh_medium_fine"]
                               ["center_negative_ey_over_reference"], 0.04)

    def test_seed_is_copied_only_with_matching_provenance_and_hash(self):
        config = load_config(ROOT / "configs/parallel_plate.json")
        config["solver"]["Nmax"] = 1
        result = solve_parallel_plate(config)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, destination = root / "source", root / "output" / "run"
            save_case(source, config, result, 0.1, {"python": "example"})
            with self.assertRaisesRegex(ValueError, "source"):
                copy_seed(source, destination, config, {"python": "different"})
            self.assertFalse(destination.exists())
            copy_seed(source, destination, config, {"python": "example"})
            self.assertTrue((destination / "solution.npz").is_file())
            with (source / "solution.npz").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash"):
                copy_seed(source, root / "other", config, {"python": "example"})
            self.assertFalse((root / "other").exists())

    def test_scaled_geometry_uses_dimensionless_coordinates_and_fields(self):
        first = self.manufactured(0.04, 1.2, 0.8)
        second = deepcopy(first)
        for key in second.config["geometry"]:
            second.config["geometry"][key] *= 2
        second.config["grid"]["h"] *= 2
        for key in ("x", "y"):
            second.arrays[key] *= 2
        for key in ("electric_field_x", "electric_field_y"):
            second.arrays[key] /= 2
        result = compare_scaled_geometries(first, second)
        self.assertTrue(result["both_converged"])
        self.assertFalse(result["interpolation"])
        self.assertLess(result["max_normalized_field_vector_difference"], 1e-14)
        center = (len(second.arrays["y"]) // 2, len(second.arrays["x"]) // 2)
        second.arrays["electric_field_x"][center] += 0.1
        self.assertAlmostEqual(compare_scaled_geometries(first, second)
                               ["max_normalized_field_vector_difference"], 0.08)
        second.config["geometry"]["plate_length"] *= 2
        with self.assertRaisesRegex(ValueError, "geometries must match"):
            compare_scaled_geometries(first, second)

    def test_unconverged_scan_writes_diagnostics_and_resumes_without_recomputation(self):
        geometries = (("tiny", 0.4, 0.4),)
        scans = {"plate_length": ("tiny",), "gap": ("tiny",)}
        runs = {"coarse": (0.25, 3.0, 2.0), "medium": (0.125, 3.0, 2.0),
                "fine": (0.0625, 3.0, 2.0), "expanded_medium": (0.125, 4.0, 3.0)}
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(scan_m2, "GEOMETRIES", geometries), \
             patch.object(scan_m2, "SCANS", scans), patch.object(scan_m2, "RUNS", runs), \
             contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp) / "scan"
            arguments = ["--output-dir", str(output), "--nmax", "1"]
            self.assertEqual(scan_m2.main(arguments), 1)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["geometries"]["tiny"]["reliability_status"], "limited")
            self.assertFalse(report["geometries"]["tiny"]["all_converged"])
            for name in report["output_files"]:
                self.assertGreater((output / name).stat().st_size, 100)
            with (output / "metrics.csv").open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(row["h_over_d"]), 0.0625)
            self.assertAlmostEqual(float(row["a_over_d"]), 3.0)
            self.assertAlmostEqual(float(row["b_over_d"]), 2.0)
            solution = output / "runs/tiny-fine/solution.npz"
            digest = hashlib.sha256(solution.read_bytes()).hexdigest()
            self.assertEqual(scan_m2.main([*arguments, "--resume"]), 1)
            self.assertEqual(hashlib.sha256(solution.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
