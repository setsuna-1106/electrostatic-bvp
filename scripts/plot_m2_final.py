"""Plot fine-grid M2 potential and field maps from a verified final report.

The values are computed on h/d=0.005 and sampled on h/d=0.025 only for
rendering. No interpolation or re-solving is performed. Input hashes and the
numerical source hashes recorded by refine_m2.py are checked before plotting.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from refine_m2 import ROOT, file_hash, target_case
from scan_m2 import GEOMETRIES, save_maps
from validate_m2 import CaseData, write_json


def plot_maps(directory):
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report["passed"]:
        raise ValueError("final report has not passed; inspect it before producing final maps")
    for path, expected in report["environment"]["source_sha256"].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"numerical source differs from report: {path}")
    cases, summaries, input_hashes = {}, {}, {}
    for label, length, gap in GEOMETRIES:
        summary = report["geometries"][label]
        if not summary["passed"]:
            raise ValueError(f"geometry has not passed: {label}")
        run = summary["runs"][f"{label}-mesh-0.005"]
        path = directory / run["path"] / "potential.npy"
        if file_hash(path) != run["solution_sha256"]:
            raise ValueError(f"potential hash mismatch: {path}")
        target = (-5 * gap, 5 * gap, -1.5 * gap, 1.5 * gap)
        fine = target_case(run["config"], run, path, target)
        arrays = fine.arrays
        # Sample exact fine-grid nodes on a coarser rendering lattice; gradients
        # have already been evaluated at the original fine spacing.
        def sampled_indices(coordinates):
            scaled = coordinates / (0.025 * gap)
            return np.flatnonzero(np.isclose(scaled, np.rint(scaled), rtol=0, atol=1e-8))
        ix, iy = sampled_indices(arrays["x"]), sampled_indices(arrays["y"])
        region = np.ix_(iy, ix)
        reduced = {"x": arrays["x"][ix], "y": arrays["y"][iy]}
        for key in ("potential", "electric_field_x", "electric_field_y", "analysis_valid_mask"):
            reduced[key] = arrays[key][region]
        cases[label] = {"fine": CaseData(fine.config, reduced, fine.metadata)}
        summaries[label] = {"all_converged": True, "length_over_gap": length / gap, "gap": gap}
        input_hashes[label] = run["solution_sha256"]
        del fine, arrays
        gc.collect()
        print(f"VERIFIED {label}", flush=True)
    save_maps(directory, summaries, cases, field=False)
    save_maps(directory, summaries, cases, field=True)
    manifest = {
        "final_report_sha256": file_hash(report_path),
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_potential_sha256": input_hashes,
        "solution_h_over_d": 0.005, "render_sampling_h_over_d": 0.025,
        "rendering_note": "Field is differentiated on the fine grid before sampling; no interpolation.",
        "output_files": {name: file_hash(directory / name)
                         for name in ("potential-comparison.png", "field-comparison.png")},
    }
    write_json(directory / "maps-manifest.json", manifest)
    print(f"M2 fine-grid maps: {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/m2-final")
    args = parser.parse_args()
    plot_maps(args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print(f"M2 map generation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
