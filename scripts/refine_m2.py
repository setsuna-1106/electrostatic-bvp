"""Finish M2 mesh/domain checks using Jacobi with a multigrid initial guess.

The original S4 zero-start Jacobi records are rechecked; each of the five S5
geometries is separately solved. Every new run retains its full potential,
configuration, residual, timings, environment and source/data hashes. Reports
must not be interpreted as a rigorous total-error bound for open space.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import numba

from electrostatic_bvp.config import load_config, validate_config
from electrostatic_bvp.fields import parallel_plate_field
from m2_warm_start import solve_with_warm_start
from scan_m2 import GEOMETRIES, SCANS, descriptive_metrics, target_region
from validate_m2 import (
    ROOT, DOMAIN_MESH_TARGET, ITERATION_TARGET, CaseData, algebraic_bounds,
    compare_cases, load_case, make_config, meets_tolerance, source_hashes,
    structural_checks, write_json,
)


DOMAIN_SIZES = ((6., 4.), (9., 6.), (12., 8.), (18., 12.), (24., 16.), (36., 24.))
MESH_SIZES = (0.02, 0.01, 0.005)
TOLERANCE = 1e-14
NMAX = 2_000_000


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance():
    sources = source_hashes()
    for path in ("scripts/refine_m2.py", "scripts/m2_warm_start.py", "scripts/validate_m2.py", "scripts/scan_m2.py"):
        sources[path] = file_hash(ROOT / path)
    return {
        "python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__,
        "machine": platform.machine(), "platform": platform.platform(),
        "backend": "numba_jacobi_with_multigrid_initial_guess", "source_sha256": sources,
    }


def config_for(base, length, gap, h_over_d, a_over_d, b_over_d):
    config = make_config(base, h=h_over_d * gap, a=a_over_d * gap, b=b_over_d * gap,
                         tol=TOLERANCE, nmax=NMAX)
    config["geometry"]["plate_length"] = length
    config["geometry"]["gap"] = gap
    return validate_config(config)


def target_case(config, metadata, potential_path, target):
    """Load only a target rectangle plus a two-node halo for centered fields."""
    geo = config["geometry"]
    h = float(config["grid"]["h"])
    ax = round(float(geo["domain_half_width"]) / h)
    by = round(float(geo["domain_half_height"]) / h)
    x_all = np.arange(-ax, ax + 1, dtype=np.float64) * h
    y_all = np.arange(-by, by + 1, dtype=np.float64) * h
    xmin, xmax, ymin, ymax = target
    eps = h * 1e-8
    ix = np.flatnonzero((x_all >= xmin - 2 * h - eps) & (x_all <= xmax + 2 * h + eps))
    iy = np.flatnonzero((y_all >= ymin - 2 * h - eps) & (y_all <= ymax + 2 * h + eps))
    if not len(ix) or not len(iy):
        raise ValueError("empty analysis region")
    x, y = x_all[ix], y_all[iy]
    array = np.load(potential_path, mmap_mode="r", allow_pickle=False)
    if array.shape != (len(y_all), len(x_all)) or array.dtype != np.float64:
        raise ValueError(f"unexpected potential shape or dtype in {potential_path}")
    potential = np.array(array[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1])
    del array
    positive = np.zeros(potential.shape, dtype=bool)
    negative = np.zeros(potential.shape, dtype=bool)
    rows_positive = np.flatnonzero(np.abs(y - geo["gap"] / 2) <= eps)
    rows_negative = np.flatnonzero(np.abs(y + geo["gap"] / 2) <= eps)
    cols = np.flatnonzero(np.abs(x) <= geo["plate_length"] / 2 + eps)
    if len(rows_positive) != 1 or len(rows_negative) != 1 or len(cols) == 0:
        raise ValueError("target region does not contain both complete plates")
    positive[rows_positive[0], cols] = True
    negative[rows_negative[0], cols] = True
    field = parallel_plate_field(x, y, potential, positive, negative, h=h)
    arrays = {"x": x, "y": y, "potential": potential,
              "electric_field_x": field.electric_field_x, "electric_field_y": field.electric_field_y,
              "positive_plate_mask": positive, "negative_plate_mask": negative,
              "field_valid_mask": field.field_valid_mask, "analysis_valid_mask": field.analysis_valid_mask}
    return CaseData(config, arrays, metadata)


def run_case(label, config, output, numerical_environment, target, *, resume):
    directory = output / "runs" / label
    data_path = directory / "potential.npy"
    meta_path = directory / "run.json"
    if directory.exists():
        if not resume:
            raise FileExistsError(f"{directory} already exists; use --resume")
        if not (data_path.is_file() and meta_path.is_file()):
            raise ValueError(f"incomplete run {directory}; inspect it before resuming")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["config"] != config or meta["provenance"] != numerical_environment:
            raise ValueError(f"config, numerical environment or source changed: {directory}")
        if file_hash(data_path) != meta["solution_sha256"]:
            raise ValueError(f"potential hash mismatch: {directory}")
        print(f"REUSED {label}", flush=True)
    else:
        print(f"START {label} h/d={config['grid']['h']/config['geometry']['gap']:g} "
              f"frame/d=({config['geometry']['domain_half_width']/config['geometry']['gap']:g},"
              f"{config['geometry']['domain_half_height']/config['geometry']['gap']:g})", flush=True)
        directory.mkdir(parents=True)
        started = time.perf_counter()
        def progress(cycle, change):
            if cycle % 20 == 0:
                print(f"  {label} warm-start cycle={cycle} max Jacobi change={change:.3g}", flush=True)
        result = solve_with_warm_start(config, progress=progress)
        temporary = directory / "potential.tmp.npy"
        np.save(temporary, result.potential, allow_pickle=False)
        temporary.replace(data_path)
        meta = {
            "config": config, "grid_shape_yx": list(result.potential.shape),
            "padded_shape_yx": list(result.padded_shape_yx),
            "coarse_grid_shapes_yx": [list(shape) for shape in result.coarse_grid_shapes_yx],
            "converged": result.converged,
            "stop_reason": "tolerance_reached" if result.converged else "maximum_iterations",
            "iterations": result.iterations, "final_change": result.final_change,
            "max_abs_laplace_residual_free_nodes": result.max_abs_residual,
            "warm_start_cycles": result.warm_start_cycles,
            "warm_start_max_jacobi_change": result.warm_start_max_change,
            "warm_start_seconds": result.warm_start_seconds, "jacobi_seconds": result.jacobi_seconds,
            "wall_seconds_including_save": time.perf_counter() - started,
            "endpoint_exclusion_radius": 0.2 * float(config["geometry"]["gap"]),
            "endpoint_exclusion_has_two_intervals": bool(0.2 * float(config["geometry"]["gap"]) / config["grid"]["h"] >= 2 - 1e-9),
            "solution_sha256": file_hash(data_path), "provenance": numerical_environment,
        }
        write_json(meta_path, meta)
        print(f"DONE {label} converged={result.converged} cycles={result.warm_start_cycles} "
              f"jacobi={result.iterations} residual={result.max_abs_residual:.3g} "
              f"seconds={meta['wall_seconds_including_save']:.1f}", flush=True)
        del result
        gc.collect()
    return target_case(config, meta, data_path, target)


def checked_legacy_case(directory, expected):
    meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if meta["provenance"]["source_sha256"] != source_hashes():
        raise ValueError(f"legacy solver source differs: {directory}")
    return load_case(directory, expected, meta["provenance"])


def legacy_checks(base, legacy_dir):
    definitions = {
        "tol-1e-7": {"tol": 1e-7}, "tol-1e-9": {"tol": 1e-9},
        "tol-1e-11": {}, "voltage-negative": {"voltage": -1.}, "voltage-double": {"voltage": 2.},
    }
    cases = {label: checked_legacy_case(legacy_dir / "runs" / label,
                                       make_config(base, **kwargs))
             for label, kwargs in definitions.items()}
    structure = structural_checks(cases["tol-1e-11"], cases["voltage-negative"], cases["voltage-double"])
    tolerance = {}
    for left, right in (("tol-1e-7", "tol-1e-9"), ("tol-1e-9", "tol-1e-11")):
        comparison = compare_cases(cases[left], cases[right])
        comparison["passed"] = meets_tolerance(comparison, ITERATION_TARGET, require_iteration_bound=False)
        tolerance[f"{left}_vs_{right}"] = comparison
    return {
        "path": str(legacy_dir), "structure": structure, "tolerance_comparisons": tolerance,
        "input_solution_sha256": {key: value.metadata["solution_sha256"] for key, value in cases.items()},
        "passed": structure["passed"] and all(item["passed"] for item in tolerance.values()),
    }


def annotate(comparison):
    comparison["within_suggested_scale"] = meets_tolerance(
        comparison, DOMAIN_MESH_TARGET, require_iteration_bound=True)
    return comparison


def report_run(case, relative_path):
    return {**case.metadata, "path": relative_path, "algebraic_error_bounds": algebraic_bounds(case)}


def write_metrics(output, summaries):
    columns = ("geometry", "L", "d", "L_over_d", "selected_a_over_d", "selected_b_over_d",
               "center_negative_ey_over_reference", "outside_negative_ey_over_reference",
               "central_max_vector_deviation_over_reference", "mesh_002_001_field_difference",
               "mesh_001_0005_field_difference", "fine_domain_field_difference", "passed")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        for label, summary in summaries.items():
            metrics = summary["metrics_fine"]
            writer.writerow({
                "geometry": label, "L": summary["plate_length"], "d": summary["gap"],
                "L_over_d": summary["length_over_gap"],
                "selected_a_over_d": summary["selected_domain_over_d"][0],
                "selected_b_over_d": summary["selected_domain_over_d"][1],
                "center_negative_ey_over_reference": metrics["center_negative_ey_over_reference"],
                "outside_negative_ey_over_reference": metrics["outside_negative_ey_over_reference"],
                "central_max_vector_deviation_over_reference": metrics["central_max_vector_deviation_over_reference"],
                "mesh_002_001_field_difference": summary["mesh_comparisons"]["0.02_vs_0.01"]["max_field_vector_difference_over_eref"],
                "mesh_001_0005_field_difference": summary["mesh_comparisons"]["0.01_vs_0.005"]["max_field_vector_difference_over_eref"],
                "fine_domain_field_difference": summary["fine_domain_comparison"]["max_field_vector_difference_over_eref"],
                "passed": summary["passed"],
            })


def write_profiles(output, profiles, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for axis, (name, labels) in zip(axes, SCANS.items()):
        for label in labels:
            x, ey = profiles[label]
            summary = summaries[label]
            flag = "" if summary["passed"] else " (limited)"
            axis.plot(x, ey, label=f"L/d={summary['length_over_gap']:g}{flag}")
        axis.axhline(1, color="0.5", linestyle="--", linewidth=0.8, label="uniform-field reference")
        axis.set_xlabel("x/d")
        axis.set_ylabel(r"$-E_y/E_{ref}$")
        axis.set_title("Varying plate length" if name == "plate_length" else "Varying plate gap")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Finite plates: midline field; h/d = 0.005, finite grounded frame")
    fig.savefig(output / "midline-comparison.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/m2-final")
    parser.add_argument("--legacy-s4", type=Path, default=ROOT / "results/m2-validation-s4-20261101")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir
    if output.exists() and not args.resume:
        parser.error(f"{output} already exists; use --resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    base = load_config(ROOT / "configs/parallel_plate.json")
    env = provenance()
    legacy = legacy_checks(base, args.legacy_s4)
    print(f"LEGACY structural/tolerance passed={legacy['passed']}", flush=True)
    # Cross-check a genuinely different initialization against the previous
    # zero-start Jacobi fine run, rather than comparing two warm starts.
    reference_target = target_region(4., 1.)
    reference_config = config_for(base, 4., 1., 0.025, 12., 8.)
    warm_reference = run_case("jacobi-crosscheck-h0025", reference_config, output, env, reference_target, resume=args.resume)
    legacy_config = make_config(base, h=0.025, a=12., b=8.)
    old_reference = checked_legacy_case(args.legacy_s4 / "runs/mesh-h0025", legacy_config)
    crosscheck = compare_cases(old_reference, warm_reference, reference_target)
    crosscheck["passed"] = meets_tolerance(crosscheck, ITERATION_TARGET, require_iteration_bound=True)
    crosscheck["legacy_solution_sha256"] = old_reference.metadata["solution_sha256"]
    crosscheck["warm_solution_sha256"] = warm_reference.metadata["solution_sha256"]
    print(f"CROSSCHECK passed={crosscheck['passed']} potential="
          f"{crosscheck['max_potential_difference_over_abs_v0']:.3g} field="
          f"{crosscheck['max_field_vector_difference_over_eref']:.3g}", flush=True)
    del old_reference, warm_reference
    summaries, profiles = {}, {}
    for label, length, gap in GEOMETRIES:
        target = target_region(length, gap)
        runs, domain_pairs, current = {}, {}, None
        consecutive = 0
        selected_index = None
        for index, (a, b) in enumerate(DOMAIN_SIZES):
            name = f"{label}-domain-{a:g}-{b:g}"
            config = config_for(base, length, gap, 0.1, a, b)
            case = run_case(name, config, output, env, target, resume=args.resume)
            runs[name] = report_run(case, f"runs/{name}/")
            if current is not None:
                previous_a, previous_b = DOMAIN_SIZES[index - 1]
                pair_name = f"({previous_a:g},{previous_b:g})_vs_({a:g},{b:g})"
                comparison = annotate(compare_cases(current, case, target))
                domain_pairs[pair_name] = comparison
                consecutive = consecutive + 1 if comparison["within_suggested_scale"] else 0
                print(f"DOMAIN {label} {pair_name} field={comparison['max_field_vector_difference_over_eref']:.5g} "
                      f"passed={comparison['within_suggested_scale']}", flush=True)
                if index >= 2 and consecutive >= 2:
                    selected_index = index - 2
                    break
            current = case
        coarse_domain_passed = selected_index is not None
        if selected_index is None:
            selected_index = len(DOMAIN_SIZES) - 2
        a, b = DOMAIN_SIZES[selected_index]
        big_a, big_b = DOMAIN_SIZES[selected_index + 1]
        mesh_cases = []
        for h in MESH_SIZES:
            name = f"{label}-mesh-{h:g}"
            config = config_for(base, length, gap, h, a, b)
            case = run_case(name, config, output, env, target, resume=args.resume)
            runs[name] = report_run(case, f"runs/{name}/")
            mesh_cases.append(case)
        fine_name = f"{label}-fine-expanded"
        config = config_for(base, length, gap, MESH_SIZES[-1], big_a, big_b)
        expanded = run_case(fine_name, config, output, env, target, resume=args.resume)
        runs[fine_name] = report_run(expanded, f"runs/{fine_name}/")
        mesh_pairs = {
            "0.02_vs_0.01": annotate(compare_cases(mesh_cases[0], mesh_cases[1], target)),
            "0.01_vs_0.005": annotate(compare_cases(mesh_cases[1], mesh_cases[2], target)),
        }
        fine_domain = annotate(compare_cases(mesh_cases[2], expanded, target))
        metrics = descriptive_metrics(mesh_cases[2])
        metric_prev, metric_expanded = descriptive_metrics(mesh_cases[1]), descriptive_metrics(expanded)
        metric_keys = ("center_negative_ey_over_reference", "outside_negative_ey_over_reference")
        metric_changes = {
            "last_mesh": {k: abs(metrics[k] - metric_prev[k]) for k in metric_keys},
            "fine_domain": {k: abs(metrics[k] - metric_expanded[k]) for k in metric_keys},
        }
        fine = mesh_cases[2]
        center_row = int(np.argmin(np.abs(fine.arrays["y"])))
        profiles[label] = (fine.arrays["x"] / gap,
                           -fine.arrays["electric_field_y"][center_row, :] / (1. / gap))
        passed = (coarse_domain_passed and all(p["within_suggested_scale"] for p in mesh_pairs.values())
                  and fine_domain["within_suggested_scale"] and all(v["converged"] for v in runs.values()))
        summaries[label] = {
            "plate_length": length, "gap": gap, "length_over_gap": length / gap,
            "target_region_xy": list(target), "selected_domain_over_d": [a, b],
            "expanded_domain_over_d": [big_a, big_b], "coarse_domain_passed": coarse_domain_passed,
            "domain_comparisons": domain_pairs, "mesh_comparisons": mesh_pairs,
            "fine_domain_comparison": fine_domain, "metrics_fine": metrics,
            "point_metric_absolute_changes": metric_changes, "runs": runs, "passed": bool(passed),
        }
        print(f"GEOMETRY {label} passed={passed} mesh_field="
              f"{mesh_pairs['0.02_vs_0.01']['max_field_vector_difference_over_eref']:.5g},"
              f"{mesh_pairs['0.01_vs_0.005']['max_field_vector_difference_over_eref']:.5g} "
              f"fine_domain_field={fine_domain['max_field_vector_difference_over_eref']:.5g}", flush=True)
        del mesh_cases, expanded, case, current, fine
        gc.collect()
        # A checkpoint makes successful individual geometry results inspectable
        # even if a later, larger run is interrupted.
        write_json(output / "partial-report.json", {"legacy": legacy, "crosscheck": crosscheck, "geometries": summaries})
    write_metrics(output, summaries)
    write_profiles(output, profiles, summaries)
    passed = legacy["passed"] and crosscheck["passed"] and all(s["passed"] for s in summaries.values())
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "2D zero-thickness fixed-potential plates, finite zero-potential outer frame",
        "environment": env,
        "method": "five-point Jacobi; multigrid correction precomputes the initial guess; original Numba Jacobi stopping step and independent residual",
        "interpretation": "Comparisons are common-node differences, not rigorous total-error bounds; residual-based algebraic bounds exclude floating-point enclosure. Endpoint disks have radius 0.2d. Each geometry is separately solved.",
        "criteria": {"mesh_and_domain_difference": DOMAIN_MESH_TARGET,
                     "combined_algebraic_bound": ITERATION_TARGET,
                     "two_successive_coarse_domain_expansions": True,
                     "final_fine_grid_domain_expansion": True},
        "mesh_h_over_d": MESH_SIZES, "domain_candidates_over_d": DOMAIN_SIZES,
        "legacy_checks": legacy, "independent_zero_start_jacobi_crosscheck": crosscheck,
        "geometries": summaries, "output_files": ["metrics.csv", "midline-comparison.png"],
        "passed": bool(passed),
    }
    write_json(output / "report.json", report)
    print(f"M2 final checks: passed={passed}; report={output / 'report.json'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"M2 refinement failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
