"""Run and record the M2 Jacobi iteration, domain, and mesh checks.

This script keeps each run separately and can resume without recomputing completed
runs. It compares only common nodes in the fixed physical target region, never
interpolates, and never treats an unconverged run as an accepted comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from electrostatic_bvp.config import load_config
from electrostatic_bvp.solver import solve_parallel_plate


ROOT = Path(__file__).resolve().parents[1]
TARGET = (-3.0, 3.0, -1.5, 1.5)
ITERATION_TARGET = 0.001
DOMAIN_MESH_TARGET = 0.01
SOURCES = (
    "configs/parallel_plate.json", "src/electrostatic_bvp/config.py",
    "src/electrostatic_bvp/cases.py", "src/electrostatic_bvp/fields.py",
    "src/electrostatic_bvp/solver.py", "src/electrostatic_bvp/acceleration.py",
    "pyproject.toml", "uv.lock",
)


def source_hashes(paths=SOURCES):
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass
class CaseData:
    config: dict
    arrays: dict[str, np.ndarray]
    metadata: dict


def save_case(directory: Path, config: dict, result, elapsed: float, provenance: dict):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        "x": result.x, "y": result.y, "potential": result.potential,
        "electric_field_x": result.field.electric_field_x,
        "electric_field_y": result.field.electric_field_y,
        "fixed_mask": result.fixed_mask,
        "positive_plate_mask": result.positive_plate_mask,
        "negative_plate_mask": result.negative_plate_mask,
        "field_valid_mask": result.field.field_valid_mask,
        "analysis_valid_mask": result.field.analysis_valid_mask,
    }
    array_path = directory / "solution.npz"
    temporary = directory / "solution.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(array_path)
    meta = {
        "config": config,
        "grid_shape_yx": list(result.potential.shape),
        "converged": result.converged,
        "stop_reason": "tolerance_reached" if result.converged else "maximum_iterations",
        "iterations": result.iterations,
        "final_change": result.final_change,
        "max_abs_laplace_residual_free_nodes": result.max_abs_residual,
        "solve_wall_seconds": elapsed,
        "endpoint_exclusion_radius": result.field.endpoint_exclusion_radius,
        "endpoint_exclusion_has_two_intervals": result.field.endpoint_exclusion_has_two_intervals,
        "solution_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
        "provenance": provenance,
    }
    write_json(directory / "run.json", meta)
    return CaseData(config, arrays, meta)


def load_case(directory: Path, config: dict, provenance: dict):
    meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    array_path = directory / "solution.npz"
    if meta["config"] != config or meta["provenance"] != provenance:
        raise ValueError(f"Cannot resume {directory}: config, source or numerical environment changed")
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != meta["solution_sha256"]:
        raise ValueError(f"Cannot resume {directory}: solution hash differs")
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return CaseData(config, arrays, meta)


def solve_record(label, config, output, provenance, *, resume, backend="numpy"):
    directory = output / "runs" / label
    if directory.exists():
        if not resume:
            raise FileExistsError(f"{directory} exists; use --resume or choose another output directory")
        if (directory / "run.json").is_file() and (directory / "solution.npz").is_file():
            case = load_case(directory, config, provenance)
            print(f"REUSED {label}: iterations={case.metadata['iterations']}", flush=True)
            return case
        raise ValueError(f"Incomplete run directory {directory}; inspect or move it before resuming")
    print(f"START {label}: h={config['grid']['h']}, tol={config['solver']['tol']}, "
          f"a={config['geometry']['domain_half_width']}, b={config['geometry']['domain_half_height']}, "
          f"Nmax={config['solver']['Nmax']}", flush=True)
    started = time.perf_counter()
    result = solve_parallel_plate(config, backend=backend)
    elapsed = time.perf_counter() - started
    case = save_case(directory, config, result, elapsed, provenance)
    print(f"DONE {label}: converged={result.converged}, iterations={result.iterations}, "
          f"residual={result.max_abs_residual:.6g}, seconds={elapsed:.2f}", flush=True)
    return case


def algebraic_bounds(case: CaseData):
    """Bound error of the discrete equation using a quadratic barrier.

    For residual R and zero error at fixed nodes, the discrete maximum principle
    with (b²-y²)/2 gives |error| <= R*b²/2. The x barrier can be used too; the
    smaller is selected. At target-region interior nodes, a central derivative
    has component error <= max|error|/h. This is a mathematical bound based on
    the reported floating-point residual, not a rigorous roundoff enclosure.
    """
    geom = case.config["geometry"]
    h = float(case.config["grid"]["h"])
    v0 = abs(float(case.config["boundary_conditions"]["V0"]))
    gap = float(geom["gap"])
    if v0 == 0:
        raise ValueError("S4 normalized comparisons require nonzero V0")
    r = float(case.metadata["max_abs_laplace_residual_free_nodes"])
    error_bound = r * min(float(geom["domain_half_width"]), float(geom["domain_half_height"])) ** 2 / 2
    return {
        "potential_over_abs_v0": error_bound / v0,
        "field_vector_over_eref": np.sqrt(2) * error_bound / h / (v0 / gap),
        "note": "discrete maximum-principle bound from reported residual; excludes floating-point enclosure",
    }


def compare_cases(first: CaseData, second: CaseData, target=TARGET):
    """Compare on the same physical nodes and the same fixed-radius mask."""
    a, b = first, second
    ga, gb = a.config["geometry"], b.config["geometry"]
    for key in ("plate_length", "gap"):
        if not np.isclose(ga[key], gb[key], rtol=0, atol=1e-12):
            raise ValueError(f"Cannot compare different {key}")
    v0 = float(a.config["boundary_conditions"]["V0"])
    if v0 == 0 or v0 != float(b.config["boundary_conditions"]["V0"]):
        raise ValueError("Comparison requires equal, nonzero voltages")
    if float(a.config["grid"]["h"]) < float(b.config["grid"]["h"]):
        a, b = b, a
    h_coarse = float(a.config["grid"]["h"])
    h_fine = float(b.config["grid"]["h"])
    ratio = h_coarse / h_fine
    if not np.isclose(ratio, round(ratio), atol=1e-9, rtol=0):
        raise ValueError("Mesh spacings must be nested integer ratios")
    xa, ya = a.arrays["x"], a.arrays["y"]
    xb, yb = b.arrays["x"], b.arrays["y"]
    xmin, xmax, ymin, ymax = target
    epsilon = h_coarse * 1e-8
    for x, y in ((xa, ya), (xb, yb)):
        if xmin < x[0] - epsilon or xmax > x[-1] + epsilon or ymin < y[0] - epsilon or ymax > y[-1] + epsilon:
            raise ValueError("The full target region must lie inside both domains")
        if xmin <= x[0] + epsilon or xmax >= x[-1] - epsilon or ymin <= y[0] + epsilon or ymax >= y[-1] - epsilon:
            raise ValueError("The target region must be strictly inside both domains for the centered-field bound")
    ix = np.flatnonzero((xa >= xmin - epsilon) & (xa <= xmax + epsilon))
    iy = np.flatnonzero((ya >= ymin - epsilon) & (ya <= ymax + epsilon))
    jx = np.rint((xa[ix] - xb[0]) / h_fine).astype(int)
    jy = np.rint((ya[iy] - yb[0]) / h_fine).astype(int)
    if not (np.allclose(xa[ix], xb[jx], atol=epsilon, rtol=0)
            and np.allclose(ya[iy], yb[jy], atol=epsilon, rtol=0)):
        raise ValueError("Comparison coordinates do not match common nodes")
    idx_a, idx_b = np.ix_(iy, ix), np.ix_(jy, jx)
    mask_a = a.arrays["analysis_valid_mask"][idx_a]
    mask_b = b.arrays["analysis_valid_mask"][idx_b]
    if not np.array_equal(mask_a, mask_b) or not mask_a.any():
        raise ValueError("Common-node analysis masks must match and contain valid nodes")
    du = a.arrays["potential"][idx_a] - b.arrays["potential"][idx_b]
    dex = a.arrays["electric_field_x"][idx_a] - b.arrays["electric_field_x"][idx_b]
    dey = a.arrays["electric_field_y"][idx_a] - b.arrays["electric_field_y"][idx_b]
    delta_u = float(np.max(np.abs(du[mask_a])) / abs(v0))
    delta_e = float(np.max(np.hypot(dex[mask_a], dey[mask_a])) / (abs(v0) / float(ga["gap"])))
    bound_a, bound_b = algebraic_bounds(a), algebraic_bounds(b)
    return {
        "target_region_xy": list(target), "common_valid_nodes": int(np.count_nonzero(mask_a)),
        "common_grid_spacing": h_coarse, "interpolation": False,
        "endpoint_exclusion_radius": a.metadata["endpoint_exclusion_radius"],
        "both_endpoint_radii_have_two_intervals": bool(
            a.metadata["endpoint_exclusion_has_two_intervals"] and b.metadata["endpoint_exclusion_has_two_intervals"]),
        "both_converged": bool(a.metadata["converged"] and b.metadata["converged"]),
        "max_potential_difference_over_abs_v0": delta_u,
        "max_field_vector_difference_over_eref": delta_e,
        "combined_algebraic_error_bound_potential": float(bound_a["potential_over_abs_v0"] + bound_b["potential_over_abs_v0"]),
        "combined_algebraic_error_bound_field": float(bound_a["field_vector_over_eref"] + bound_b["field_vector_over_eref"]),
    }


def meets_tolerance(comparison, limit, *, require_iteration_bound):
    passed = (comparison["both_converged"]
              and comparison["both_endpoint_radii_have_two_intervals"]
              and comparison["max_potential_difference_over_abs_v0"] < limit
              and comparison["max_field_vector_difference_over_eref"] < limit)
    if require_iteration_bound:
        passed = (passed and comparison["combined_algebraic_error_bound_potential"] < ITERATION_TARGET
                  and comparison["combined_algebraic_error_bound_field"] < ITERATION_TARGET)
    return bool(passed)


def structural_checks(reference: CaseData, negative: CaseData, doubled: CaseData):
    u = reference.arrays["potential"]
    pos, neg = reference.arrays["positive_plate_mask"], reference.arrays["negative_plate_mask"]
    v0 = float(reference.config["boundary_conditions"]["V0"])
    if v0 <= 0:
        raise ValueError("Structural checks require positive V0")
    frame = reference.arrays["fixed_mask"] & ~(pos | neg)
    scale = abs(v0)
    x_error = float(np.max(np.abs(u - u[:, ::-1])) / scale)
    y_error = float(np.max(np.abs(u + u[::-1, :])) / scale)
    negative_error = float(np.max(np.abs(negative.arrays["potential"] + u)) / scale)
    doubled_error = float(np.max(np.abs(doubled.arrays["potential"] - 2 * u)) / scale)
    valid = reference.arrays["field_valid_mask"]
    eref = scale / float(reference.config["geometry"]["gap"])
    ex, ey = reference.arrays["electric_field_x"], reference.arrays["electric_field_y"]
    negative_field_error = float(np.max(np.hypot(negative.arrays["electric_field_x"][valid] + ex[valid],
                                                  negative.arrays["electric_field_y"][valid] + ey[valid])) / eref)
    doubled_field_error = float(np.max(np.hypot(doubled.arrays["electric_field_x"][valid] - 2 * ex[valid],
                                                 doubled.arrays["electric_field_y"][valid] - 2 * ey[valid])) / eref)
    y0, x0 = len(reference.arrays["y"]) // 2, len(reference.arrays["x"]) // 2
    center_ey = float(reference.arrays["electric_field_y"][y0, x0])
    fixed_error = max(float(np.max(np.abs(u[pos] - v0 / 2))),
                      float(np.max(np.abs(u[neg] + v0 / 2))),
                      float(np.max(np.abs(u[frame]))))
    min_potential, max_potential = float(np.min(u)), float(np.max(u))
    criteria = {
        "all_three_converged": all(c.metadata["converged"] for c in (reference, negative, doubled)),
        "fixed_nodes_exact": fixed_error == 0,
        "even_in_x": x_error < 1e-12,
        "odd_in_y": y_error < 1e-12,
        "maximum_principle": min_potential >= -v0 / 2 - 1e-12 and max_potential <= v0 / 2 + 1e-12,
        "central_field_down": center_ey < 0,
        "negative_voltage_linearity": negative_error < 1e-8,
        "double_voltage_linearity": doubled_error < 1e-8,
        "negative_field_linearity": negative_field_error < 1e-8,
        "double_field_linearity": doubled_field_error < 1e-8,
    }
    return {"checks": {k: bool(v) for k, v in criteria.items()}, "passed": all(criteria.values()),
            "max_fixed_value_error": fixed_error, "x_symmetry_error_over_v0": x_error,
            "y_antisymmetry_error_over_v0": y_error, "negative_linearity_error_over_v0": negative_error,
            "double_linearity_error_over_v0": doubled_error, "center_electric_field_y": center_ey,
            "negative_field_linearity_error_over_eref": negative_field_error,
            "double_field_linearity_error_over_eref": doubled_field_error,
            "potential_min": min_potential, "potential_max": max_potential,
            "symmetry_tolerance": 1e-12, "linearity_tolerance": 1e-8}


def make_config(base, *, h=0.1, a=6.0, b=4.0, tol=1e-11, nmax=2_000_000, voltage=1.0):
    config = deepcopy(base)
    config["grid"]["h"] = h
    config["geometry"]["domain_half_width"] = a
    config["geometry"]["domain_half_height"] = b
    config["solver"]["tol"] = tol
    config["solver"]["Nmax"] = nmax
    config["boundary_conditions"]["V0"] = voltage
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/m2-validation")
    parser.add_argument("--nmax", type=int, default=2_000_000,
                        help="upper iteration cap for each run (tolerance is not relaxed)")
    parser.add_argument("--resume", action="store_true", help="reuse complete runs with matching config and source hashes")
    parser.add_argument("--backend", choices=("numpy", "numba"), default="numpy",
                        help="Numba compiles the same Jacobi steps; requires the s4 extra")
    args = parser.parse_args()
    if args.nmax <= 0:
        parser.error("--nmax must be a positive integer")
    output = args.output_dir
    if output.exists() and not args.resume:
        parser.error(f"{output} already exists; choose another directory or use --resume")
    output.mkdir(parents=True, exist_ok=True)
    base = load_config(ROOT / "configs/parallel_plate.json")
    provenance = {"python": platform.python_version(), "numpy": np.__version__,
                  "machine": platform.machine(), "backend": args.backend, "source_sha256": source_hashes()}
    if args.backend == "numba":
        try:
            import numba
        except ImportError as exc:
            raise ValueError("Numba backend requires uv sync --extra s4") from exc
        provenance["numba"] = numba.__version__
    cases = {}

    def run(label, **kwargs):
        config = make_config(base, nmax=args.nmax, **kwargs)
        cases[label] = solve_record(label, config, output, provenance, resume=args.resume, backend=args.backend)
        return cases[label]

    reference_loose = run("tol-1e-7", tol=1e-7)
    reference_medium = run("tol-1e-9", tol=1e-9)
    reference = run("tol-1e-11")
    negative = run("voltage-negative", voltage=-1.0)
    doubled = run("voltage-double", voltage=2.0)
    tolerance_pairs = {
        "1e-7_vs_1e-9": compare_cases(reference_loose, reference_medium),
        "1e-9_vs_1e-11": compare_cases(reference_medium, reference),
    }
    structural = structural_checks(reference, negative, doubled)
    # Preserve the planned (6,4), (9,6), (12,8) checks, and extend outward
    # if necessary. The mesh domain is the first in two consecutive passing
    # expansion comparisons; its immediate successor is used on the fine grid.
    domain_sizes = [(6.0, 4.0), (9.0, 6.0), (12.0, 8.0), (18.0, 12.0), (24.0, 16.0)]
    domain_cases = [reference]
    domain_pairs = {}
    consecutive_passes = 0
    selected_index = None
    for index, (outer_a, outer_b) in enumerate(domain_sizes[1:], 1):
        label = f"domain-a{int(outer_a)}-b{int(outer_b)}"
        domain_cases.append(run(label, a=outer_a, b=outer_b))
        previous_a, previous_b = domain_sizes[index - 1]
        name = f"a{int(previous_a)}-b{int(previous_b)}_vs_a{int(outer_a)}-b{int(outer_b)}"
        comparison = compare_cases(domain_cases[index - 1], domain_cases[index])
        domain_pairs[name] = comparison
        passed = meets_tolerance(comparison, DOMAIN_MESH_TARGET, require_iteration_bound=True)
        consecutive_passes = consecutive_passes + 1 if passed else 0
        print(f"DOMAIN {name}: potential={comparison['max_potential_difference_over_abs_v0']:.6g}, "
              f"field={comparison['max_field_vector_difference_over_eref']:.6g}, passed={passed}", flush=True)
        if index >= 2 and consecutive_passes >= 2:
            selected_index = index - 2
            break
    if selected_index is None:
        # Continue with a checked domain to report mesh behaviour, but retain
        # the failed domain status: the selected domain is not accepted.
        selected_index = 2
    a, b = domain_sizes[selected_index]
    a_big, b_big = domain_sizes[selected_index + 1]
    mesh_coarse = domain_cases[selected_index]
    mesh_medium = run("mesh-h005", h=0.05, a=a, b=b)
    mesh_fine = run("mesh-h0025", h=0.025, a=a, b=b)
    domain_fine = run("fine-domain-expanded", h=0.025, a=a_big, b=b_big)
    mesh_pairs = {
        "h01_vs_h005": compare_cases(mesh_coarse, mesh_medium),
        "h005_vs_h0025": compare_cases(mesh_medium, mesh_fine),
    }
    final_domain = compare_cases(mesh_fine, domain_fine)
    checks = {f"structure: {key}": bool(value) for key, value in structural["checks"].items()}
    checks.update({f"tolerance: {name}": meets_tolerance(pair, ITERATION_TARGET, require_iteration_bound=False)
                   for name, pair in tolerance_pairs.items()})
    checks["domain: two successive expansions"] = consecutive_passes >= 2
    checks.update({f"mesh: {name}": meets_tolerance(pair, DOMAIN_MESH_TARGET, require_iteration_bound=True)
                   for name, pair in mesh_pairs.items()})
    checks["fine-grid domain expansion"] = meets_tolerance(final_domain, DOMAIN_MESH_TARGET,
                                                            require_iteration_bound=True)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "2D zero-thickness fixed-potential plates; finite zero-potential outer frame",
        "environment": {**provenance, "platform": platform.platform()},
        "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "target_region_xy": list(TARGET),
        "normalization": {"potential": "abs(V0)", "field": "abs(V0)/gap"},
        "limits": {"tolerance_difference": ITERATION_TARGET, "domain_and_mesh_difference": DOMAIN_MESH_TARGET,
                   "combined_algebraic_error_bound_for_domain_and_mesh": ITERATION_TARGET},
        "limits_note": "Suggested comparison scales, not rigorous total-error bounds or proof of open-space accuracy.",
        "algebraic_bound_note": "Maximum principle for the discrete equation based on reported residual; excludes floating-point enclosure.",
        "runs": {name: {**case.metadata, "path": f"runs/{name}/", "algebraic_error_bounds": algebraic_bounds(case)}
                 for name, case in cases.items()},
        "structure": structural,
        "iteration_tolerance_comparisons": tolerance_pairs,
        "domain_comparisons": domain_pairs,
        "mesh_domain_selected": {"a": a, "b": b, "coarse_domain_accepted": consecutive_passes >= 2,
                                 "reason": "first domain of two passing consecutive expansions" if consecutive_passes >= 2 else "checked domain only; no two passing expansions"},
        "mesh_comparisons": mesh_pairs,
        "final_fine_grid_domain_comparison": {"from": [a, b], "to": [a_big, b_big], **final_domain},
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_json(output / "report.json", report)
    for key, value in checks.items():
        if not value:
            print(f"FAIL: {key}")
    print(f"M2 reliability: {sum(checks.values())}/{len(checks)} checks passed; report: {output / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"M2 validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
