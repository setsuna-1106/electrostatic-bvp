"""Reproduce M1 numerical acceptance; keep all artifacts under results/."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from electrostatic_bvp.config import load_config
from electrostatic_bvp.main import save_results
from electrostatic_bvp.solver import solve_rectangle_sine


ROOT = Path(__file__).resolve().parents[1]


def inspect_result(config, result):
    """Independently compare against continuum fields and a discrete mode."""
    lx, ly = config["geometry"]["Lx"], config["geometry"]["Ly"]
    v0, h = config["boundary_conditions"]["V0"], config["grid"]["h"]
    x, y = np.meshgrid(result.x, result.y)
    k = np.pi / lx
    exact_ex = -v0 * k * np.cos(k * x) * np.sinh(k * y) / np.sinh(k * ly)
    exact_ey = -v0 * k * np.sin(k * x) * np.cosh(k * y) / np.sinh(k * ly)

    nx, ny = len(result.x) - 1, len(result.y) - 1
    alpha = np.arccosh(2.0 - np.cos(np.pi / nx))
    j = np.arange(ny + 1)
    y_mode = np.exp(alpha * (j - ny)) * np.expm1(-2 * alpha * j) / np.expm1(-2 * alpha * ny)
    discrete_exact = v0 * np.outer(y_mode, np.sin(np.pi * np.arange(nx + 1) / nx))
    u = result.potential
    residual = (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1]
                + u[2:, 1:-1] - 4 * u[1:-1, 1:-1]) / h**2
    boundary_error = max(
        np.max(np.abs(u[0])), np.max(np.abs(u[:, 0])), np.max(np.abs(u[:, -1])),
        np.max(np.abs(u[-1] - v0 * np.sin(k * result.x))),
    )
    return {
        "config": config,
        "converged": result.converged,
        "iterations": result.iterations,
        "final_change": result.final_change,
        "boundary_max_error": float(boundary_error),
        "max_abs_laplace_residual": float(np.max(np.abs(residual))),
        "max_abs_error": result.max_abs_error,
        "rmse": result.rmse,
        "max_error_to_discrete_solution": float(np.max(np.abs(u - discrete_exact))),
        "electric_field_max_vector_error_all_nodes": float(np.max(np.hypot(
            result.electric_field_x - exact_ex, result.electric_field_y - exact_ey))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/m1-validation")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = load_config(ROOT / "configs/rectangle.json")
    checks = {}

    def run(label, config):
        result = solve_rectangle_sine(config)
        save_results(config, result, args.output_dir / label)
        metrics = inspect_result(config, result)
        checks[f"{label}: fixed boundaries"] = metrics["boundary_max_error"] < 1e-14
        checks[f"{label}: finite output"] = all(np.isfinite(a).all() for a in (
            result.potential, result.analytic_potential, result.electric_field_x,
            result.electric_field_y))
        if result.converged:
            tol = config["solver"]["tol"]
            checks[f"{label}: stopping threshold"] = result.final_change <= tol
            checks[f"{label}: Laplace residual"] = (
                metrics["max_abs_laplace_residual"] < 4 * tol / config["grid"]["h"]**2 + 1e-11)
        return result, metrics

    default, default_metrics = run("default", base)
    repeated = solve_rectangle_sine(base)
    checks["default: converged"] = default.converged
    checks["default: reproducible potential"] = np.array_equal(default.potential, repeated.potential)
    checks["default: reproducible field"] = np.array_equal(default.electric_field_y, repeated.electric_field_y)

    mesh = []
    for h in (0.1, 0.05, 0.025):
        config = deepcopy(base)
        config["grid"]["h"] = h
        config["solver"]["tol"] = 1e-11
        result, metrics = run(f"mesh-h-{h}", config)
        checks[f"mesh h={h}: converged"] = result.converged
        checks[f"mesh h={h}: iteration error separated"] = metrics["max_error_to_discrete_solution"] < 2e-8
        mesh.append(metrics)

    orders = {}
    for key in ("max_abs_error", "electric_field_max_vector_error_all_nodes"):
        orders[key] = [float(np.log2(coarse[key] / fine[key]))
                       for coarse, fine in zip(mesh, mesh[1:])]
        checks[f"{key}: second-order mesh convergence"] = all(1.8 < order < 2.2 for order in orders[key])

    tolerance = []
    for tol in (1e-6, 1e-8, 1e-10):
        config = deepcopy(base)
        config["solver"]["tol"] = tol
        result, metrics = run(f"tol-{tol}", config)
        checks[f"tol={tol}: converged"] = result.converged
        tolerance.append(metrics)
    checks["tightening tol reduces iteration error"] = all(
        fine["max_error_to_discrete_solution"] < coarse["max_error_to_discrete_solution"]
        for coarse, fine in zip(tolerance, tolerance[1:]))
    checks["tightening tol requires more iterations"] = all(
        fine["iterations"] > coarse["iterations"] for coarse, fine in zip(tolerance, tolerance[1:]))

    config = deepcopy(base)
    config["solver"]["Nmax"] = 1
    stopped, stopped_metrics = run("nmax-1", config)
    checks["Nmax=1 is reported as nonconverged"] = not stopped.converged and stopped.iterations == 1

    source_paths = ["configs/rectangle.json", "pyproject.toml", "uv.lock", "scripts/validate_m1.py",
                    "src/electrostatic_bvp/config.py", "src/electrostatic_bvp/solver.py",
                    "src/electrostatic_bvp/main.py"]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths},
        "default": default_metrics,
        "mesh_refinement": mesh,
        "observed_orders": orders,
        "tolerance_sensitivity": tolerance,
        "maximum_iteration_case": stopped_metrics,
        "checks": {name: bool(passed) for name, passed in checks.items()},
        "passed": all(checks.values()),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, passed in checks.items():
        if not passed:
            print(f"FAIL: {name}")
    print(f"M1 numerical acceptance: {sum(checks.values())}/{len(checks)} checks passed")
    print(f"Report: {report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
