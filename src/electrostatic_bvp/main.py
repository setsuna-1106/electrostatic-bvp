"""Run the rectangle_sine case and save its numerical result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from electrostatic_bvp.config import ConfigError, load_config
from electrostatic_bvp.solver import RectangleResult, solve_rectangle_sine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the rectangle_sine case and save numerical results."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rectangle.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/rectangle")
    )
    return parser.parse_args()


def save_results(
    config: dict,
    result: RectangleResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write arrays to NPZ and run metadata and metrics to JSON."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    solution_path = destination / "solution.npz"
    summary_path = destination / "summary.json"

    absolute_error = np.abs(result.potential - result.analytic_potential)
    np.savez_compressed(
        solution_path,
        x=result.x,
        y=result.y,
        potential=result.potential,
        analytic_potential=result.analytic_potential,
        absolute_error=absolute_error,
        electric_field_x=result.electric_field_x,
        electric_field_y=result.electric_field_y,
    )

    nx = len(result.x) - 1
    ny = len(result.y) - 1
    summary = {
        "case": config["case"],
        "config": config,
        "grid": {
            "nx_intervals": nx,
            "ny_intervals": ny,
            "array_shape_yx": list(result.potential.shape),
            "dx": float(result.x[1] - result.x[0]),
            "dy": float(result.y[1] - result.y[0]),
        },
        "solver": {
            "method": config["solver"]["method"],
            "converged": result.converged,
            "stop_reason": "tolerance_reached" if result.converged else "maximum_iterations",
            "iterations": result.iterations,
            "final_change": result.final_change,
            "max_abs_laplace_residual": result.max_abs_residual,
            "tol": config["solver"]["tol"],
            "Nmax": config["solver"]["Nmax"],
        },
        "analytic_comparison": {
            "error_region": "interior_grid_nodes",
            "max_abs_error": result.max_abs_error,
            "rmse": result.rmse,
        },
        "electric_field": {
            "definition": "E = -grad(potential)",
            "array_axes": ["y", "x"],
            "interior_difference_order": 2,
            "boundary_difference_order": 2,
        },
        "outputs": {"solution_npz": solution_path.name},
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")

    return solution_path, summary_path


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        result = solve_rectangle_sine(config)
        solution_path, summary_path = save_results(config, result, args.output_dir)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"计算失败：{exc}", file=sys.stderr)
        return 2

    print(
        f"完成：converged={result.converged}, iterations={result.iterations}, "
        f"max_abs_error={result.max_abs_error:.6g}"
    )
    print(f"数值结果：{solution_path}")
    print(f"运行摘要：{summary_path}")
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
