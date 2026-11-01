"""Run a configured case and save its numerical result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from electrostatic_bvp.config import ConfigError, load_config
from electrostatic_bvp.fields import parallel_plate_metrics
from electrostatic_bvp.solver import (
    ParallelPlateResult, RectangleResult, solve_parallel_plate, solve_rectangle_sine,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a configured case and save numerical results."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rectangle.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, help="defaults to results/rectangle or results/parallel_plate"
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


def save_parallel_plate_results(
    config: dict, result: ParallelPlateResult, output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save plate potential, masked fields, figures, and descriptive metrics."""
    from electrostatic_bvp.visualize import save_parallel_plate_figures

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    solution_path = destination / "solution.npz"
    summary_path = destination / "summary.json"
    np.savez_compressed(
        solution_path, x=result.x, y=result.y, potential=result.potential,
        fixed_mask=result.fixed_mask, positive_plate_mask=result.positive_plate_mask,
        negative_plate_mask=result.negative_plate_mask,
        electric_field_x=result.field.electric_field_x,
        electric_field_y=result.field.electric_field_y,
        field_valid_mask=result.field.field_valid_mask,
        analysis_valid_mask=result.field.analysis_valid_mask,
    )
    positive_y, positive_x = np.where(result.positive_plate_mask)
    negative_y, negative_x = np.where(result.negative_plate_mask)

    def endpoints(ys: np.ndarray, xs: np.ndarray) -> list[list[float]]:
        return [[float(result.x[xs[0]]), float(result.y[ys[0]])],
                [float(result.x[xs[-1]]), float(result.y[ys[-1]])]]

    metrics = parallel_plate_metrics(
        result.x, result.y, result.field, result.positive_plate_mask,
        result.negative_plate_mask, v0=float(config["boundary_conditions"]["V0"]),
        h=float(config["grid"]["h"]),
    )
    figures = save_parallel_plate_figures(result, config, destination)
    summary = {
        "case": config["case"],
        "model": "2D Laplace equation; zero-thickness fixed-potential plates; zero-potential outer frame",
        "units": "Use consistent length and voltage units; potential uses the input voltage unit.",
        "config": config,
        "actual_geometry": {
            "x_bounds": [float(result.x[0]), float(result.x[-1])],
            "y_bounds": [float(result.y[0]), float(result.y[-1])],
            "positive_plate_endpoints_xy": endpoints(positive_y, positive_x),
            "negative_plate_endpoints_xy": endpoints(negative_y, negative_x),
            "positive_plate_potential": float(config["boundary_conditions"]["V0"]) / 2,
            "negative_plate_potential": -float(config["boundary_conditions"]["V0"]) / 2,
            "outer_potential": float(config["boundary_conditions"]["outer_potential"]),
        },
        "grid": {
            "nx_intervals": len(result.x) - 1,
            "ny_intervals": len(result.y) - 1,
            "array_shape_yx": list(result.potential.shape),
            "dx": float(config["grid"]["h"]),
            "dy": float(config["grid"]["h"]),
            "fixed_node_count": int(np.count_nonzero(result.fixed_mask)),
            "free_node_count": int(np.count_nonzero(~result.fixed_mask)),
        },
        "solver": {
            "method": config["solver"]["method"],
            "converged": result.converged,
            "stop_reason": "tolerance_reached" if result.converged else "maximum_iterations",
            "iterations": result.iterations,
            "final_change": result.final_change,
            "max_abs_laplace_residual": result.max_abs_residual,
            "diagnostic_region": "free_grid_nodes_only",
            "tol": config["solver"]["tol"],
            "Nmax": config["solver"]["Nmax"],
        },
        "electric_field": {
            "available": True,
            "definition": "E = -grad(potential); centered inside, second-order one-sided at outer edges",
            "validity": "plate nodes are invalid (NaN); field_valid_mask identifies bulk values",
            "analysis_mask": "analysis_valid_mask also excludes fixed physical-radius disks around plate endpoints",
            "stencil": "free-node stencils may end on but do not cross a plate",
            "figure_arrows": "length normalized and shortened near conductors; color shows field magnitude",
            "valid_node_count": int(np.count_nonzero(result.field.field_valid_mask)),
            "analysis_valid_node_count": int(np.count_nonzero(result.field.analysis_valid_mask)),
        },
        "field_comparison": metrics,
        "interpretation": "Descriptive comparisons to an ideal uniform field; not analytic errors or validation of open-space accuracy.",
        "outputs": {"solution_npz": solution_path.name, **figures},
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return solution_path, summary_path


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        if config["case"] == "rectangle_sine":
            result = solve_rectangle_sine(config)
            output = args.output_dir or Path("results/rectangle")
            solution_path, summary_path = save_results(config, result, output)
            diagnostic = f"max_abs_error={result.max_abs_error:.6g}"
        else:
            result = solve_parallel_plate(config)
            output = args.output_dir or Path("results/parallel_plate")
            solution_path, summary_path = save_parallel_plate_results(config, result, output)
            diagnostic = f"max_abs_laplace_residual={result.max_abs_residual:.6g}"
    except (ConfigError, OSError, ValueError) as exc:
        print(f"计算失败：{exc}", file=sys.stderr)
        return 2

    print(
        f"完成：converged={result.converged}, iterations={result.iterations}, "
        + diagnostic
    )
    print(f"数值结果：{solution_path}")
    print(f"运行摘要：{summary_path}")
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
