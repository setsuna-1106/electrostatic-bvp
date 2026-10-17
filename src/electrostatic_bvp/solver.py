"""Jacobi solver and analytic comparison for the rectangle_sine MVP case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from electrostatic_bvp.config import validate_config


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RectangleResult:
    """Numerical solution and comparison data on a (y, x) grid."""

    x: FloatArray
    y: FloatArray
    potential: FloatArray
    analytic_potential: FloatArray
    converged: bool
    iterations: int
    final_change: float
    max_abs_error: float
    rmse: float


def analytic_rectangle_sine(
    x: FloatArray,
    y: FloatArray,
    *,
    lx: float,
    ly: float,
    v0: float,
) -> FloatArray:
    """Evaluate the exact solution for a sine-potential top boundary.

    The returned array has shape ``(len(y), len(x))``.
    """
    x_factor = np.sin(np.pi * x / lx)
    y_factor = np.sinh(np.pi * y / lx) / np.sinh(np.pi * ly / lx)
    return v0 * np.outer(y_factor, x_factor)


def solve_rectangle_sine(config: dict[str, Any]) -> RectangleResult:
    """Solve the validated rectangle_sine case with five-point Jacobi updates.

    ``final_change`` is the maximum absolute change to any interior node in
    the last iteration. Error metrics are calculated on interior grid nodes.
    """
    config = validate_config(config)
    lx = float(config["geometry"]["Lx"])
    ly = float(config["geometry"]["Ly"])
    v0 = float(config["boundary_conditions"]["V0"])
    h = float(config["grid"]["h"])
    tol = float(config["solver"]["tol"])
    nmax = int(config["solver"]["Nmax"])

    nx = round(lx / h)
    ny = round(ly / h)
    x = np.linspace(0.0, lx, nx + 1, dtype=np.float64)
    y = np.linspace(0.0, ly, ny + 1, dtype=np.float64)
    dx = lx / nx
    dy = ly / ny

    potential = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    potential[-1, :] = v0 * np.sin(np.pi * x / lx)
    # Set the side edges after the top edge so the shared corner values are exact.
    potential[:, 0] = 0.0
    potential[:, -1] = 0.0

    converged = False
    final_change = float("inf")
    for iteration in range(1, nmax + 1):
        updated = potential.copy()
        updated[1:-1, 1:-1] = (
            dy**2 * (potential[1:-1, :-2] + potential[1:-1, 2:])
            + dx**2 * (potential[:-2, 1:-1] + potential[2:, 1:-1])
        ) / (2.0 * (dx**2 + dy**2))

        final_change = float(
            np.max(np.abs(updated[1:-1, 1:-1] - potential[1:-1, 1:-1]))
        )
        potential = updated
        if final_change <= tol:
            converged = True
            break

    analytic = analytic_rectangle_sine(x, y, lx=lx, ly=ly, v0=v0)
    # Enforce the mathematically exact zero values at the two side edges.
    analytic[:, 0] = 0.0
    analytic[:, -1] = 0.0

    interior_error = potential[1:-1, 1:-1] - analytic[1:-1, 1:-1]
    max_abs_error = float(np.max(np.abs(interior_error)))
    rmse = float(np.sqrt(np.mean(interior_error**2)))

    return RectangleResult(
        x=x,
        y=y,
        potential=potential,
        analytic_potential=analytic,
        converged=converged,
        iterations=iteration,
        final_change=final_change,
        max_abs_error=max_abs_error,
        rmse=rmse,
    )

