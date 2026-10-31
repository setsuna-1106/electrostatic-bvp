"""Fixed-potential Jacobi solver and case-specific result helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from electrostatic_bvp.cases import build_parallel_plate, build_rectangle_sine
from electrostatic_bvp.config import ConfigError, validate_config


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class JacobiResult:
    """Potential and diagnostics for the unconstrained nodes of a grid."""

    potential: FloatArray
    converged: bool
    iterations: int
    final_change: float
    max_abs_residual: float


@dataclass(frozen=True)
class ParallelPlateResult:
    """Potential and fixed constraints; electric-field output is a later stage."""

    x: FloatArray
    y: FloatArray
    potential: FloatArray
    fixed_mask: BoolArray
    positive_plate_mask: BoolArray
    negative_plate_mask: BoolArray
    converged: bool
    iterations: int
    final_change: float
    max_abs_residual: float


@dataclass(frozen=True)
class RectangleResult:
    """Numerical solution and comparison data on a (y, x) grid."""

    x: FloatArray
    y: FloatArray
    potential: FloatArray
    analytic_potential: FloatArray
    electric_field_x: FloatArray
    electric_field_y: FloatArray
    converged: bool
    iterations: int
    final_change: float
    max_abs_residual: float
    max_abs_error: float
    rmse: float


def solve_jacobi(
    fixed_mask: BoolArray,
    fixed_values: FloatArray,
    *,
    dx: float,
    dy: float,
    tol: float,
    nmax: int,
) -> JacobiResult:
    """Solve a five-point Laplace problem with fixed outer and internal nodes.

    Both stopping change and residual are measured only at free nodes. Fixed
    nodes are imposed before iteration and copied unchanged on every step.
    """
    if (fixed_mask.ndim != 2 or fixed_values.shape != fixed_mask.shape
            or fixed_mask.dtype != np.bool_ or min(fixed_mask.shape) < 3):
        raise ValueError("fixed_mask must be a 2-D boolean array matching fixed_values, at least 3x3")
    if not np.isfinite(fixed_values).all():
        raise ValueError("fixed_values must be finite")
    if not (fixed_mask[0, :].all() and fixed_mask[-1, :].all()
            and fixed_mask[:, 0].all() and fixed_mask[:, -1].all()):
        raise ValueError("all outer boundary nodes must be fixed")
    free = ~fixed_mask[1:-1, 1:-1]
    if not free.any():
        raise ValueError("at least one free node is required")
    if not (np.isfinite(dx) and np.isfinite(dy) and np.isfinite(tol)
            and dx > 0 and dy > 0 and tol > 0):
        raise ValueError("dx, dy, and tol must be finite and positive")
    if isinstance(nmax, bool) or not isinstance(nmax, int) or nmax <= 0:
        raise ValueError("nmax must be a positive integer")

    potential = np.zeros(fixed_mask.shape, dtype=np.float64)
    potential[fixed_mask] = fixed_values[fixed_mask]
    weight_x = 1.0 / (2.0 * (1.0 + (dx / dy) ** 2))
    weight_y = 0.5 - weight_x
    converged = False
    for iteration in range(1, nmax + 1):
        # Every candidate is computed from the old potential, including
        # neighbours on both outer and internal fixed nodes.
        candidate = (
            weight_x * potential[1:-1, :-2]
            + weight_x * potential[1:-1, 2:]
            + weight_y * potential[:-2, 1:-1]
            + weight_y * potential[2:, 1:-1]
        )
        updated = potential.copy()
        updated[1:-1, 1:-1][free] = candidate[free]
        final_change = float(np.max(np.abs(candidate[free] - potential[1:-1, 1:-1][free])))
        potential = updated
        if final_change <= tol:
            converged = True
            break

    residual = (
        (potential[1:-1, :-2] - 2.0 * potential[1:-1, 1:-1]
         + potential[1:-1, 2:]) / dx**2
        + (potential[:-2, 1:-1] - 2.0 * potential[1:-1, 1:-1]
           + potential[2:, 1:-1]) / dy**2
    )
    return JacobiResult(
        potential=potential,
        converged=converged,
        iterations=iteration,
        final_change=final_change,
        max_abs_residual=float(np.max(np.abs(residual[free]))),
    )


def solve_parallel_plate(config: dict[str, Any]) -> ParallelPlateResult:
    """Solve the parallel-plate potential with fixed frame and plates."""
    config = validate_config(config)
    if config["case"] != "parallel_plate":
        raise ConfigError("expected case 'parallel_plate'")
    grid = build_parallel_plate(config)
    h = float(config["grid"]["h"])
    result = solve_jacobi(
        grid.fixed_mask, grid.fixed_values, dx=h, dy=h,
        tol=float(config["solver"]["tol"]), nmax=config["solver"]["Nmax"],
    )
    assert grid.positive_plate_mask is not None and grid.negative_plate_mask is not None
    return ParallelPlateResult(
        x=grid.x, y=grid.y, potential=result.potential, fixed_mask=grid.fixed_mask,
        positive_plate_mask=grid.positive_plate_mask, negative_plate_mask=grid.negative_plate_mask,
        converged=result.converged, iterations=result.iterations,
        final_change=result.final_change, max_abs_residual=result.max_abs_residual,
    )


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
    x_factor = np.sin(np.pi * (x / lx))
    # Equivalent to sinh(pi*y/lx) / sinh(pi*ly/lx), without overflowing
    # for tall rectangles. expm1 also avoids cancellation for small arguments.
    y_factor = (
        np.exp(np.pi * ((y - ly) / lx))
        * np.expm1(-2.0 * np.pi * (y / lx))
        / np.expm1(-2.0 * np.pi * (ly / lx))
    )
    return v0 * np.outer(y_factor, x_factor)


def electric_field(
    potential: FloatArray, *, dx: float, dy: float
) -> tuple[FloatArray, FloatArray]:
    """Return (Ex, Ey) = -grad(phi) on a (y, x) grid.

    Use centered second-order differences inside and second-order one-sided
    differences at the edges. Each axis must contain at least three nodes.
    """
    derivative_y, derivative_x = np.gradient(potential, dy, dx, edge_order=2)
    return -derivative_x, -derivative_y


def solve_rectangle_sine(config: dict[str, Any]) -> RectangleResult:
    """Solve the validated rectangle_sine case with five-point Jacobi updates.

    ``final_change`` is the maximum absolute change to any interior node in
    the last iteration. Error metrics are calculated on interior grid nodes.
    """
    config = validate_config(config)
    if config["case"] != "rectangle_sine":
        raise ConfigError("expected case 'rectangle_sine'")
    lx = float(config["geometry"]["Lx"])
    ly = float(config["geometry"]["Ly"])
    v0 = float(config["boundary_conditions"]["V0"])
    tol = float(config["solver"]["tol"])
    nmax = int(config["solver"]["Nmax"])

    grid = build_rectangle_sine(config)
    x, y = grid.x, grid.y
    dx = lx / (len(x) - 1)
    dy = ly / (len(y) - 1)
    solved = solve_jacobi(grid.fixed_mask, grid.fixed_values, dx=dx, dy=dy,
                          tol=tol, nmax=nmax)
    potential = solved.potential

    analytic = analytic_rectangle_sine(x, y, lx=lx, ly=ly, v0=v0)
    # Enforce the mathematically exact zero values at the two side edges.
    analytic[:, 0] = 0.0
    analytic[:, -1] = 0.0

    field_x, field_y = electric_field(potential, dx=dx, dy=dy)
    interior_error = potential[1:-1, 1:-1] - analytic[1:-1, 1:-1]
    max_abs_error = float(np.max(np.abs(interior_error)))
    rmse = float(np.sqrt(np.mean(interior_error**2)))

    return RectangleResult(
        x=x,
        y=y,
        potential=potential,
        analytic_potential=analytic,
        electric_field_x=field_x,
        electric_field_y=field_y,
        converged=solved.converged,
        iterations=solved.iterations,
        final_change=solved.final_change,
        max_abs_residual=solved.max_abs_residual,
        max_abs_error=max_abs_error,
        rmse=rmse,
    )

