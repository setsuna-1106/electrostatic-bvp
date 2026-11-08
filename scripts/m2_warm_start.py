"""Build an initial guess for M2, then check it with the original Jacobi step.

This is an optional *validation* helper, not a replacement for the project's
zero-initialized solver. Geometric multigrid approximately solves the same
five-point equations to prepare an initial potential. Every returned result is
then subjected to the original Numba Jacobi update and to an independently
computed residual. The caller must check both convergence and the residual-based
error bound before accepting a comparison.

Coarse masks approximate the internal lines and may merge at very coarse levels;
they affect only the initial guess. Padding consists entirely of fixed-zero
nodes and leaves the fine-grid physical domain and constraints unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from numba import njit

from electrostatic_bvp.acceleration import compiled_jacobi
from electrostatic_bvp.config import validate_config


@njit(cache=False)
def _smooth(u, rhs, fixed, sweeps):
    ny, nx = u.shape
    has_rhs = rhs.size != 1
    for _ in range(sweeps):
        for color in range(2):
            for row in range(1, ny - 1):
                for col in range(1 + ((row + color + 1) % 2), nx - 1, 2):
                    if not fixed[row, col]:
                        source = rhs[row, col] if has_rhs else 0.0
                        u[row, col] = 0.25 * (
                            source + u[row - 1, col] + u[row + 1, col]
                            + u[row, col - 1] + u[row, col + 1]
                        )


@njit(cache=False)
def _residual_array(u, rhs, fixed):
    """Return rhs - (4u - neighbours), zero on all fixed nodes."""
    ny, nx = u.shape
    residual = np.zeros_like(u)
    has_rhs = rhs.size != 1
    for row in range(1, ny - 1):
        for col in range(1, nx - 1):
            if not fixed[row, col]:
                source = rhs[row, col] if has_rhs else 0.0
                residual[row, col] = (
                    source + u[row - 1, col] + u[row + 1, col]
                    + u[row, col - 1] + u[row, col + 1] - 4.0 * u[row, col]
                )
    return residual


@njit(cache=False)
def _restrict(residual, coarse_fixed):
    """Full-weight the residual; factor four accounts for the doubled h."""
    ny, nx = coarse_fixed.shape
    result = np.zeros((ny, nx), dtype=np.float64)
    for row in range(1, ny - 1):
        for col in range(1, nx - 1):
            if not coarse_fixed[row, col]:
                i, j = 2 * row, 2 * col
                result[row, col] = 0.25 * (
                    4.0 * residual[i, j]
                    + 2.0 * (residual[i - 1, j] + residual[i + 1, j]
                             + residual[i, j - 1] + residual[i, j + 1])
                    + residual[i - 1, j - 1] + residual[i - 1, j + 1]
                    + residual[i + 1, j - 1] + residual[i + 1, j + 1]
                )
    return result


@njit(cache=False)
def _prolong_add(u, coarse_error, fixed):
    """Bilinearly add a coarse correction only at free fine nodes."""
    ny, nx = u.shape
    for row in range(1, ny - 1):
        for col in range(1, nx - 1):
            if not fixed[row, col]:
                i, j = row // 2, col // 2
                if row % 2 == 0 and col % 2 == 0:
                    value = coarse_error[i, j]
                elif row % 2 == 0:
                    value = 0.5 * (coarse_error[i, j] + coarse_error[i, j + 1])
                elif col % 2 == 0:
                    value = 0.5 * (coarse_error[i, j] + coarse_error[i + 1, j])
                else:
                    value = 0.25 * (coarse_error[i, j] + coarse_error[i + 1, j]
                                    + coarse_error[i, j + 1] + coarse_error[i + 1, j + 1])
                u[row, col] += value


@njit(cache=False)
def _maximum_jacobi_change(u, fixed):
    maximum = 0.0
    for row in range(1, u.shape[0] - 1):
        for col in range(1, u.shape[1] - 1):
            if not fixed[row, col]:
                candidate = (0.25 * u[row, col - 1] + 0.25 * u[row, col + 1]
                             + 0.25 * u[row - 1, col] + 0.25 * u[row + 1, col])
                maximum = max(maximum, abs(candidate - u[row, col]))
    return maximum


@njit(cache=False)
def _maximum_laplace_residual(u, fixed, h):
    """Use the same component-wise residual definition as solver.py."""
    maximum = 0.0
    h2 = h * h
    for row in range(1, u.shape[0] - 1):
        for col in range(1, u.shape[1] - 1):
            if not fixed[row, col]:
                value = ((u[row, col - 1] - 2.0 * u[row, col] + u[row, col + 1]) / h2
                         + (u[row - 1, col] - 2.0 * u[row, col] + u[row + 1, col]) / h2)
                maximum = max(maximum, abs(value))
    return maximum


def _cycle(level, u, rhs, masks):
    fixed = masks[level]
    if level == len(masks) - 1:
        _smooth(u, rhs, fixed, 60)
        return
    _smooth(u, rhs, fixed, 3)
    residual = _residual_array(u, rhs, fixed)
    coarse_rhs = _restrict(residual, masks[level + 1])
    del residual
    correction = np.zeros_like(coarse_rhs)
    _cycle(level + 1, correction, coarse_rhs, masks)
    _prolong_add(u, correction, fixed)
    _smooth(u, rhs, fixed, 3)


def _masks_and_initial(config):
    """Construct exact fine constraints; pad at right/top with fixed zeros."""
    geometry = config["geometry"]
    h = float(config["grid"]["h"])
    ax = round(float(geometry["domain_half_width"]) / h)
    by = round(float(geometry["domain_half_height"]) / h)
    half_length = round(float(geometry["plate_length"]) / (2.0 * h))
    half_gap = round(float(geometry["gap"]) / (2.0 * h))
    nx, ny = 2 * ax + 1, 2 * by + 1
    # Divisibility keeps the coarse correction inexpensive even for domains
    # such as (18,12). These added cells are fixed, never physical nodes.
    block = 64
    padded_nx = ((nx - 1 + block - 1) // block) * block + 1
    padded_ny = ((ny - 1 + block - 1) // block) * block + 1
    positions = ((by + half_gap, ax - half_length, ax + half_length),
                 (by - half_gap, ax - half_length, ax + half_length))
    fine = np.zeros((padded_ny, padded_nx), dtype=bool)
    fine[0, :] = True
    fine[:, 0] = True
    fine[ny - 1:, :] = True
    fine[:, nx - 1:] = True
    initial = np.zeros(fine.shape, dtype=np.float64)
    voltage = float(config["boundary_conditions"]["V0"])
    for (row, lo, hi), value in zip(positions, (voltage / 2, -voltage / 2)):
        fine[row, lo:hi + 1] = True
        initial[row, lo:hi + 1] = value
    masks = [fine]
    level_ny, level_nx, step = padded_ny, padded_nx, 1
    while (level_ny - 1) % 2 == 0 and (level_nx - 1) % 2 == 0 and min(level_ny, level_nx) > 20:
        level_ny = (level_ny - 1) // 2 + 1
        level_nx = (level_nx - 1) // 2 + 1
        step *= 2
        coarse = np.zeros((level_ny, level_nx), dtype=bool)
        coarse[0, :] = True
        coarse[:, 0] = True
        end_y = min(round((ny - 1) / step), level_ny - 1)
        end_x = min(round((nx - 1) / step), level_nx - 1)
        coarse[end_y:, :] = True
        coarse[:, end_x:] = True
        for row, lo, hi in positions:
            coarse[round(row / step), round(lo / step):round(hi / step) + 1] = True
        masks.append(coarse)
    return initial, masks, (ny, nx)


@dataclass(frozen=True)
class WarmStartResult:
    potential: np.ndarray
    converged: bool
    iterations: int
    final_change: float
    max_abs_residual: float
    warm_start_cycles: int
    warm_start_max_change: float
    warm_start_seconds: float
    jacobi_seconds: float
    padded_shape_yx: tuple[int, int]
    coarse_grid_shapes_yx: tuple[tuple[int, int], ...]


def solve_with_warm_start(config, *, max_cycles=200, progress=None):
    """Prepare an initial guess and execute actual Jacobi until tol or Nmax."""
    config = validate_config(config)
    if config["case"] != "parallel_plate":
        raise ValueError("warm-start validation supports parallel_plate only")
    if max_cycles <= 0:
        raise ValueError("max_cycles must be positive")
    initial, masks, physical_shape = _masks_and_initial(config)
    tolerance = float(config["solver"]["tol"])
    rhs = np.zeros((1, 1), dtype=np.float64)  # zero source on the fine level
    started = time.perf_counter()
    for cycle in range(1, max_cycles + 1):
        _cycle(0, initial, rhs, masks)
        change = float(_maximum_jacobi_change(initial, masks[0]))
        if progress is not None and (cycle == 1 or cycle % 10 == 0):
            progress(cycle, change)
        if not np.isfinite(change):
            raise ValueError("multigrid warm start produced a non-finite change")
        if change <= tolerance / 4:
            break
    else:
        raise ValueError(f"warm start did not reach tol/4 in {max_cycles} cycles; last change {change:g}")
    warm_seconds = time.perf_counter() - started
    started = time.perf_counter()
    # The actual method, stopping rule, fixed mask, and update order are the
    # same as the existing Numba Jacobi backend; only the initial guess differs.
    potential, iterations, final_change = compiled_jacobi(
        masks[0], initial, 0.25, 0.25, tolerance, config["solver"]["Nmax"])
    jacobi_seconds = time.perf_counter() - started
    residual = float(_maximum_laplace_residual(potential, masks[0], float(config["grid"]["h"])))
    if not np.isfinite(residual):
        raise ValueError("non-finite final residual")
    ny, nx = physical_shape
    return WarmStartResult(
        potential=potential[:ny, :nx], converged=bool(final_change <= tolerance),
        iterations=int(iterations), final_change=float(final_change), max_abs_residual=residual,
        warm_start_cycles=cycle, warm_start_max_change=change, warm_start_seconds=warm_seconds,
        jacobi_seconds=jacobi_seconds, padded_shape_yx=potential.shape,
        coarse_grid_shapes_yx=tuple(mask.shape for mask in masks[1:]),
    )
