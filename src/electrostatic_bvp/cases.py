"""Build grids and fixed-potential constraints from validated case inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from electrostatic_bvp.config import ConfigError, validate_config


@dataclass(frozen=True)
class CaseGrid:
    """Coordinates and constraints on arrays indexed as (y, x)."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fixed_mask: NDArray[np.bool_]
    fixed_values: NDArray[np.float64]
    positive_plate_mask: NDArray[np.bool_] | None = None
    negative_plate_mask: NDArray[np.bool_] | None = None


def build_rectangle_sine(config: dict[str, Any]) -> CaseGrid:
    """Construct the original rectangle boundary values and grid."""
    config = validate_config(config)
    if config["case"] != "rectangle_sine":
        raise ConfigError("expected case 'rectangle_sine'")
    lx, ly = (float(config["geometry"][key]) for key in ("Lx", "Ly"))
    h = float(config["grid"]["h"])
    v0 = float(config["boundary_conditions"]["V0"])
    nx, ny = round(lx / h), round(ly / h)
    x = np.linspace(0.0, lx, nx + 1, dtype=np.float64)
    y = np.linspace(0.0, ly, ny + 1, dtype=np.float64)
    fixed_mask = np.zeros((ny + 1, nx + 1), dtype=bool)
    fixed_mask[[0, -1], :] = True
    fixed_mask[:, [0, -1]] = True
    fixed_values = np.zeros(fixed_mask.shape, dtype=np.float64)
    fixed_values[-1, 1:-1] = v0 * np.sin(np.pi * x[1:-1] / lx)
    return CaseGrid(x=x, y=y, fixed_mask=fixed_mask, fixed_values=fixed_values)


def build_parallel_plate(config: dict[str, Any]) -> CaseGrid:
    """Construct an exactly indexed, centered, finite parallel-plate geometry.

    The validator rejects dimensions which cannot be represented on this grid;
    coordinates reflect integer multiples of h and preserve the input separately.
    """
    config = validate_config(config)
    if config["case"] != "parallel_plate":
        raise ConfigError("expected case 'parallel_plate'")
    geometry = config["geometry"]
    h = float(config["grid"]["h"])
    a = round(float(geometry["domain_half_width"]) / h)
    b = round(float(geometry["domain_half_height"]) / h)
    half_plate = round((float(geometry["plate_length"]) / 2) / h)
    half_gap = round((float(geometry["gap"]) / 2) / h)
    x = np.arange(-a, a + 1, dtype=np.float64) * h
    y = np.arange(-b, b + 1, dtype=np.float64) * h
    shape = (len(y), len(x))
    positive = np.zeros(shape, dtype=bool)
    negative = np.zeros(shape, dtype=bool)
    positive[b + half_gap, a - half_plate:a + half_plate + 1] = True
    negative[b - half_gap, a - half_plate:a + half_plate + 1] = True
    fixed_mask = positive | negative
    fixed_mask[[0, -1], :] = True
    fixed_mask[:, [0, -1]] = True
    fixed_values = np.zeros(shape, dtype=np.float64)
    v0 = float(config["boundary_conditions"]["V0"])
    fixed_values[positive] = v0 / 2
    fixed_values[negative] = -v0 / 2
    return CaseGrid(x, y, fixed_mask, fixed_values, positive, negative)
