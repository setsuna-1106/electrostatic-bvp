"""Open-space solution for two centered, zero-thickness plates in 2D."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from electrostatic_bvp.cases import CaseGrid, build_parallel_plate
from electrostatic_bvp.fields import PlateField, parallel_plate_field


FIT_TOLERANCE = 1e-11


@dataclass(frozen=True)
class OpenPlate:
    half_length: float
    gap: float
    voltage: float
    coefficients: np.ndarray
    boundary_error_over_voltage: float
    relative_coefficient_change: float

    def _coordinates(self, x, y, plate_y):
        z = np.asarray(x) / self.half_length + 1j * (
            (np.asarray(y) - plate_y) / self.half_length
        )
        # This branch has s(z) ~ z at infinity, and |w| <= 1.
        s = np.sqrt(z - 1) * np.sqrt(z + 1)
        return 1 / (z + s), s

    def potential(self, x, y):
        """Evaluate the continuous potential, including on the plates."""
        upper, _ = self._coordinates(x, y, self.gap / 2)
        lower, _ = self._coordinates(x, y, -self.gap / 2)
        c = self.coefficients
        # Real part of the integrated Cauchy transform. Constants from the
        # logarithmic potentials cancel between equal and opposite charges.
        primitive = np.r_[0.0, c[1:] / np.arange(1, len(c))]
        poly = np.polynomial.polynomial.polyval
        return self.voltage * (
            c[0] * (np.log(np.abs(upper)) - np.log(np.abs(lower)))
            + (poly(upper, primitive) - poly(lower, primitive)).real
        )

    def field(self, x, y):
        """Evaluate the bulk field off the plates (do not pass conductor points)."""
        upper, su = self._coordinates(x, y, self.gap / 2)
        lower, sl = self._coordinates(x, y, -self.gap / 2)
        poly = np.polynomial.polynomial.polyval
        value = (self.voltage / self.half_length) * (
            poly(upper, self.coefficients) / su - poly(lower, self.coefficients) / sl
        )
        return value.real, -value.imag


def fit_open_plate(length, gap, voltage):
    """Fit a cosine series for the weighted charge on the upper plate.

    The lower charge is its negative. Changing x' to (L/2)*cos(theta')
    absorbs the inverse-square-root endpoint factor. Treat the logarithmic
    self-interaction analytically in the cosine basis; integrate the smooth
    interaction with the other plate by midpoint quadrature in theta'.
    Coefficients use unit voltage, also permitting V0=0.
    """
    half = length / 2
    previous = None
    for n in (16, 32, 64, 128, 256, 512, 1024):
        theta = (np.arange(n) + 0.5) * np.pi / n
        t = np.cos(theta)
        cosines = np.cos(theta[:, None] * np.arange(n))
        self_weights = np.r_[np.log(2), 1 / np.arange(1, n)]
        other = np.log(np.hypot(t[:, None] - t, gap / half))
        matrix = cosines * self_weights + (other @ cosines) / n
        try:
            coefficients = np.linalg.solve(matrix, np.full(n, 0.5))
        except np.linalg.LinAlgError as exc:
            raise ValueError("open-space charge fit failed") from exc
        # Enforce the exact reflection symmetry of the centered plates.
        coefficients[1::2] = 0
        sample = half * np.cos(np.linspace(0, np.pi, 4 * n + 1))
        unit = OpenPlate(half, gap, 1.0, coefficients, 0.0, 0.0)
        boundary_error = float(np.max(np.abs(unit.potential(sample, gap / 2) - 0.5)))
        if previous is not None:
            delta = np.linalg.norm(coefficients - np.pad(previous, (0, n - len(previous))), 1)
            change = float(delta / np.linalg.norm(coefficients, 1))
            if boundary_error <= FIT_TOLERANCE and change <= FIT_TOLERANCE:
                return OpenPlate(half, gap, voltage, coefficients, boundary_error, change)
        previous = coefficients
    raise ValueError("open-space charge fit did not stabilize through 1024 cosine modes")


@dataclass(frozen=True)
class OpenPlateResult:
    x: np.ndarray
    y: np.ndarray
    potential: np.ndarray
    fixed_mask: np.ndarray
    positive_plate_mask: np.ndarray
    negative_plate_mask: np.ndarray
    field: PlateField
    model: OpenPlate
    converged: bool = True  # fit_open_plate raises if its checks fail


def evaluate_open_grid(grid: CaseGrid, model: OpenPlate, h: float) -> OpenPlateResult:
    """Sample the open-space solution; the rectangle only limits the output."""
    pos, neg = grid.positive_plate_mask, grid.negative_plate_mask
    assert pos is not None and neg is not None
    potential = model.potential(grid.x[None, :], grid.y[:, None])
    potential[pos | neg] = grid.fixed_values[pos | neg]
    field = parallel_plate_field(grid.x, grid.y, potential, pos, neg, h=h)
    iy, ix = np.nonzero(field.field_valid_mask)
    # Reuse the geometric masks, replacing the finite-difference derivatives.
    field.electric_field_x[iy, ix], field.electric_field_y[iy, ix] = model.field(
        grid.x[ix], grid.y[iy]
    )
    return OpenPlateResult(grid.x, grid.y, potential, pos | neg, pos, neg, field, model)


def solve_open_plate(config: dict) -> OpenPlateResult:
    grid = build_parallel_plate(config)
    pos, neg = grid.positive_plate_mask, grid.negative_plate_mask
    assert pos is not None and neg is not None
    rows, columns = np.nonzero(pos)
    # Use the actual node-aligned plate geometry, shared with the Jacobi case.
    model = fit_open_plate(
        float(grid.x[columns[-1]] - grid.x[columns[0]]),
        float(grid.y[rows[0]] - grid.y[np.nonzero(neg)[0][0]]),
        float(config["boundary_conditions"]["V0"]),
    )
    return evaluate_open_grid(grid, model, float(config["grid"]["h"]))
