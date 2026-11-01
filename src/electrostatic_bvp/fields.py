"""Electric fields and analysis regions on the project's (y, x) grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class PlateField:
    """Bulk field, its validity mask, and the separate endpoint analysis mask."""

    electric_field_x: FloatArray
    electric_field_y: FloatArray
    field_valid_mask: BoolArray
    analysis_valid_mask: BoolArray
    endpoint_exclusion_radius: float
    endpoint_exclusion_has_two_intervals: bool


def electric_field(
    potential: FloatArray, *, dx: float, dy: float,
) -> tuple[FloatArray, FloatArray]:
    """Return (Ex, Ey) = -grad(phi), centered inside and one-sided at edges."""
    derivative_y, derivative_x = np.gradient(potential, dy, dx, edge_order=2)
    return -derivative_x, -derivative_y


def parallel_plate_field(
    x: FloatArray,
    y: FloatArray,
    potential: FloatArray,
    positive_plate_mask: BoolArray,
    negative_plate_mask: BoolArray,
    *,
    h: float,
) -> PlateField:
    """Compute bulk field without assigning a field on either conductor.

    For this node-aligned, zero-thickness geometry, a centered stencil at a
    free node may terminate at a plate node, but cannot cross a plate. The
    outer-edge stencil can likewise terminate at a plate because validation
    keeps plates at least two grid intervals from the frame.
    """
    positive_y, positive_x = np.where(positive_plate_mask)
    negative_y, negative_x = np.where(negative_plate_mask)
    gap = float(y[positive_y[0]] - y[negative_y[0]])
    ex, ey = electric_field(potential, dx=h, dy=h)
    valid = ~(positive_plate_mask | negative_plate_mask)
    ex[~valid] = np.nan
    ey[~valid] = np.nan

    radius = 0.2 * gap
    x_grid, y_grid = np.meshgrid(x, y)
    excluded = np.zeros(potential.shape, dtype=bool)
    for ys, xs in ((positive_y, positive_x), (negative_y, negative_x)):
        for col in (xs[0], xs[-1]):
            distance = np.hypot(x_grid - x[col], y_grid - y[ys[0]])
            excluded |= distance <= radius + h * 1e-9
    # At least two grid intervals across the radius is a conservative flag
    # for later mesh comparisons, not an accuracy guarantee.
    resolved = radius / h >= 2 - 1e-9
    return PlateField(ex, ey, valid, valid & ~excluded, radius, bool(resolved))


def parallel_plate_metrics(
    x: FloatArray,
    y: FloatArray,
    field: PlateField,
    positive_plate_mask: BoolArray,
    negative_plate_mask: BoolArray,
    *,
    v0: float,
    h: float,
) -> dict:
    """Compute normalized descriptive quantities, never an analytic error."""
    pos_y, pos_x = np.where(positive_plate_mask)
    neg_y, _ = np.where(negative_plate_mask)
    gap = float(y[pos_y[0]] - y[neg_y[0]])
    length = float(x[pos_x[-1]] - x[pos_x[0]])
    center_x, center_y = len(x) // 2, len(y) // 2
    # Geometry and gap are integer multiples of h, so this point is on a node
    # when it lies inside the domain; no interpolation or snapping is needed.
    outside_x_index = int(pos_x[-1] + (pos_y[0] - neg_y[0]) // 2)
    outside_xy = [float(x[pos_x[-1]] + gap / 2), 0.0]
    xg, yg = np.meshgrid(x, y)
    central = (np.abs(xg) <= length / 4 + h * 1e-9) & (np.abs(yg) <= gap / 4 + h * 1e-9)
    central &= field.analysis_valid_mask
    metrics = {
        "reference_field_magnitude": abs(v0) / gap,
        "reference_field_y": -v0 / gap,
        "normalization_unavailable_reason": None,
        "center_negative_ey_over_reference": None,
        "central_max_vector_deviation_over_reference": None,
        "outside_negative_ey_over_reference": None,
        "outside_point_xy": outside_xy,
        "outside_point_unavailable_reason": None,
        "central_region": {
            "definition": "|x| <= L/4 and |y| <= d/4; analysis_valid_mask only",
            "x_bounds": [-length / 4, length / 4],
            "y_bounds": [-gap / 4, gap / 4],
            "node_count": int(np.count_nonzero(central)),
        },
        "endpoint_exclusion": {
            "radius": field.endpoint_exclusion_radius,
            "definition": "exclude nodes at distance <= 0.2*d from any plate endpoint",
            "radius_grid_intervals": field.endpoint_exclusion_radius / h,
            "has_two_grid_intervals": field.endpoint_exclusion_has_two_intervals,
            "resolution_rule": "radius >= 2*h is a conservative sampling flag, not an accuracy guarantee",
        },
    }
    if v0 == 0:
        metrics["normalization_unavailable_reason"] = "V0 is zero, so the reference field magnitude is zero."
    else:
        reference = metrics["reference_field_magnitude"]
        metrics["center_negative_ey_over_reference"] = float(-field.electric_field_y[center_y, center_x] / reference)
        deviation = np.hypot(field.electric_field_x[central],
                             field.electric_field_y[central] - metrics["reference_field_y"])
        metrics["central_max_vector_deviation_over_reference"] = float(np.max(deviation) / reference)
        if outside_x_index < len(x) and field.analysis_valid_mask[center_y, outside_x_index]:
            metrics["outside_negative_ey_over_reference"] = float(
                -field.electric_field_y[center_y, outside_x_index] / reference
            )
        else:
            metrics["outside_point_unavailable_reason"] = "Requested point lies outside the grid or analysis mask."
    if v0 == 0:
        metrics["outside_point_unavailable_reason"] = metrics["normalization_unavailable_reason"]
    return metrics
