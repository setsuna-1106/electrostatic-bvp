"""Noninteractive figures for the finite parallel-plate case."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from electrostatic_bvp.solver import ParallelPlateResult


def _plates_and_frame(ax, result: ParallelPlateResult) -> tuple[float, float, float, float]:
    x, y = result.x, result.y
    pos_y, pos_x = np.where(result.positive_plate_mask)
    neg_y, _ = np.where(result.negative_plate_mask)
    left, right = float(x[pos_x[0]]), float(x[pos_x[-1]])
    lower, upper = float(y[neg_y[0]]), float(y[pos_y[0]])
    for plate_y in (lower, upper):
        # A light underlay keeps the real plate visible on both colormaps.
        ax.plot([left, right], [plate_y, plate_y], color="white", lw=4.5, zorder=8)
        ax.plot([left, right], [plate_y, plate_y], color="#151515", lw=2.4, zorder=9)
    ax.plot([x[0], x[-1], x[-1], x[0], x[0]],
            [y[0], y[0], y[-1], y[-1], y[0]], color="#151515", lw=1, zorder=7)
    ax.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), xlabel="x (length units)", ylabel="y (length units)")
    ax.set_aspect("equal", adjustable="box")
    return left, right, lower, upper


def _status(result: ParallelPlateResult) -> str:
    return "" if result.converged else " — NOT CONVERGED"


def _arrow_lengths(x, y, ux, uy, length, left, right, plate_ys, h):
    """Keep arrow shafts on their starting side of both conductor lines."""
    lengths = np.full(x.shape, length, dtype=float)
    for plate_y in plate_ys:
        t = np.full(x.shape, np.inf)
        np.divide(plate_y - y, uy, out=t, where=np.abs(uy) > 1e-14)
        crossing_x = x + np.where(np.isfinite(t), t, 0.0) * ux
        hits = (t > 0) & np.isfinite(t) & (crossing_x >= left) & (crossing_x <= right)
        lengths[hits] = np.minimum(lengths[hits], 0.8 * t[hits])
        # A horizontal arrow in the plate's row may also hit its endpoint.
        horizontal = (np.abs(y - plate_y) < h * 1e-8) & (np.abs(uy) <= 1e-14)
        for hit, distance in (
            (horizontal & (x < left) & (ux > 0), left - x),
            (horizontal & (x > right) & (ux < 0), x - right),
        ):
            lengths[hit] = np.minimum(lengths[hit], 0.8 * distance[hit] / np.abs(ux[hit]))
    return lengths


def save_parallel_plate_figures(result: ParallelPlateResult, config: dict, output_dir: str | Path) -> dict[str, str]:
    """Save potential, masked field, and dimensionless midline figures."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    x, y, u = result.x, result.y, result.potential
    field = result.field
    h = float(config["grid"]["h"])
    v0 = float(config["boundary_conditions"]["V0"])
    filenames = {"potential_figure": "potential.png", "field_figure": "electric_field.png",
                 "midline_figure": "midline.png"}

    fig, ax = plt.subplots(figsize=(9, 6.3), layout="constrained")
    extent = max(float(np.max(np.abs(u))), abs(v0) / 2)
    if extent == 0:
        extent = 1.0
    image = ax.pcolormesh(x, y, u, shading="auto", cmap="RdBu_r", vmin=-extent, vmax=extent)
    if float(np.ptp(u)) > 0:
        levels = np.linspace(-extent, extent, 11)[1:-1]
        ax.contour(x, y, u, levels=levels, colors="#303030", linewidths=0.55, alpha=0.65)
    _plates_and_frame(ax, result)
    ax.set_title("Electric potential and equipotentials" + _status(result))
    fig.colorbar(image, ax=ax, label="Potential (voltage units)", shrink=0.85)
    fig.savefig(destination / filenames["potential_figure"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6.3), layout="constrained")
    left, right, lower, upper = _plates_and_frame(ax, result)
    step_x = max(1, int(np.ceil(len(x) / 27)))
    step_y = max(1, int(np.ceil(len(y) / 21)))
    xs, ys = x[::step_x], y[::step_y]
    xg, yg = np.meshgrid(xs, ys)
    ex = field.electric_field_x[::step_y, ::step_x]
    ey = field.electric_field_y[::step_y, ::step_x]
    valid = field.field_valid_mask[::step_y, ::step_x]
    magnitude = np.hypot(ex, ey)
    drawable = valid & np.isfinite(magnitude) & (magnitude > 0)
    max_magnitude = float(np.max(magnitude[drawable])) if drawable.any() else 0.0
    norm = Normalize(vmin=0, vmax=max_magnitude if max_magnitude > 0 else 1)
    if drawable.any():
        ux = np.zeros_like(ex)
        uy = np.zeros_like(ey)
        np.divide(ex, magnitude, out=ux, where=drawable)
        np.divide(ey, magnitude, out=uy, where=drawable)
        base_length = 0.04 * min(float(x[-1] - x[0]), float(y[-1] - y[0]))
        lengths = _arrow_lengths(xg, yg, ux, uy, base_length, left, right, (lower, upper), h)
        ax.quiver(xg, yg, np.ma.array(ux * lengths, mask=~drawable),
                  np.ma.array(uy * lengths, mask=~drawable),
                  np.ma.array(magnitude, mask=~drawable), cmap="viridis", norm=norm,
                  angles="xy", scale_units="xy", scale=1, width=0.0024,
                  pivot="tail", zorder=4)
    else:
        ax.text(0.5, 0.04, "Electric field is zero", transform=ax.transAxes, ha="center")
    ax.set_title("Electric field" + _status(result))
    ax.text(0.5, -0.13, "Arrow lengths normalized; shortened near conductors. Color: field magnitude.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax,
                 label="|E| (voltage / length)", shrink=0.85)
    fig.savefig(destination / filenames["field_figure"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    gap = upper - lower
    ax.axvline(left / gap, color="#555555", lw=1, ls="--", label="Plate endpoints")
    ax.axvline(right / gap, color="#555555", lw=1, ls="--")
    if v0 == 0:
        ax.text(0.5, 0.5, "Normalized field is undefined when V0 = 0",
                transform=ax.transAxes, ha="center", va="center")
    else:
        ref = abs(v0) / gap
        row = len(y) // 2
        profile = np.ma.array(-field.electric_field_y[row] / ref,
                              mask=~field.field_valid_mask[row])
        ax.plot(x / gap, profile, color="#1564a8", lw=2, label=r"$-E_y/E_{ref}$ at $y=0$")
        ax.axhline(np.sign(v0), color="#a66819", lw=1, ls=":", label="Uniform-field reference")
    ax.set(xlabel=r"$x/d$", ylabel=r"$-E_y/E_{ref}$", title="Midline field" + _status(result))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(destination / filenames["midline_figure"], dpi=160)
    plt.close(fig)
    return filenames
