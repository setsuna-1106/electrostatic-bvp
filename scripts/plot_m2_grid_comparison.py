"""Plot an S4 common-node electric-field difference and endpoint exclusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

if __package__:
    from .validate_m2 import TARGET, compare_cases, load_case
else:
    from validate_m2 import TARGET, compare_cases, load_case


def load_run(directory):
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    return load_case(directory, metadata["config"], metadata["provenance"])


def save_figure(first, second, output, *, target=TARGET):
    """Plot only common valid physical nodes, with no interpolation."""
    result = compare_cases(first, second, target)
    if not result["both_converged"]:
        raise ValueError("Both runs must converge before plotting a grid comparison")
    if not result["both_endpoint_radii_have_two_intervals"]:
        raise ValueError("Endpoint exclusion radius must cover at least two intervals in both grids")
    if first.config["grid"]["h"] < second.config["grid"]["h"]:
        first, second = second, first
    a, b = first.arrays, second.arrays
    hx, hy = a["x"], a["y"]
    fine_h = float(second.config["grid"]["h"])
    epsilon = float(first.config["grid"]["h"]) * 1e-8
    xmin, xmax, ymin, ymax = target
    ix = np.flatnonzero((hx >= xmin - epsilon) & (hx <= xmax + epsilon))
    iy = np.flatnonzero((hy >= ymin - epsilon) & (hy <= ymax + epsilon))
    jx = np.rint((hx[ix] - b["x"][0]) / fine_h).astype(int)
    jy = np.rint((hy[iy] - b["y"][0]) / fine_h).astype(int)
    ia, ib = np.ix_(iy, ix), np.ix_(jy, jx)
    mask = a["analysis_valid_mask"][ia] & b["analysis_valid_mask"][ib]
    norm = abs(first.config["boundary_conditions"]["V0"]) / first.config["geometry"]["gap"]
    dx = a["electric_field_x"][ia] - b["electric_field_x"][ib]
    dy = a["electric_field_y"][ia] - b["electric_field_y"][ib]
    difference = np.hypot(dx, dy) / norm
    valid_difference = np.ma.array(difference, mask=~mask)
    peak = np.unravel_index(np.argmax(np.where(mask, difference, -1)), difference.shape)
    peak_x, peak_y = float(hx[ix[peak[1]]]), float(hy[iy[peak[0]]])
    geo = first.config["geometry"]
    half_plate, half_gap = geo["plate_length"] / 2, geo["gap"] / 2
    radius = result["endpoint_exclusion_radius"]

    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    mesh = ax.pcolormesh(hx[ix], hy[iy], valid_difference, cmap="magma", shading="auto", vmin=0)
    for y_plate in (-half_gap, half_gap):
        ax.plot([-half_plate, half_plate], [y_plate, y_plate], color="white", linewidth=4.5, zorder=4)
        ax.plot([-half_plate, half_plate], [y_plate, y_plate], color="black", linewidth=2, zorder=5)
        for x_plate in (-half_plate, half_plate):
            ax.add_patch(Circle((x_plate, y_plate), radius, facecolor="white", edgecolor="#4b7698",
                                linewidth=1, linestyle="--", zorder=3))
    ax.scatter([peak_x], [peak_y], marker="x", s=55, c="#46eafa", linewidths=1.5, zorder=6,
               label=f"Maximum {100 * difference[peak]:.2f}% at ({peak_x:g}, {peak_y:g})")
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="x (length units)", ylabel="y (length units)",
           title=f"Field difference: h={first.config['grid']['h']:g} vs h={fine_h:g}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8)
    fig.colorbar(mesh, ax=ax, label=r"$|\Delta E| / E_{ref}$", shrink=0.88)
    ax.text(0.5, -0.17, "Dashed circles: 0.2 d endpoint exclusion; plate nodes masked.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return result, (peak_x, peak_y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True, help="directory of the coarser S4 run")
    parser.add_argument("--fine", type=Path, required=True, help="directory of the finer S4 run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison, peak = save_figure(load_run(args.coarse), load_run(args.fine), args.output)
    print(f"Saved {args.output}: max |ΔE|/Eref={comparison['max_field_vector_difference_over_eref']:.6g} at {peak}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print(f"Grid comparison plot failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
