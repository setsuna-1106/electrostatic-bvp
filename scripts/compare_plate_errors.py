"""Separate iteration, frame and mesh effects using the open-space plate model."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import numpy as np

from electrostatic_bvp.cases import build_parallel_plate
from electrostatic_bvp.config import load_config
from electrostatic_bvp.fields import parallel_plate_field
from electrostatic_bvp.open_plate import evaluate_open_grid, fit_open_plate
from electrostatic_bvp.solver import solve_jacobi
from validate_m2 import make_config, source_hashes, write_json

ROOT = Path(__file__).resolve().parents[1]


def shared_nodes(grid, h, x, y):
    """Indices of the benchmark's aligned common nodes."""
    ix = len(grid.x) // 2 + np.rint(x / h).astype(int)
    iy = len(grid.y) // 2 + np.rint(y / h).astype(int)
    return np.ix_(iy, ix)


def fd_sample(config, reference, x, y, *, open_trace, backend):
    grid = build_parallel_plate(config)
    h = config["grid"]["h"]
    fixed = grid.fixed_values
    if open_trace:
        # Put the continuum open-space potential on the finite-difference
        # frame. Everything inside (stencil, plates, grid) is unchanged.
        fixed[[0, -1], :] = reference.potential(grid.x[None, :], grid.y[[0, -1], None])
        fixed[:, [0, -1]] = reference.potential(grid.x[[0, -1]][None, :], grid.y[:, None])
    result = solve_jacobi(grid.fixed_mask, fixed, dx=h, dy=h, tol=config["solver"]["tol"],
                          nmax=config["solver"]["Nmax"], backend=backend)
    if not result.converged:
        raise ValueError(f"Jacobi did not converge: {config}")
    field = parallel_plate_field(grid.x, grid.y, result.potential,
                                 grid.positive_plate_mask, grid.negative_plate_mask, h=h)
    # All comparisons use the original grid's nodes, even for finer solves.
    index = shared_nodes(grid, h, x, y)
    return (result.potential[index], field.electric_field_x[index], field.electric_field_y[index]), {
        "config": config, "open_space_trace_on_frame": open_trace, "iterations": result.iterations,
        "last_change": result.final_change, "max_abs_laplace_residual": result.max_abs_residual,
    }


def compare(first, second, x, y, valid, central):
    dv = np.abs(first[0] - second[0])
    de = np.hypot(first[1] - second[1], first[2] - second[2])
    row, col = np.unravel_index(np.argmax(np.where(valid, de, -np.inf)), de.shape)
    outside = np.abs(x - 2.5).argmin()
    return {
        # This benchmark has V0=1 and d=1, so its normalization factors are one.
        "max_potential_difference_over_voltage": float(np.max(dv[valid])),
        "max_field_difference_over_reference": float(np.max(de[valid])),
        "max_field_difference_xy": [float(x[col]), float(y[row])],
        "central_max_field_difference_over_reference": float(np.max(de[central])),
        "outside_point_field_difference_over_reference": float(de[len(y) // 2, outside]),
    }


def plot(path, x, y, runs, exact, valid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    panels = (("grounded", "Grounded frame, h/d = 0.10"),
              ("open_trace", "Open-space potential on frame, h/d = 0.10"),
              ("open_trace_fine", "Open-space potential on frame, h/d = 0.025"))
    errors = [100 * np.hypot(runs[k][1] - exact[1], runs[k][2] - exact[2]) for k, _ in panels]
    norm = Normalize(vmin=0, vmax=max(float(np.max(e[valid])) for e in errors))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), layout="constrained", sharey=True)
    for ax, (_, title), error in zip(axes, panels, errors):
        image = ax.pcolormesh(x, y, np.ma.masked_where(~valid, error),
                              cmap="magma", norm=norm, shading="nearest")
        for yy in (-0.5, 0.5):
            ax.plot([-2, 2], [yy, yy], color="steelblue", lw=1.4)
        ax.set(title=title, xlabel="x / d", aspect="equal")
    axes[0].set_ylabel("y / d")
    fig.colorbar(image, ax=axes, label=r"$|\Delta\mathbf{E}|/E_{ref}$ (%)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/plate-error-analysis"))
    parser.add_argument("--backend", choices=("numba", "numpy"), default="numba")
    args = parser.parse_args()
    base = load_config(ROOT / "configs/parallel_plate.json")
    reference = fit_open_plate(4.0, 1.0, 1.0)
    # This grid is only the common comparison window, not a modeled enclosure.
    target_grid = build_parallel_plate(make_config(base, a=3.0, b=1.5))
    model = evaluate_open_grid(target_grid, reference, 0.1)
    x, y = model.x, model.y
    runs, meta = {}, {}
    for name, options, open_trace in (
        ("loose", {"tol": 1e-7}, False),
        ("grounded", {"tol": 1e-9}, False),
        ("grounded_tight", {}, False),
        ("larger_frame", {"a": 12.0, "b": 8.0}, False),
        ("open_trace", {}, True),
        ("open_trace_mid", {"h": 0.05}, True),
        ("open_trace_fine", {"h": 0.025}, True),
    ):
        runs[name], meta[name] = fd_sample(make_config(base, **options), reference, x, y,
                                          open_trace=open_trace, backend=args.backend)
        print(f"{name}: {meta[name]['iterations']} iterations", flush=True)

    valid = model.field.analysis_valid_mask
    central = valid & (np.abs(x)[None, :] <= 1.0) & (np.abs(y)[:, None] <= 0.25)
    runs["open_reference"] = (model.potential, model.field.electric_field_x, model.field.electric_field_y)
    # Use the parent frame's full grid so even target-edge derivatives are centered.
    full_grid = build_parallel_plate(base)
    continuum = reference.potential(full_grid.x[None, :], full_grid.y[:, None])
    sampled_field = parallel_plate_field(full_grid.x, full_grid.y, continuum,
                                         full_grid.positive_plate_mask, full_grid.negative_plate_mask, h=0.1)
    window = shared_nodes(full_grid, 0.1, x, y)
    runs["sampled_reference_stencil"] = (continuum[window],
                                        sampled_field.electric_field_x[window],
                                        sampled_field.electric_field_y[window])
    comparisons = {
        "iteration_loose_to_default": ("loose", "grounded"),
        "iteration_default_to_tight": ("grounded", "grounded_tight"),
        "frame_enlargement": ("grounded_tight", "larger_frame"),
        "frame_alone_at_h_0_1": ("grounded_tight", "open_trace"),
        "mesh_at_h_0_1": ("open_trace", "open_reference"),
        "mesh_at_h_0_05": ("open_trace_mid", "open_reference"),
        "mesh_at_h_0_025": ("open_trace_fine", "open_reference"),
        "differentiation_on_continuum_samples_h_0_1": ("sampled_reference_stencil", "open_reference"),
        "discrete_potential_effect_on_field_h_0_1": ("open_trace", "sampled_reference_stencil"),
        "original_total": ("grounded", "open_reference"),
        "larger_grounded_frame_total": ("larger_frame", "open_reference"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "report.json", {
        "comparison_region": {"x": [-3, 3], "y": [-1.5, 1.5], "spacing": 0.1,
                              "excluded_tip_radius": 0.2, "plate_nodes_excluded": True},
        "reference": {
            "cosine_modes": len(reference.coefficients),
            "plate_potential_check_unit_voltage": reference.boundary_error_over_voltage,
            "successive_relative_coefficient_change": reference.relative_coefficient_change,
            "center_negative_ey_over_reference": float(-reference.field(0, 0)[1]),
            "outside_negative_ey_over_reference": float(-reference.field(2.5, 0)[1]),
        },
        "runs": meta,
        "comparisons": {label: compare(runs[first], runs[second], x, y, valid, central)
                        for label, (first, second) in comparisons.items()},
        "software": {"backend": args.backend, "python": platform.python_version(), "numpy": np.__version__,
                     "source_sha256": source_hashes((
                         "scripts/compare_plate_errors.py", "scripts/validate_m2.py",
                         "src/electrostatic_bvp/open_plate.py",
                         "src/electrostatic_bvp/solver.py", "src/electrostatic_bvp/acceleration.py",
                         "src/electrostatic_bvp/fields.py", "src/electrostatic_bvp/cases.py",
                         "src/electrostatic_bvp/config.py", "configs/parallel_plate.json", "uv.lock",
                     ))},
    })
    plot(args.output_dir / "field-error-comparison.png", x, y, runs, runs["open_reference"], valid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
