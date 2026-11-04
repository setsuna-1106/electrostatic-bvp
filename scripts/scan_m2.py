"""Compare finite parallel plates across the planned one-dimensional scans.

Each geometry has two nested mesh comparisons at the same frame and an
independent frame expansion at the intermediate mesh. Existing S4 baseline
runs can be copied only after their complete provenance and hashes match.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from electrostatic_bvp.config import load_config, validate_config
from electrostatic_bvp.fields import PlateField, parallel_plate_metrics
if __package__:
    from .validate_m2 import (
        DOMAIN_MESH_TARGET, ITERATION_TARGET, ROOT, algebraic_bounds, compare_cases,
        load_case, make_config, meets_tolerance, solve_record, source_hashes, write_json,
    )
else:
    from validate_m2 import (
        DOMAIN_MESH_TARGET, ITERATION_TARGET, ROOT, algebraic_bounds, compare_cases,
        load_case, make_config, meets_tolerance, solve_record, source_hashes, write_json,
    )


# The shared L=4, d=1 baseline appears in both plots but is solved only once.
GEOMETRIES = (
    ("L2-d1", 2.0, 1.0),
    ("L4-d1", 4.0, 1.0),
    ("L8-d1", 8.0, 1.0),
    ("L4-d05", 4.0, 0.5),
    ("L4-d2", 4.0, 2.0),
)
SCANS = {
    "plate_length": ("L2-d1", "L4-d1", "L8-d1"),
    "gap": ("L4-d05", "L4-d1", "L4-d2"),
}
RUNS = {
    "coarse": (0.1, 12.0, 8.0),
    "medium": (0.05, 12.0, 8.0),
    "fine": (0.025, 12.0, 8.0),
    "expanded_medium": (0.05, 18.0, 12.0),
}
S4_BASELINE_SEEDS = {
    "coarse": "domain-a12-b8", "medium": "mesh-h005", "fine": "mesh-h0025",
}


def configuration(base, length, gap, run, nmax):
    """Build a node-aligned case with identical dimensionless frame and mesh."""
    h_over_d, a_over_d, b_over_d = RUNS[run]
    config = make_config(base, h=h_over_d * gap, a=a_over_d * gap,
                         b=b_over_d * gap, nmax=nmax)
    config["geometry"]["plate_length"] = length
    config["geometry"]["gap"] = gap
    return validate_config(config)


def target_region(length, gap):
    """Cover the whole plate and one gap beyond each end, with no interpolation."""
    return (-length / 2 - gap, length / 2 + gap, -1.5 * gap, 1.5 * gap)


def copy_seed(source, destination, config, provenance):
    """Copy a prior run only after source and destination are both verified."""
    load_case(source, config, provenance)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    try:
        load_case(destination, config, provenance)
    except Exception:
        shutil.rmtree(destination)
        raise


def descriptive_metrics(case):
    a = case.arrays
    field = PlateField(a["electric_field_x"], a["electric_field_y"],
                       a["field_valid_mask"], a["analysis_valid_mask"],
                       case.metadata["endpoint_exclusion_radius"],
                       case.metadata["endpoint_exclusion_has_two_intervals"])
    return parallel_plate_metrics(a["x"], a["y"], field,
                                  a["positive_plate_mask"], a["negative_plate_mask"],
                                  v0=float(case.config["boundary_conditions"]["V0"]),
                                  h=float(case.config["grid"]["h"]))


def analyze_geometry(length, gap, cases):
    """Record actual comparisons; one domain comparison is not full S4 acceptance."""
    target = target_region(length, gap)
    pairs = {
        "mesh_coarse_medium": compare_cases(cases["coarse"], cases["medium"], target),
        "mesh_medium_fine": compare_cases(cases["medium"], cases["fine"], target),
        "domain_medium": compare_cases(cases["medium"], cases["expanded_medium"], target),
    }
    for comparison in pairs.values():
        comparison["within_suggested_scale"] = meets_tolerance(
            comparison, DOMAIN_MESH_TARGET, require_iteration_bound=True)
    metrics_by_run = {name: descriptive_metrics(case) for name, case in cases.items()}
    metric_keys = ("center_negative_ey_over_reference", "outside_negative_ey_over_reference")
    metric_changes = {
        name: {key: abs(metrics_by_run[first][key] - metrics_by_run[second][key])
               for key in metric_keys}
        for name, first, second in (
            ("mesh_medium_fine", "medium", "fine"),
            ("domain_medium", "medium", "expanded_medium"),
        )
    }
    all_converged = all(case.metadata["converged"] for case in cases.values())
    within_scale = all(pair["within_suggested_scale"] for pair in pairs.values())
    return {
        "plate_length": length,
        "gap": gap,
        "length_over_gap": length / gap,
        "target_region_xy": list(target),
        "target_region_definition": "|x| <= L/2+d, |y| <= 1.5d",
        "metrics_source": "fine",
        "metrics": metrics_by_run["fine"],
        "metrics_by_run": metrics_by_run,
        "point_metric_absolute_changes": metric_changes,
        "comparisons": pairs,
        "all_converged": all_converged,
        "reliability_status": "provisional" if all_converged and within_scale else "limited",
        "reliability_note": (
            "Only one domain expansion at h/d=0.05; no expanded-frame check at h/d=0.025. "
            "Suggested pairwise scales do not bound total error or establish open-space accuracy."
        ),
    }


def compare_scaled_geometries(first, second):
    """Compare matching dimensionless geometries on their normalized nodes."""
    ga, gb = first.config["geometry"], second.config["geometry"]
    da, db = float(ga["gap"]), float(gb["gap"])
    va = float(first.config["boundary_conditions"]["V0"])
    vb = float(second.config["boundary_conditions"]["V0"])
    if va * vb <= 0:
        raise ValueError("Scaled-geometry comparison requires nonzero voltages with the same sign")
    for key in ("plate_length", "domain_half_width", "domain_half_height"):
        if not np.isclose(ga[key] / da, gb[key] / db, atol=1e-10, rtol=0):
            raise ValueError("Dimensionless geometries must match")
    if not np.isclose(first.config["grid"]["h"] / da,
                      second.config["grid"]["h"] / db, atol=1e-10, rtol=0):
        raise ValueError("Dimensionless grid spacings must match")
    a, b = first.arrays, second.arrays
    if (a["potential"].shape != b["potential"].shape
            or not np.allclose(a["x"] / da, b["x"] / db, atol=1e-10, rtol=0)
            or not np.allclose(a["y"] / da, b["y"] / db, atol=1e-10, rtol=0)
            or not np.array_equal(a["analysis_valid_mask"], b["analysis_valid_mask"])):
        raise ValueError("Normalized coordinates and analysis masks must match")
    valid = a["analysis_valid_mask"]
    du = a["potential"] / abs(va) - b["potential"] / abs(vb)
    dex = a["electric_field_x"] / (abs(va) / da) - b["electric_field_x"] / (abs(vb) / db)
    dey = a["electric_field_y"] / (abs(va) / da) - b["electric_field_y"] / (abs(vb) / db)
    return {
        "both_converged": bool(first.metadata["converged"] and second.metadata["converged"]),
        "common_valid_nodes": int(np.count_nonzero(valid)),
        "interpolation": False,
        "max_normalized_potential_difference": float(np.max(np.abs(du[valid]))),
        "max_normalized_field_vector_difference": float(np.max(np.hypot(dex[valid], dey[valid]))),
        "note": "Same dimensionless finite frame and grid; agreement does not establish accuracy.",
    }


def _case_label(label, summary):
    convergence = "" if summary["all_converged"] else " [NOT CONVERGED]"
    return f"L/d={summary['length_over_gap']:g}{convergence}"


def save_profiles(output, summaries, cases):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, layout="constrained")
    colors = ("#1663a5", "#cc5634", "#27815a")
    for ax, (scan, labels) in zip(axes, SCANS.items()):
        for label, color in zip(labels, colors):
            case, summary = cases[label]["fine"], summaries[label]
            a = case.arrays
            gap = summary["gap"]
            y0 = len(a["y"]) // 2
            x_over_d = a["x"] / gap
            selected = np.abs(x_over_d) <= 5 + 1e-9
            reference = abs(float(case.config["boundary_conditions"]["V0"])) / gap
            profile = np.ma.array(-a["electric_field_y"][y0] / reference,
                                  mask=~a["analysis_valid_mask"][y0])
            ax.plot(x_over_d[selected], profile[selected], color=color, lw=1.8,
                    label=_case_label(label, summary))
            for endpoint in (-summary["length_over_gap"] / 2, summary["length_over_gap"] / 2):
                ax.axvline(endpoint, color=color, lw=0.7, ls=":", alpha=0.65)
        ax.axhline(1, color="#555555", lw=0.8, ls="--", alpha=0.75)
        ax.set(xlim=(-5, 5), xlabel=r"$x/d$", ylabel=r"$-E_y/E_{ref}$ at $y=0$",
               title="Vary L, fixed d=1" if scan == "plate_length" else "Vary d, fixed L=4")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("Midline fields — descriptive; mesh/domain limitations apply", fontsize=13)
    fig.text(0.5, 0.01, "Dotted verticals: plate endpoints; dashed horizontal: uniform-field reference.",
             ha="center", fontsize=9)
    fig.savefig(output / "midline-comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_maps(output, summaries, cases, *, field):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 5), layout="constrained")
    artist = None
    for ax, (label, length, gap) in zip(axes.flat, GEOMETRIES):
        case, summary = cases[label]["fine"], summaries[label]
        a = case.arrays
        x, y = a["x"] / gap, a["y"] / gap
        ix = np.flatnonzero(np.abs(x) <= 5.05)
        iy = np.flatnonzero(np.abs(y) <= 1.55)
        region = np.ix_(iy, ix)
        if field:
            reference = abs(float(case.config["boundary_conditions"]["V0"])) / gap
            values = np.hypot(a["electric_field_x"][region], a["electric_field_y"][region]) / reference
            values = np.ma.array(values, mask=~a["analysis_valid_mask"][region])
            artist = ax.pcolormesh(x[ix], y[iy], values, shading="auto", cmap="viridis", vmin=0, vmax=2)
        else:
            values = a["potential"][region] / abs(float(case.config["boundary_conditions"]["V0"]))
            artist = ax.pcolormesh(x[ix], y[iy], values, shading="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        for yplate in (-0.5, 0.5):
            ax.plot([-length / (2 * gap), length / (2 * gap)], [yplate, yplate],
                    color="white", lw=3.5, zorder=4)
            ax.plot([-length / (2 * gap), length / (2 * gap)], [yplate, yplate],
                    color="black", lw=1.5, zorder=5)
        suffix = "" if summary["all_converged"] else " — NOT CONVERGED"
        ax.set(title=f"L={length:g}, d={gap:g}, L/d={length/gap:g}{suffix}",
               xlabel=r"$x/d$", ylabel=r"$y/d$", xlim=(-5, 5), ylim=(-1.5, 1.5))
        ax.set_aspect("equal", adjustable="box")
    axes.flat[-1].axis("off")
    if field:
        label, title, filename = (r"$|E|/E_{ref}$ (clipped at 2)",
                                  "Field magnitude — endpoint disks excluded; limitations apply",
                                  "field-comparison.png")
    else:
        label, title, filename = (r"$\varphi/|V_0|$", "Potential — descriptive; limitations apply",
                                  "potential-comparison.png")
    fig.colorbar(artist, ax=axes.ravel().tolist(), label=label, shrink=0.8,
                 extend="max" if field else "neither")
    fig.suptitle(title, fontsize=13)
    fig.savefig(output / filename, dpi=160)
    plt.close(fig)


def write_table(output, summaries):
    columns = ("geometry", "L", "d", "L_over_d", "h_over_d", "a_over_d", "b_over_d",
               "tol", "Nmax", "all_converged", "center_negative_ey_over_reference",
               "central_max_vector_deviation_over_reference", "outside_negative_ey_over_reference",
               "mesh_01_005_field_difference", "mesh_005_0025_field_difference",
               "domain_medium_field_difference", "mesh_fine_center_change", "mesh_fine_outside_change",
               "domain_medium_center_change", "domain_medium_outside_change", "mesh_01_005_within_scale",
               "mesh_005_0025_within_scale", "domain_medium_within_scale", "reliability_status")
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for label, summary in summaries.items():
            metrics, comparisons = summary["metrics"], summary["comparisons"]
            changes = summary["point_metric_absolute_changes"]
            config = summary["runs"]["fine"]["config"]
            writer.writerow({
                "geometry": label, "L": summary["plate_length"], "d": summary["gap"],
                "L_over_d": summary["length_over_gap"],
                "h_over_d": config["grid"]["h"] / summary["gap"],
                "a_over_d": config["geometry"]["domain_half_width"] / summary["gap"],
                "b_over_d": config["geometry"]["domain_half_height"] / summary["gap"],
                "tol": config["solver"]["tol"],
                "Nmax": config["solver"]["Nmax"], "all_converged": summary["all_converged"],
                **{key: metrics[key] for key in ("center_negative_ey_over_reference",
                    "central_max_vector_deviation_over_reference", "outside_negative_ey_over_reference")},
                "mesh_01_005_field_difference": comparisons["mesh_coarse_medium"]["max_field_vector_difference_over_eref"],
                "mesh_005_0025_field_difference": comparisons["mesh_medium_fine"]["max_field_vector_difference_over_eref"],
                "domain_medium_field_difference": comparisons["domain_medium"]["max_field_vector_difference_over_eref"],
                "mesh_fine_center_change": changes["mesh_medium_fine"]["center_negative_ey_over_reference"],
                "mesh_fine_outside_change": changes["mesh_medium_fine"]["outside_negative_ey_over_reference"],
                "domain_medium_center_change": changes["domain_medium"]["center_negative_ey_over_reference"],
                "domain_medium_outside_change": changes["domain_medium"]["outside_negative_ey_over_reference"],
                "mesh_01_005_within_scale": comparisons["mesh_coarse_medium"]["within_suggested_scale"],
                "mesh_005_0025_within_scale": comparisons["mesh_medium_fine"]["within_suggested_scale"],
                "domain_medium_within_scale": comparisons["domain_medium"]["within_suggested_scale"],
                "reliability_status": summary["reliability_status"],
            })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/m2-scan")
    parser.add_argument("--nmax", type=int, default=2_000_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backend", choices=("numpy", "numba"), default="numpy")
    parser.add_argument("--seed-s4-runs", type=Path,
                        help="optional S4 runs directory; only matching baseline results are copied")
    args = parser.parse_args(argv)
    if args.nmax <= 0:
        parser.error("--nmax must be positive")
    output = args.output_dir
    if output.exists() and not args.resume:
        parser.error(f"{output} already exists; use --resume or choose another directory")
    output.mkdir(parents=True, exist_ok=True)
    base = load_config(ROOT / "configs/parallel_plate.json")
    provenance = {"python": platform.python_version(), "numpy": np.__version__,
                  "machine": platform.machine(), "backend": args.backend, "source_sha256": source_hashes()}
    if args.backend == "numba":
        try:
            import numba
        except ImportError as exc:
            raise ValueError("Numba backend requires uv sync --extra s4") from exc
        provenance["numba"] = numba.__version__
    cases, summaries = {}, {}
    for label, length, gap in GEOMETRIES:
        cases[label] = {}
        for run in RUNS:
            config = configuration(base, length, gap, run, args.nmax)
            run_label = f"{label}-{run}"
            destination = output / "runs" / run_label
            if args.seed_s4_runs and label == "L4-d1" and run in S4_BASELINE_SEEDS and not destination.exists():
                copy_seed(args.seed_s4_runs / S4_BASELINE_SEEDS[run], destination, config, provenance)
                print(f"SEEDED {run_label} from verified S4 baseline", flush=True)
            cases[label][run] = solve_record(run_label, config, output, provenance,
                                             resume=args.resume or destination.exists(), backend=args.backend)
        summary = analyze_geometry(length, gap, cases[label])
        summary["runs"] = {
            run: {**case.metadata, "path": f"runs/{label}-{run}/", "algebraic_error_bounds": algebraic_bounds(case)}
            for run, case in cases[label].items()
        }
        summaries[label] = summary
        print(f"CHECK {label}: status={summary['reliability_status']}; "
              + ", ".join(f"{key}={v['max_field_vector_difference_over_eref']:.6g}"
                          for key, v in summary['comparisons'].items()), flush=True)
    write_table(output, summaries)
    save_profiles(output, summaries, cases)
    save_maps(output, summaries, cases, field=False)
    save_maps(output, summaries, cases, field=True)
    scripts = (Path(__file__), ROOT / "scripts/validate_m2.py")
    similarity_pairs = (("L2-d1", "L4-d2"), ("L8-d1", "L4-d05"))
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "2D zero-thickness plates and a finite zero-potential outer frame",
        "environment": {**provenance, "platform": platform.platform(), "matplotlib": matplotlib.__version__},
        "script_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in scripts},
        "scans": SCANS,
        "normalization": {"coordinates": "x/d, y/d", "potential": "phi/abs(V0)", "field": "E/(abs(V0)/d)"},
        "run_design": RUNS,
        "comparison_limits": {"potential_and_field_difference": DOMAIN_MESH_TARGET,
                              "combined_algebraic_bound": ITERATION_TARGET},
        "interpretation": (
            "Comparison scales are not total-error bounds; descriptive metrics are not analytic errors. "
            "A single intermediate-grid frame expansion does not establish S4 domain acceptance. "
            "Endpoint disks of physical radius 0.2d are excluded from field comparisons. "
            "The central maximum uses a region whose width changes with L."
        ),
        "geometries": summaries,
        "scaled_geometry_checks": {
            f"{first}_vs_{second}": compare_scaled_geometries(cases[first]["fine"], cases[second]["fine"])
            for first, second in similarity_pairs if first in cases and second in cases
        },
        "output_files": ["metrics.csv", "midline-comparison.png", "potential-comparison.png", "field-comparison.png"],
        "all_pairwise_checks_within_suggested_scale": all(
            summary["reliability_status"] == "provisional" for summary in summaries.values()),
    }
    write_json(output / "report.json", report)
    limited = sum(s["reliability_status"] == "limited" for s in summaries.values())
    print(f"S5 outputs: {output}; {limited}/{len(summaries)} geometries have limited reliability", flush=True)
    # As in S4, a failed scale or convergence check returns 1 while retaining diagnostics.
    return 1 if limited else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"M2 scan failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
