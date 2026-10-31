"""Collect case parameters and write a validated JSON configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from electrostatic_bvp.config import ConfigError, save_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a validated case JSON configuration.")
    parser.add_argument("--case", choices=("rectangle_sine", "parallel_plate"), default="rectangle_sine")
    parser.add_argument("--output", type=Path, help="defaults to configs/<case>.json (rectangle.json for rectangle_sine)")
    parser.add_argument("--lx", type=float, help="rectangle width")
    parser.add_argument("--ly", type=float, help="rectangle height")
    parser.add_argument("--plate-length", type=float, help="finite plate length")
    parser.add_argument("--gap", type=float, help="plate separation")
    parser.add_argument("--domain-half-width", type=float, help="outer frame half-width")
    parser.add_argument("--domain-half-height", type=float, help="outer frame half-height")
    parser.add_argument("--outer-potential", type=float, help="outer frame potential (only 0 supported)")
    parser.add_argument("--v0", type=float, default=1.0, help="rectangle amplitude or plate voltage difference")
    parser.add_argument("--h", type=float, help="square-grid spacing")
    parser.add_argument("--tol", type=float, help="Jacobi stopping threshold")
    parser.add_argument("--nmax", type=int, default=100_000, help="maximum iterations")
    args = parser.parse_args()
    plate_args = (args.plate_length, args.gap, args.domain_half_width,
                  args.domain_half_height, args.outer_potential)
    if args.case == "rectangle_sine" and any(value is not None for value in plate_args):
        parser.error("plate geometry options require --case parallel_plate")
    if args.case == "parallel_plate" and (args.lx is not None or args.ly is not None):
        parser.error("--lx and --ly require --case rectangle_sine")
    return args


def default(value: float | None, fallback: float) -> float:
    return fallback if value is None else value


def main() -> int:
    args = parse_args()
    if args.case == "rectangle_sine":
        geometry = {"Lx": default(args.lx, 1.5), "Ly": default(args.ly, 1.0)}
        boundary = {"V0": args.v0}
        h, tol = default(args.h, 0.05), default(args.tol, 1e-8)
        output = args.output or Path("configs/rectangle.json")
    else:
        geometry = {
            "plate_length": default(args.plate_length, 4.0),
            "gap": default(args.gap, 1.0),
            "domain_half_width": default(args.domain_half_width, 6.0),
            "domain_half_height": default(args.domain_half_height, 4.0),
        }
        boundary = {"V0": args.v0, "outer_potential": default(args.outer_potential, 0.0)}
        h, tol = default(args.h, 0.1), default(args.tol, 1e-9)
        output = args.output or Path("configs/parallel_plate.json")
    config = {
        "case": args.case,
        "geometry": geometry,
        "boundary_conditions": boundary,
        "grid": {"h": h},
        "solver": {"method": "jacobi", "tol": tol, "Nmax": args.nmax},
    }

    try:
        path = save_config(config, output)
    except (ConfigError, OSError) as exc:
        print(f"配置未写入：{exc}")
        return 2

    print(f"已写入配置：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
