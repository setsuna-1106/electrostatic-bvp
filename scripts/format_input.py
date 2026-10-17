"""Collect MVP parameters and write a validated JSON configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from electrostatic_bvp.config import ConfigError, save_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a validated rectangle_sine JSON configuration."
    )
    parser.add_argument("--output", type=Path, default=Path("configs/rectangle.json"))
    parser.add_argument("--lx", type=float, default=1.5, help="rectangle width")
    parser.add_argument("--ly", type=float, default=1.0, help="rectangle height")
    parser.add_argument("--v0", type=float, default=1.0, help="top-boundary amplitude")
    parser.add_argument("--h", type=float, default=0.05, help="square-grid spacing")
    parser.add_argument("--tol", type=float, default=1e-8, help="Jacobi stopping threshold")
    parser.add_argument("--nmax", type=int, default=100_000, help="maximum iterations")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = {
        "case": "rectangle_sine",
        "geometry": {"Lx": args.lx, "Ly": args.ly},
        "boundary_conditions": {"V0": args.v0},
        "grid": {"h": args.h},
        "solver": {"method": "jacobi", "tol": args.tol, "Nmax": args.nmax},
    }

    try:
        path = save_config(config, args.output)
    except (ConfigError, OSError) as exc:
        print(f"配置未写入：{exc}")
        return 2

    print(f"已写入配置：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

