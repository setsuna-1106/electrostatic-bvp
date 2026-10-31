"""Read, validate, and write case-specific JSON configurations."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a configuration is malformed or outside the supported scope."""


def _object_with_keys(value: Any, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{name}' must be a JSON object")

    missing = keys - value.keys()
    extra = value.keys() - keys
    if missing:
        raise ConfigError(f"'{name}' is missing: {', '.join(sorted(missing))}")
    if extra:
        raise ConfigError(f"'{name}' has unsupported keys: {', '.join(sorted(extra))}")
    return value


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{name}' must be a number")

    try:
        number = float(value)
    except OverflowError as exc:
        raise ConfigError(f"'{name}' must be finite") from exc
    if not math.isfinite(number):
        raise ConfigError(f"'{name}' must be finite")
    if positive and number <= 0:
        raise ConfigError(f"'{name}' must be greater than zero")
    return number


def _aligned_intervals(length: float, h: float, name: str, minimum: int = 1) -> int:
    intervals = length / h
    if not math.isfinite(intervals) or not math.isclose(
        intervals, round(intervals), rel_tol=0.0, abs_tol=1e-9
    ) or round(intervals) < minimum:
        raise ConfigError(f"'{name}/grid.h' must be an integer of at least {minimum}")
    return round(intervals)


def validate_config(config: Any) -> dict[str, Any]:
    """Validate either supported case, retaining the original input values."""
    root = _object_with_keys(
        config,
        "config",
        {"case", "geometry", "boundary_conditions", "grid", "solver"},
    )

    case = root["case"]
    if case not in ("rectangle_sine", "parallel_plate"):
        raise ConfigError("'case' must be 'rectangle_sine' or 'parallel_plate'")

    grid = _object_with_keys(root["grid"], "grid", {"h"})
    h = _finite_number(grid["h"], "grid.h", positive=True)
    if case == "rectangle_sine":
        geometry = _object_with_keys(root["geometry"], "geometry", {"Lx", "Ly"})
        for key in ("Lx", "Ly"):
            length = _finite_number(geometry[key], f"geometry.{key}", positive=True)
            _aligned_intervals(length, h, f"geometry.{key}", minimum=2)
        boundary = _object_with_keys(root["boundary_conditions"], "boundary_conditions", {"V0"})
    else:
        geometry = _object_with_keys(
            root["geometry"], "geometry",
            {"plate_length", "gap", "domain_half_width", "domain_half_height"},
        )
        lengths = {
            key: _finite_number(value, f"geometry.{key}", positive=True)
            for key, value in geometry.items()
        }
        a = _aligned_intervals(lengths["domain_half_width"], h, "geometry.domain_half_width")
        b = _aligned_intervals(lengths["domain_half_height"], h, "geometry.domain_half_height")
        half_plate = _aligned_intervals(lengths["plate_length"] / 2, h, "geometry.plate_length/2")
        half_gap = _aligned_intervals(lengths["gap"] / 2, h, "geometry.gap/2", minimum=2)
        if a - half_plate < 2 or b - half_gap < 2:
            raise ConfigError("plates must be at least two grid intervals inside the outer boundary")
        boundary = _object_with_keys(
            root["boundary_conditions"], "boundary_conditions", {"V0", "outer_potential"}
        )
        outer = _finite_number(boundary["outer_potential"], "boundary_conditions.outer_potential")
        if outer != 0:
            raise ConfigError("'boundary_conditions.outer_potential' must be zero")

    _finite_number(boundary["V0"], "boundary_conditions.V0")

    solver = _object_with_keys(
        root["solver"], "solver", {"method", "tol", "Nmax"}
    )
    if solver["method"] != "jacobi":
        raise ConfigError("'solver.method' must be 'jacobi' for the current MVP")
    _finite_number(solver["tol"], "solver.tol", positive=True)
    nmax = solver["Nmax"]
    if isinstance(nmax, bool) or not isinstance(nmax, int) or nmax <= 0:
        raise ConfigError("'solver.Nmax' must be a positive integer")

    return dict(root)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON file and validate its case configuration."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Invalid JSON in '{config_path}' at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc
    except OSError as exc:
        raise ConfigError(f"Could not read '{config_path}': {exc.strerror}") from exc

    return validate_config(config)


def save_config(config: Any, path: str | Path) -> Path:
    """Validate and write a configuration as formatted JSON."""
    validated = validate_config(config)
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(validated, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return config_path

