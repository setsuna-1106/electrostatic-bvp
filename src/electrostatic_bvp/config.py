"""Read, validate, and write the small JSON configuration used by the MVP."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a configuration is malformed or outside the MVP scope."""


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


def validate_config(config: Any) -> dict[str, Any]:
    """Validate and return an MVP rectangle-sine configuration."""
    root = _object_with_keys(
        config,
        "config",
        {"case", "geometry", "boundary_conditions", "grid", "solver"},
    )

    if root["case"] != "rectangle_sine":
        raise ConfigError("'case' must be 'rectangle_sine' for the current MVP")

    geometry = _object_with_keys(root["geometry"], "geometry", {"Lx", "Ly"})
    lx = _finite_number(geometry["Lx"], "geometry.Lx", positive=True)
    ly = _finite_number(geometry["Ly"], "geometry.Ly", positive=True)

    boundary = _object_with_keys(
        root["boundary_conditions"], "boundary_conditions", {"V0"}
    )
    _finite_number(boundary["V0"], "boundary_conditions.V0")

    grid = _object_with_keys(root["grid"], "grid", {"h"})
    h = _finite_number(grid["h"], "grid.h", positive=True)
    for length, name in ((lx, "geometry.Lx"), (ly, "geometry.Ly")):
        intervals = length / h
        if not math.isfinite(intervals) or intervals < 2 or not math.isclose(
            intervals, round(intervals), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ConfigError(
                f"'{name}/grid.h' must be an integer of at least 2"
            )

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
    """Load a JSON file and validate it against the current MVP configuration."""
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

