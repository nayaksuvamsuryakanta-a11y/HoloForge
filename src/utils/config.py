"""Validated runtime tuning with CLI > JSON > built-in precedence."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

logger = logging.getLogger("HoloForge")
DEFAULTS = {
    "grab_threshold": 0.35,
    "release_threshold": 0.50,
    "smoothing_alpha": 0.20,
    "rotation_sensitivity": 3.0,
    "pan_sensitivity": 1.0,
}


def _valid(key: str, value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not math.isfinite(value):
        return False
    if key in ("grab_threshold", "release_threshold"):
        return 0.0 <= value <= 1.0
    if key == "smoothing_alpha":
        return 0.0 < value <= 1.0
    return value > 0.0


def load_config(path: str | Path = "config.json", cli_overrides: dict | None = None) -> dict:
    values = dict(DEFAULTS)
    config_path = Path(path)
    if config_path.is_file():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("root must be an object")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Invalid config.json: %s; using defaults", exc)
            raw = {}
        for key, value in raw.items():
            if key not in DEFAULTS:
                logger.warning("Invalid config key %s; ignoring it", key)
            elif _valid(key, value):
                values[key] = float(value)
            else:
                logger.warning("Invalid config key %s; using default", key)
    if values["grab_threshold"] >= values["release_threshold"]:
        logger.warning("Invalid config key grab_threshold; using default")
        logger.warning("Invalid config key release_threshold; using default")
        values["grab_threshold"] = DEFAULTS["grab_threshold"]
        values["release_threshold"] = DEFAULTS["release_threshold"]
    for key, value in (cli_overrides or {}).items():
        if value is None:
            continue
        if key in values and _valid(key, value):
            values[key] = float(value)
        else:
            raise ValueError(f"Invalid command-line value for {key}")
    if values["grab_threshold"] >= values["release_threshold"]:
        raise ValueError("grab_threshold must be less than release_threshold")
    return values


def write_config(values: dict, path: str | Path = "config.json") -> Path:
    target = Path(path)
    target.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    return target
