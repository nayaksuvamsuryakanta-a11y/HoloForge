"""Shared infrastructure helpers used by every HoloForge package."""

from __future__ import annotations

import logging
from pathlib import Path

__all__ = ["ensure_directories", "get_project_root", "setup_logging"]


def get_project_root() -> Path:
    """Return the repository root, independent of the current working directory.

    Walks upward from this module's own location, so paths resolve correctly
    whether the project is launched from the repo root or a subdirectory.

    Returns
    -------
    Path
        Absolute path to the workspace root that contains ``src/``.
    """
    return Path(__file__).resolve().parent.parent.parent


def setup_logging(
    level: int = logging.INFO,
    fmt: str = "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
) -> None:
    """Configure the root logger for the HoloForge application.

    Parameters
    ----------
    level : int, default ``logging.INFO``
        Minimum severity level to emit.
    fmt : str
        ``logging.LogRecord`` format string.
    """
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger("HoloForge").setLevel(level)


def ensure_directories(*paths: str | Path) -> list[Path]:
    """Create every directory in ``paths``, including any missing parents.

    Parameters
    ----------
    *paths : str | Path
        One or more directory paths to materialize.

    Returns
    -------
    list[Path]
        Absolute paths of the directories, in the order requested.
    """
    created: list[Path] = []
    for path in paths:
        resolved = Path(path).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        created.append(resolved)
    return created