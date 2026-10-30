"""Entry point for the HoloForge command-line interface.

Bootstraps logging and prints the application banner followed by the
initialization phases queued for the current session.
"""

from __future__ import annotations

import logging

from src.utils.helpers import setup_logging

__all__ = ["main"]


def main() -> int:
    """Run the HoloForge bootstrap sequence.

    Returns
    -------
    int
        Exit code; zero when bootstrapping completes successfully.
    """
    setup_logging()
    logger = logging.getLogger("HoloForge")

    logger.info("HoloForge v0.1.0 - Initializing...")
    logger.info("[PENDING] Load hand-landmarker model and camera stream")
    logger.info("[PENDING] Initialize 3D projection and rendering surface")
    logger.info("[PENDING] Start gesture command analysis pipeline")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())