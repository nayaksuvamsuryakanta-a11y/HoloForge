"""Smoke tests for the Day 1 scaffold.

Asserts that every runtime dependency declared in ``requirements.txt`` is
importable and that the on-disk layout matches the tree in ``README.md``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent


def test_numpy_imports() -> None:
    assert np.__version__


def test_cv2_imports() -> None:
    assert cv2.__version__


def test_mediapipe_imports() -> None:
    assert mediapipe.__version__


def test_trimesh_imports() -> None:
    assert trimesh.__version__


def test_src_directory_exists() -> None:
    assert (ROOT / "src").is_dir()


def test_models_directory_exists() -> None:
    assert (ROOT / "models").is_dir()


def test_tests_directory_exists() -> None:
    assert (ROOT / "tests").is_dir()