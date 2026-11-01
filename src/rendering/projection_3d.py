"""Perspective projection and cube geometry for the 3D wireframe engine.

Vertices are rotated by a composition of axis-aligned rotations, scaled, and
then projected through a pinhole with a fixed focal length. The projection
divides by depth, so geometry that sits nearer to the camera appears larger.
``Z_OFFSET`` pushes the model out past the camera plane to keep the divisor
away from the singularity at ``z == 0``, and ``MIN_Z`` floors the divisor so
that a vertex can never drive the division to infinity.

The rotation order ``Rz @ Ry @ Rx`` is applied to the vertex in a right-handed
frame: pitch (x), then yaw (y), then roll (z).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "FOCAL_LENGTH",
    "MIN_Z",
    "Z_OFFSET",
    "make_cube",
    "project_points",
    "rotation_matrix",
]

# Pinhole parameters
FOCAL_LENGTH = 2000.0  # focal length in pixels; larger => narrower field of view
Z_OFFSET = 1500.0  # depth offset pushing the model out from the camera plane
MIN_Z = 1.0  # floor on the projected depth to avoid division by zero


def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Build a 3x3 rotation matrix as ``Rz @ Ry @ Rx``.

    Each angle is in radians and the matrices are the standard right-handed
    rotations about the x, y and z axes. The composition means a vertex is
    pitched first, then yawed, then rolled.

    Parameters
    ----------
    rx : float
        Rotation about the x axis, radians.
    ry : float
        Rotation about the y axis, radians.
    rz : float
        Rotation about the z axis, radians.

    Returns
    -------
    np.ndarray
        ``(3, 3)`` rotation matrix.
    """
    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)

    mat_x = np.array([[1.0, 0.0, 0.0], [0.0, cos_x, -sin_x], [0.0, sin_x, cos_x]])
    mat_y = np.array([[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]])
    mat_z = np.array([[cos_z, -sin_z, 0.0], [sin_z, cos_z, 0.0], [0.0, 0.0, 1.0]])

    return mat_z @ mat_y @ mat_x


def project_points(
    vertices: np.ndarray,
    rot: np.ndarray,
    scale: float,
    offset: Tuple[float, float],
    frame_shape: Tuple[int, ...],
    transformed: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project 3D vertices into 2D screen coordinates.

    Each vertex is rotated, scaled and then divided by its depth, which is the
    perspective step. The resulting coordinates are biased by the frame centre
    so that a zero offset and zero rotation place the model at the centre of
    the viewport.

    Parameters
    ----------
    vertices : np.ndarray
        ``(n, 3)`` array of model-space vertices.
    rot : np.ndarray
        ``(3, 3)`` rotation matrix, typically from :func:`rotation_matrix`.
    scale : float
        Uniform scale applied after rotation.
    offset : tuple[float, float]
        ``(x, y)`` translation applied before the centre bias, in pixels.
    frame_shape : tuple[int, ...]
        Frame dimensions ``(height, width, ...)``; only the first two axes
        are read.
    transformed : np.ndarray | None
        Pre-rotated and scaled vertices ``(n, 3)``. When supplied the rotation
        and scaling step is skipped so the caller can reuse the same array,
        for example to derive a depth buffer for the renderer.

    Returns
    -------
    np.ndarray
        ``(n, 2)`` array of screen-space ``(x, y)`` coordinates.
    """
    height, width = frame_shape[:2]

    if transformed is None:
        transformed = (vertices @ rot.T) * scale
    depth = transformed[:, 2] + Z_OFFSET
    depth = np.where(depth < MIN_Z, MIN_Z, depth)

    factor = FOCAL_LENGTH / depth
    x = transformed[:, 0] * factor + offset[0] + width / 2.0
    y = transformed[:, 1] * factor + offset[1] + height / 2.0

    return np.stack([x, y], axis=-1)


def make_cube(size: float = 150.0) -> Tuple[np.ndarray, np.ndarray]:
    """Build a centred cube and its canonical edge table.

    The cube is centred on the origin with sides aligned to the axes, so a
    zero rotation and zero offset project it at the centre of the viewport.

    Parameters
    ----------
    size : float, default ``150.0``
        Edge length of the cube in model units.

    Returns
    -------
    vertices : np.ndarray
        ``(8, 3)`` array of vertex positions.
    edges : np.ndarray
        ``(12, 2)`` array of vertex index pairs defining the wireframe edges.
    """
    half = size / 2.0
    vertices = np.array(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ],
        dtype=np.float64,
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # back face
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # front face
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # struts
        ],
        dtype=np.intp,
    )
    return vertices, edges