"""Renders a 3D wireframe onto an OpenCV frame.

The whole edge table is drawn in a single ``cv2.polylines`` call rather than
one call per edge: the ``(n, 2)`` edge table is reshaped to ``(-1, 1, 2)`` so
that OpenCV treats each consecutive pair as an independent polyline. That
keeps the per-frame cost constant regardless of edge count.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

__all__ = ["WIREFRAME_COLOR", "WIREFRAME_THICKNESS", "WireframeRenderer"]

WIREFRAME_COLOR = (0, 255, 0)  # BGR green
WIREFRAME_THICKNESS = 2
VERTEX_RADIUS = 3  # pixels; 0 disables vertex dots
VERTEX_COLOR = (0, 255, 255)  # BGR yellow


class WireframeRenderer:
    """Draws a wireframe defined by 2D points and an edge index table.

    Parameters
    ----------
    color : tuple[int, int, int], default ``WIREFRAME_COLOR``
        BGR edge colour.
    thickness : int, default ``WIREFRAME_THICKNESS``
        Edge stroke width in pixels.
    vertex_radius : int, default ``VERTEX_RADIUS``
        Radius of the per-vertex dots; ``0`` disables them.
    vertex_color : tuple[int, int, int], default ``VERTEX_COLOR``
        BGR vertex dot colour.
    """

    def __init__(
        self,
        color: tuple[int, int, int] = WIREFRAME_COLOR,
        thickness: int = WIREFRAME_THICKNESS,
        vertex_radius: int = VERTEX_RADIUS,
        vertex_color: tuple[int, int, int] = VERTEX_COLOR,
    ) -> None:
        self.color = color
        self.thickness = thickness
        self.vertex_radius = vertex_radius
        self.vertex_color = vertex_color

    def draw(
        self,
        frame: np.ndarray,
        points2d: np.ndarray,
        edges: np.ndarray,
    ) -> np.ndarray:
        """Render the wireframe onto a copy of ``frame``.

        Parameters
        ----------
        frame : np.ndarray
            ``(height, width, 3)`` BGR image to annotate.
        points2d : np.ndarray
            ``(n, 2)`` screen-space vertex coordinates.
        edges : np.ndarray
            ``(m, 2)`` pairs of vertex indices.

        Returns
        -------
        np.ndarray
            A new frame carrying the wireframe; the input is never modified.
        """
        if points2d.size == 0 or edges.size == 0:
            return frame.copy()

        drawing = frame.copy()

        # Reshape to (-1, 1, 2) so polylines draws each edge as its own
        # disconnected polyline instead of joining them into one chain.
        polylines = points2d[edges].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(
            drawing,
            polylines,
            isClosed=False,
            color=self.color,
            thickness=self.thickness,
            lineType=cv2.LINE_AA,
        )

        if self.vertex_radius > 0:
            vertices = points2d.astype(np.int32)
            for index in range(vertices.shape[0]):
                cv2.circle(
                    drawing,
                    (int(vertices[index, 0]), int(vertices[index, 1])),
                    self.vertex_radius,
                    self.vertex_color,
                    -1,
                )

        return drawing