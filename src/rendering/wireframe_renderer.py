"""Renders a 3D wireframe onto an OpenCV frame.

The whole edge table is drawn in a single ``cv2.polylines`` call rather than
one call per edge: the ``(n, 2)`` edge table is reshaped to ``(-1, 1, 2)`` so
that OpenCV treats each consecutive pair as an independent polyline. That
keeps the per-frame cost constant regardless of edge count.

When a depth buffer is supplied the renderer switches to a "Z-Binned Double-
Stroke" holographic look: every edge is drawn once as a thick outer glow and
once more from its depth bin (Far / Mid / Near) with a brighter cyan, so the
model reads as volumetric with only four draw calls.
"""

from __future__ import annotations

from typing import Optional, Sequence

import logging

import cv2
import numpy as np

__all__ = [
    "WIREFRAME_COLOR",
    "WIREFRAME_THICKNESS",
    "HOLO_GLOW_OUTER",
    "HOLO_FAR",
    "HOLO_MID",
    "HOLO_NEAR",
    "GLOW_THICKNESS",
    "CORE_THICKNESS",
    "WireframeRenderer",
]

logger = logging.getLogger("HoloForge")

# Plain wireframe (Day 5)
WIREFRAME_COLOR = (0, 255, 0)  # BGR green
WIREFRAME_THICKNESS = 2

# Holographic look
HOLO_GLOW_OUTER = (80, 40, 0)     # dark cyan BGR, thick outer stroke
HOLO_FAR = (150, 100, 0)         # dim cyan, edges farthest from the camera
HOLO_MID = (220, 180, 0)         # medium cyan
HOLO_NEAR = (255, 255, 0)        # bright cyan, edges nearest the camera
GLOW_THICKNESS = 3               # outer glow stroke, pixels
CORE_THICKNESS = 2               # per-bin core stroke, pixels
VERTEX_RADIUS = 3                # pixels; 0 disables vertex dots
VERTEX_COLOR = (0, 255, 255)     # BGR yellow

# Depth bins are computed from the frame's own depth distribution, so they
# track the model as it moves rather than being hardcoded to one pose.


class WireframeRenderer:
    """Draws a wireframe defined by 2D points and an edge index table.

    Parameters
    ----------
    color : tuple[int, int, int], default ``WIREFRAME_COLOR``
        BGR edge colour for the plain (non-holographic) pass.
    thickness : int, default ``WIREFRAME_THICKNESS``
        Edge stroke width in pixels.
    vertex_radius : int, default ``VERTEX_RADIUS``
        Radius of the per-vertex dots; ``0`` disables them.
    vertex_color : tuple[int, int, int], default ``VERTEX_COLOR``
        BGR vertex dot colour.
    holographic : bool, default ``True``
        When ``depth`` is supplied, apply the Z-Binned Double-Stroke look
        instead of the single-colour wireframe.
    """

    def __init__(
        self,
        color: tuple[int, int, int] = WIREFRAME_COLOR,
        thickness: int = WIREFRAME_THICKNESS,
        vertex_radius: int = VERTEX_RADIUS,
        vertex_color: tuple[int, int, int] = VERTEX_COLOR,
        holographic: bool = True,
    ) -> None:
        self.color = color
        self.thickness = thickness
        self.vertex_radius = vertex_radius
        self.vertex_color = vertex_color
        self.holographic = holographic

    def draw(
        self,
        frame: np.ndarray,
        points2d: np.ndarray,
        edges: np.ndarray,
        depth: Optional[np.ndarray] = None,
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
        depth : np.ndarray | None
            ``(n,)`` transformed Z-depth of each vertex, or ``None`` to draw a
            plain single-colour wireframe.

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

        if depth is None or not self.holographic:
            cv2.polylines(
                drawing,
                polylines,
                isClosed=False,
                color=self.color,
                thickness=self.thickness,
                lineType=cv2.LINE_AA,
            )
            self._draw_vertex_dots(drawing, points2d, np.arange(points2d.shape[0]))
            return drawing

        depth = np.asarray(depth, dtype=np.float64)
        if depth.ndim != 1 or depth.shape[0] != points2d.shape[0]:
            logger.warning(
                "Depth buffer shape %r does not match %d vertices; "
                "falling back to the plain wireframe",
                depth.shape,
                points2d.shape[0],
            )
            cv2.polylines(
                drawing,
                polylines,
                isClosed=False,
                color=self.color,
                thickness=self.thickness,
                lineType=cv2.LINE_AA,
            )
            self._draw_vertex_dots(drawing, points2d, np.arange(points2d.shape[0]))
            return drawing

        # Each edge's depth is the mean of its two endpoints, then binned by
        # percentile so the split tracks the model as it rotates.
        edge_depth = (depth[edges[:, 0]] + depth[edges[:, 1]]) * 0.5
        if edge_depth.size < 2:
            cv2.polylines(
                drawing,
                polylines,
                isClosed=False,
                color=self.color,
                thickness=self.thickness,
                lineType=cv2.LINE_AA,
            )
            self._draw_vertex_dots(drawing, points2d, np.arange(points2d.shape[0]))
            return drawing

        far_bound, near_bound = np.percentile(edge_depth, [33.3, 66.7])

        # Pass 1: outer glow, every edge in one call.
        cv2.polylines(
            drawing,
            polylines,
            isClosed=False,
            color=HOLO_GLOW_OUTER,
            thickness=GLOW_THICKNESS,
            lineType=cv2.LINE_AA,
        )

        # Pass 2: depth-cued core, one call per bin.
        for mask, color in (
            (edge_depth < far_bound, HOLO_FAR),
            ((edge_depth >= far_bound) & (edge_depth < near_bound), HOLO_MID),
            (edge_depth >= near_bound, HOLO_NEAR),
        ):
            if not np.any(mask):
                continue
            bin_polylines = points2d[edges[mask]].reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(
                drawing,
                bin_polylines,
                isClosed=False,
                color=color,
                thickness=CORE_THICKNESS,
                lineType=cv2.LINE_AA,
            )

        # Vertex dots only for the nearest bin, to keep the hologram readable.
        near_indices = edges[(edge_depth >= near_bound).nonzero()[0]].ravel()
        if near_indices.size > 0:
            self._draw_vertex_dots(drawing, points2d, np.unique(near_indices))

        return drawing

    def _draw_vertex_dots(
        self, drawing: np.ndarray, points2d: np.ndarray, indices: np.ndarray
    ) -> None:
        """Draw filled vertex circles at ``indices``.

        ``cv2.circle`` has no batch form, but vertices are drawn only for the
        nearest depth bin so the loop is far smaller than the edge count.

        Parameters
        ----------
        drawing : np.ndarray
            Image to annotate.
        points2d : np.ndarray
            ``(n, 2)`` screen-space vertex coordinates.
        indices : np.ndarray
            Vertex indices to draw.
        """
        if self.vertex_radius <= 0 or indices.size == 0:
            return

        for index in indices:
            cv2.circle(
                drawing,
                (int(points2d[index, 0]), int(points2d[index, 1])),
                self.vertex_radius,
                self.vertex_color,
                -1,
            )