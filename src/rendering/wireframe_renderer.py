"""Renders a 3D wireframe onto an OpenCV frame.

The whole edge table is drawn in a single ``cv2.polylines`` call rather than
one call per edge: the ``(n, 2)`` edge table is reshaped to ``(-1, 1, 2)`` so
that OpenCV treats each consecutive pair as an independent polyline. That
keeps the per-frame cost constant regardless of edge count.

When a depth buffer is supplied the renderer switches to a "Z-Binned Double-
Stroke" holographic look: every edge is drawn once as a thick outer glow and
once more from its depth bin (Far / Mid / Near) with a brighter cyan, so the
model reads as volumetric with only four draw calls.

A render-only Level-of-Detail ladder sits on top of that. High-density scans
project hundreds of edges onto a handful of pixels and overlap until the
wireframe reads as a filled disc, so the edge table is down-sampled when the
mean projected edge length drops. The ladder is built once from the prepared
edges and reused every frame, so it costs no allocation in the hot path.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

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
    "VERTEX_RADIUS",
    "VERTEX_COLOR",
    "LOD_TARGETS",
    "LOD_UPGRADE_PX",
    "LOD_DOWNGRADE_PX",
    "stroke_rules",
    "WireframeRenderer",
]

logger = logging.getLogger("HoloForge")

# Plain wireframe (Day 5)
WIREFRAME_COLOR = (0, 255, 0)  # BGR green
WIREFRAME_THICKNESS = 2

# Holographic look -- every constant is BGR cyan: B >= G > R == 0.
HOLO_GLOW_OUTER = (100, 70, 0)     # dark cyan BGR, thick outer stroke
HOLO_FAR = (150, 120, 0)           # dim cyan, edges farthest from the camera
HOLO_MID = (200, 170, 0)           # medium cyan
HOLO_NEAR = (255, 255, 0)          # bright cyan, edges nearest the camera
GLOW_THICKNESS = 3                 # outer glow stroke, pixels
CORE_THICKNESS = 2                 # per-bin core stroke, pixels
VERTEX_RADIUS = 3                  # pixels; 0 disables vertex dots
VERTEX_COLOR = (0, 255, 255)       # BGR yellow

# Level-of-Detail ladder. Levels are edge counts, descending; each level is a
# strict subset of the one above it so the silhouette stays recognisable as it
# thins. Built once from the prepared edges and cached for the run.
LOD_TARGETS = (5000, 2500, 1200, 600)
LOD_UPGRADE_PX = 8.0   # mean projected length above which detail is added
LOD_DOWNGRADE_PX = 4.0 # mean projected length below which detail is dropped
# The stroke rules pivot on these edge counts.
LOD_GLOW_LIMIT = 1200  # outer glow drawn only at or below this many edges
LOD_VERTEX_LIMIT = 600 # vertex dots drawn only at or below this many edges


def stroke_rules(edge_count: int) -> Tuple[bool, int, bool]:
    """Return whether glow/dots are enabled and the core stroke width."""
    return (
        edge_count <= LOD_GLOW_LIMIT,
        1 if edge_count > LOD_GLOW_LIMIT else 2,
        edge_count <= LOD_VERTEX_LIMIT,
    )

# Depth bins are computed from the frame's own depth distribution, so they
# track the model as it moves rather than being hardcoded to one pose.


def build_ladder(edges: np.ndarray) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
    """Deterministic nested subsets of ``edges``, plus their index tables.

    A fixed seed makes the selection independent of allocation order, so the
    same input always yields the same ladder. Levels that would not be a
    strict subset of the previous one are dropped, which matters for small
    meshes that cannot fill every rung.

    Parameters
    ----------
    edges : np.ndarray
        ``(m, 2)`` prepared edge index table.

    Returns
    -------
    index_tables : list[np.ndarray]
        One ``(k, 2)`` index array per level, each a strict subset of the
        previous.
    edge_tables : list[np.ndarray]
        The corresponding subsets of ``edges``, ready to index without
        per-frame allocation.
    """
    rng = np.random.default_rng(42)
    order = rng.permutation(edges.shape[0])

    index_tables: list[np.ndarray] = []
    edge_tables: list[np.ndarray] = []
    previous = edges.shape[0] + 1
    for target in LOD_TARGETS:
        count = min(target, edges.shape[0])
        if count >= previous:
            continue
        indices = order[:count]
        index_tables.append(indices)
        edge_tables.append(edges[indices])
        previous = count
    return index_tables, edge_tables


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

        # LOD state; the ladder is built from the first edge table seen.
        self._ladder_edges: Optional[Sequence[np.ndarray]] = None
        self._ladder_indices: Optional[Sequence[np.ndarray]] = None
        self._lod_index = 0
        # Exposed for the HUD; 0 until the first frame has been rendered.
        self.active_edge_count = 0
        self.mean_edge_length = 0.0

    def prepare_lod(self, edges: np.ndarray) -> None:
        """Precompute deterministic nested edge indices and tables once."""
        self._ladder_indices, self._ladder_edges = build_ladder(edges)
        self._lod_index = 0

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
            self.active_edge_count = 0
            self.mean_edge_length = 0.0
            return frame.copy()

        drawing = frame.copy()

        # Level of detail: pick a subset of the edges based on how many pixels
        # they span, with hysteresis so the level does not chatter at the
        # threshold. The metric is computed over the full prepared table so the
        # choice is independent of the level currently drawn, and the ladder is
        # built once from that same table.
        active_edges, active_count, mean_length = self._select_lod_level(points2d, edges)
        self.active_edge_count = active_count
        self.mean_edge_length = mean_length

        # Keep each edge as a separate two-point contour. Flattening to
        # (-1, 1, 2) gives OpenCV one-point contours, so no line is drawn.
        polylines = points2d[active_edges].reshape(-1, 2, 1, 2).astype(np.int32)

        if depth is None or not self.holographic:
            cv2.polylines(
                drawing,
                polylines,
                isClosed=False,
                color=self.color,
                thickness=self.thickness,
                lineType=cv2.LINE_AA,
            )
            if stroke_rules(active_count)[2]:
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
            if stroke_rules(active_count)[2]:
                self._draw_vertex_dots(drawing, points2d, np.arange(points2d.shape[0]))
            return drawing

        # Each edge's depth is the mean of its two endpoints, then binned by
        # percentile so the split tracks the model as it rotates. The binning
        # is applied to the active edges only, so depth cues do not outlast the
        # geometry they describe.
        edge_depth = (depth[active_edges[:, 0]] + depth[active_edges[:, 1]]) * 0.5
        if edge_depth.size < 2:
            cv2.polylines(
                drawing,
                polylines,
                isClosed=False,
                color=self.color,
                thickness=self.thickness,
                lineType=cv2.LINE_AA,
            )
            if stroke_rules(active_count)[2]:
                self._draw_vertex_dots(drawing, points2d, np.arange(points2d.shape[0]))
            return drawing

        far_bound, near_bound = np.percentile(edge_depth, [33.3, 66.7])

        # Pass 1: outer glow, only when the silhouette is sparse enough that
        # extra strokes do not saturate into a solid blob.
        glow_enabled, core_thickness, dots_enabled = stroke_rules(active_count)
        if glow_enabled:
            cv2.polylines(
                drawing,
                polylines,
                isClosed=False,
                color=HOLO_GLOW_OUTER,
                thickness=GLOW_THICKNESS,
                lineType=cv2.LINE_AA,
            )

        # Pass 2: depth-cued core, one call per bin. Thinner strokes at high
        # density, thicker once the edges have been thinned.
        for mask, color in (
            (edge_depth < far_bound, HOLO_FAR),
            ((edge_depth >= far_bound) & (edge_depth < near_bound), HOLO_MID),
            (edge_depth >= near_bound, HOLO_NEAR),
        ):
            if not np.any(mask):
                continue
            bin_polylines = points2d[active_edges[mask]].reshape(-1, 2, 1, 2).astype(np.int32)
            cv2.polylines(
                drawing,
                bin_polylines,
                isClosed=False,
                color=color,
                thickness=core_thickness,
                lineType=cv2.LINE_AA,
            )

        # Vertex dots only for the nearest bin, and only at the lowest density.
        if dots_enabled:
            near_indices = active_edges[(edge_depth >= near_bound).nonzero()[0]].ravel()
            if near_indices.size > 0:
                self._draw_vertex_dots(drawing, points2d, np.unique(near_indices))

        return drawing

    def _select_lod_level(
        self, points2d: np.ndarray, edges: np.ndarray
    ) -> Tuple[np.ndarray, int, float]:
        """Pick the LOD level for ``edges`` and return its table, size and mean pixel length.

        The ladder is built once from the prepared edges and cached, so no
        allocation happens in the hot path. Each call may advance one rung up
        or down depending on the mean projected edge length; moving a single
        rung per frame is what keeps the level from flickering.

        Parameters
        ----------
        points2d : np.ndarray
            ``(n, 2)`` screen-space vertex coordinates.
        edges : np.ndarray
            ``(m, 2)`` prepared edge index table.

        Returns
        -------
        active_edges : np.ndarray
            ``(k, 2)`` edges for the selected level.
        active_count : int
            ``k``, the number of edges actually rendered this frame.
        mean_length : float
            Mean projected edge length in pixels, computed over the full
            table so the metric does not depend on the selected level.
        """
        if self._ladder_edges is None:
            self.prepare_lod(edges)

        mean_length = _mean_projected_length(points2d, edges)
        levels = len(self._ladder_edges)
        if mean_length > LOD_UPGRADE_PX and self._lod_index > 0:
            self._lod_index -= 1
        elif mean_length < LOD_DOWNGRADE_PX and self._lod_index < levels - 1:
            self._lod_index += 1

        active_edges = self._ladder_edges[self._lod_index]
        return active_edges, int(active_edges.shape[0]), float(mean_length)

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


def _mean_projected_length(points2d: np.ndarray, edges: np.ndarray) -> float:
    """Mean length in pixels of the projected edges."""
    if edges.size == 0:
        return 0.0
    vectors = points2d[edges[:, 1]] - points2d[edges[:, 0]]
    lengths = np.linalg.norm(vectors, axis=1)
    return float(lengths.mean()) if lengths.size else 0.0
