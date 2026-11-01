"""Headless visibility checks through the renderer's production draw path."""

import numpy as np

from src.analysis.model_analyzer import load_and_prepare_mesh
from src.rendering.projection_3d import make_cube, project_points
from src.rendering.wireframe_renderer import WireframeRenderer


FRAME_SHAPE = (480, 640, 3)
BACKGROUNDS = (0, 128)


def _family_pixel_count(rendered: np.ndarray, background: int) -> int:
    """Count cyan-family signal pixels after subtracting the flat background."""
    delta = rendered.astype(np.int16) - int(background)
    blue, green, red = (delta[..., index] for index in range(3))
    return int(np.count_nonzero((blue >= green - 3) & (green > red + 2) & (blue > 8)))


def _render_at_lod(
    vertices: np.ndarray, edges: np.ndarray, background: int, lod_index: int
) -> int:
    frame = np.full(FRAME_SHAPE, background, dtype=np.uint8)
    renderer = WireframeRenderer(holographic=True)
    renderer.prepare_lod(edges)

    if vertices.shape[0] == 8:
        points = project_points(
            vertices, np.eye(3), 1.0, (0.0, 0.0), FRAME_SHAPE
        )
    else:
        # Keep the projected mean in the hysteresis band so the production
        # selector retains the explicitly selected rung during this draw.
        raw_lengths = np.linalg.norm(vertices[edges[:, 1], :2] - vertices[edges[:, 0], :2], axis=1)
        factor = 6.0 / float(raw_lengths.mean())
        points = (vertices[:, :2] * factor) + np.array([320.0, 240.0])

    renderer._lod_index = lod_index
    rendered = renderer.draw(frame, points, edges, depth=vertices[:, 2])
    assert renderer.active_edge_count == len(renderer._ladder_edges[lod_index])
    return _family_pixel_count(rendered, background)


def test_cube_hologram_is_visible_at_every_available_lod() -> None:
    vertices, edges = make_cube()
    counts = [
        _render_at_lod(vertices, edges, background, 0)
        for background in BACKGROUNDS
    ]
    assert min(counts) > 500, f"cube family-pixel counts by background: {counts}"


def test_icosphere_hologram_is_visible_at_every_lod_and_background() -> None:
    vertices, edges, _ = load_and_prepare_mesh("assets/models/test_icosphere.obj")
    level_counts = []
    for level in range(4):
        counts = [
            _render_at_lod(vertices, edges, background, level)
            for background in BACKGROUNDS
        ]
        level_counts.append(counts)
    assert all(min(counts) > 2000 for counts in level_counts), (
        f"icosphere family-pixel counts by LOD (black, grey): {level_counts}"
    )
