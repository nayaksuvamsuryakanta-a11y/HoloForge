"""Focused coverage for the density-adaptive renderer."""

import numpy as np

from src.rendering.wireframe_renderer import (
    HOLO_FAR,
    HOLO_GLOW_OUTER,
    HOLO_MID,
    HOLO_NEAR,
    WireframeRenderer,
    build_ladder,
    stroke_rules,
)


def test_ladder_is_a_deterministic_strict_nested_subset() -> None:
    edges = np.column_stack((np.arange(5000), np.arange(5000) + 1))
    indices, tables = build_ladder(edges)
    assert [len(level) for level in indices] == [5000, 2500, 1200, 600]
    for larger, smaller in zip(indices, indices[1:]):
        assert len(smaller) < len(larger)
        assert set(smaller).issubset(set(larger))
    repeated, _ = build_ladder(edges)
    assert all(np.array_equal(a, b) for a, b in zip(indices, repeated))
    assert all(np.array_equal(edges[index], table) for index, table in zip(indices, tables))


def test_lod_hysteresis_moves_one_level_at_thresholds() -> None:
    edges = np.tile(np.array([[0, 1]]), (5000, 1))
    points = np.array([[0.0, 0.0], [9.0, 0.0]])
    renderer = WireframeRenderer()
    renderer.prepare_lod(edges)

    assert renderer._select_lod_level(points, edges)[1] == 5000
    points[1, 0] = 3.0
    assert renderer._select_lod_level(points, edges)[1] == 2500
    points[1, 0] = 5.0
    assert renderer._select_lod_level(points, edges)[1] == 2500
    points[1, 0] = 9.0
    assert renderer._select_lod_level(points, edges)[1] == 5000


def test_stroke_rules_follow_active_edge_count() -> None:
    assert stroke_rules(5000) == (False, 1, False)
    assert stroke_rules(1200) == (True, 2, False)
    assert stroke_rules(600) == (True, 2, True)


def test_hologram_colours_are_bgr_cyan_family() -> None:
    for blue_green_red in (HOLO_GLOW_OUTER, HOLO_FAR, HOLO_MID, HOLO_NEAR):
        blue, green, red = blue_green_red
        assert blue >= green > red == 0
