"""Unit tests for the 3D projection layer.

Pure numpy: no webcam, no model bundle and no OpenCV context are needed to
verify the geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rendering.projection_3d import (
    FOCAL_LENGTH,
    MIN_Z,
    Z_OFFSET,
    make_cube,
    project_points,
    rotation_matrix,
)


FRAME_SHAPE = (480, 640)
CENTRE = np.array([FRAME_SHAPE[1] / 2.0, FRAME_SHAPE[0] / 2.0])


class TestRotationMatrix:
    def test_identity_at_zero_angles(self) -> None:
        assert np.allclose(rotation_matrix(0.0, 0.0, 0.0), np.eye(3))

    def test_is_orthogonal(self) -> None:
        matrix = rotation_matrix(0.4, -0.7, 1.2)
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-10)

    def test_determinant_is_plus_one(self) -> None:
        matrix = rotation_matrix(0.4, -0.7, 1.2)
        assert np.isclose(np.linalg.det(matrix), 1.0, atol=1e-10)

    def test_composes_as_rz_at_ry_at_rx(self) -> None:
        rx, ry, rz = 0.3, -0.5, 0.9
        expected = rotation_matrix_z(rz) @ rotation_matrix_y(ry) @ rotation_matrix_x(rx)
        assert np.allclose(rotation_matrix(rx, ry, rz), expected, atol=1e-12)

    def test_is_invertible_by_transpose(self) -> None:
        matrix = rotation_matrix(0.4, -0.7, 1.2)
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-10)


def rotation_matrix_x(angle: float) -> np.ndarray:
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]]
    )


def rotation_matrix_y(angle: float) -> np.ndarray:
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]])


def rotation_matrix_z(angle: float) -> np.ndarray:
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])


class TestMakeCube:
    def test_vertex_and_edge_shapes(self) -> None:
        vertices, edges = make_cube()
        assert vertices.shape == (8, 3)
        assert edges.shape == (12, 2)

    def test_edge_indices_are_in_range(self) -> None:
        _, edges = make_cube()
        assert edges.min() >= 0
        assert edges.max() < 8

    def test_edge_count_matches_a_cube(self) -> None:
        _, edges = make_cube()
        assert edges.shape[0] == 12

    def test_centred_on_origin(self) -> None:
        vertices, _ = make_cube()
        assert np.allclose(vertices.mean(axis=0), 0.0, atol=1e-12)

    def test_span_equals_size(self) -> None:
        vertices, _ = make_cube(size=150.0)
        extent = vertices.ptp(axis=0)
        assert np.allclose(extent, 150.0, atol=1e-12)

    def test_custom_size_scales_vertices(self) -> None:
        big, _ = make_cube(size=300.0)
        small, _ = make_cube(size=75.0)
        assert np.allclose(big, 4.0 * small, atol=1e-12)


class TestProjectPoints:
    @pytest.fixture
    def cube(self) -> tuple[np.ndarray, np.ndarray]:
        return make_cube()

    @pytest.fixture
    def identity_rotation(self) -> np.ndarray:
        return rotation_matrix(0.0, 0.0, 0.0)

    def test_positive_x_vertex_lands_right_of_centre(
        self, cube: tuple[np.ndarray, np.ndarray], identity_rotation: np.ndarray
    ) -> None:
        vertices, _ = cube
        points = project_points(vertices, identity_rotation, 1.0, (0.0, 0.0), FRAME_SHAPE)
        assert points[:, 0].max() > CENTRE[0]

    def test_positive_y_vertex_lands_below_centre(
        self, cube: tuple[np.ndarray, np.ndarray], identity_rotation: np.ndarray
    ) -> None:
        vertices, _ = cube
        points = project_points(vertices, identity_rotation, 1.0, (0.0, 0.0), FRAME_SHAPE)
        assert points[:, 1].max() > CENTRE[1]

    def test_doubling_scale_doubles_radial_distance(self, identity_rotation: np.ndarray) -> None:
        # Scale is applied in 3D, before the perspective divide, so the
        # radial-doubling law only holds when every vertex shares one depth.
        # A cube spans two depths (its front and back faces), so its vertices
        # would be foreshortened by different factors; use a coplanar cloud.
        vertices = np.array([
            [100.0, 0.0, 0.0], [-100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0], [0.0, -100.0, 0.0],
        ])
        once = project_points(vertices, identity_rotation, 1.0, (0.0, 0.0), FRAME_SHAPE)
        twice = project_points(vertices, identity_rotation, 2.0, (0.0, 0.0), FRAME_SHAPE)
        radii_once = np.linalg.norm(once - CENTRE, axis=1)
        radii_twice = np.linalg.norm(twice - CENTRE, axis=1)
        assert np.allclose(radii_twice, 2.0 * radii_once, atol=1e-9)

    def test_offset_translates_output(
        self, cube: tuple[np.ndarray, np.ndarray], identity_rotation: np.ndarray
    ) -> None:
        vertices, _ = cube
        unshifted = project_points(vertices, identity_rotation, 1.0, (0.0, 0.0), FRAME_SHAPE)
        shifted = project_points(vertices, identity_rotation, 1.0, (50.0, -30.0), FRAME_SHAPE)
        assert np.allclose(shifted, unshifted + np.array([50.0, -30.0]), atol=1e-9)

    def test_centres_unrotated_cube(self, cube: tuple[np.ndarray, np.ndarray]) -> None:
        vertices, _ = cube
        points = project_points(
            vertices, rotation_matrix(0.0, 0.0, 0.0), 1.0, (0.0, 0.0), FRAME_SHAPE
        )
        assert np.allclose(points.mean(axis=0), CENTRE, atol=1e-9)

    def test_min_z_clamps_the_divisor(self) -> None:
        # A vertex placed well behind the camera yields a negative depth;
        # MIN_Z must floor the divisor so the projection stays finite.
        vertices = np.array([[0.0, 0.0, -Z_OFFSET - 2000.0]])
        points = project_points(
            vertices, rotation_matrix(0.0, 0.0, 0.0), 1.0, (0.0, 0.0), FRAME_SHAPE
        )
        expected_x = 0.0 * FOCAL_LENGTH / MIN_Z + FRAME_SHAPE[1] / 2.0
        assert np.isclose(points[0, 0], expected_x, atol=1e-9)

    def test_rotation_moves_vertices_off_centre(
        self, cube: tuple[np.ndarray, np.ndarray]
    ) -> None:
        vertices, _ = cube
        rotated = project_points(
            vertices, rotation_matrix(0.5, 0.0, 0.0), 1.0, (0.0, 0.0), FRAME_SHAPE
        )
        unrotated = project_points(
            vertices, rotation_matrix(0.0, 0.0, 0.0), 1.0, (0.0, 0.0), FRAME_SHAPE
        )
        assert not np.allclose(rotated, unrotated, atol=1e-9)