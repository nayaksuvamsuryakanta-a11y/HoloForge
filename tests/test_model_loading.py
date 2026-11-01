"""Headless tests for the hardened mesh loader.

No webcam, no MediaPipe bundle, no GUI and no network access are needed.
Meshes are generated in memory with ``trimesh`` and exported to a temporary
directory created with ``tempfile``; pytest's ``tmp_path`` fixture is unusable
on this host because its ``AppData\\Local\\Temp`` scratch dir is owned by a
different security context.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import trimesh

from src.analysis.model_analyzer import MAX_EDGES, load_and_prepare_mesh
from src.rendering.projection_3d import make_cube

SUBDIVISIONS = 3  # icosphere: 20 * 4**3 == 1280 faces


def _subdivided_icosahedron() -> trimesh.Trimesh:
    """Return an icosphere with ``SUBDIVISIONS`` recursive subdivisions."""
    mesh = trimesh.creation.icosahedron()
    for _ in range(SUBDIVISIONS):
        mesh = mesh.subdivide()
    return mesh


def _export(exported: trimesh.Trimesh) -> str:
    """Write ``exported`` to a temp .obj and return the path."""
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "icosphere.obj")
    exported.export(path, file_type="obj")
    return path


def _export_pointcloud() -> str:
    """Write a vertices-only cloud to a temp .obj and return the path."""
    cloud = trimesh.PointCloud(vertices=np.random.default_rng(1).random((20, 3)))
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "cloud.obj")
    cloud.export(path, file_type="obj")
    return path


class TestPolygonalScan:
    def test_returns_vertices_edges_and_face_count(self) -> None:
        path = _export(_subdivided_icosahedron())
        vertices, edges, face_count = load_and_prepare_mesh(path)

        assert face_count == 20 * 4 ** SUBDIVISIONS
        assert vertices.shape[1] == 3
        assert edges.shape[1] == 2
        assert edges.shape[0] > 0

    def test_vertices_are_centred_on_the_origin(self) -> None:
        vertices, _, _ = load_and_prepare_mesh(_export(_subdivided_icosahedron()))
        # Centering uses the bounding-box centroid, so the box is balanced.
        assert np.allclose(vertices.max(axis=0) + vertices.min(axis=0), 0.0, atol=1e-9)

    def test_vertices_are_normalised_to_the_cube_box(self) -> None:
        vertices, _, _ = load_and_prepare_mesh(_export(_subdivided_icosahedron()))
        assert np.isclose(float(vertices.ptp(axis=0).max()), 150.0, atol=1e-6)

    def test_edges_stay_within_vertex_range(self) -> None:
        vertices, edges, _ = load_and_prepare_mesh(_export(_subdivided_icosahedron()))
        assert edges.min() >= 0
        assert edges.max() < vertices.shape[0]

    def test_edge_count_is_capped(self) -> None:
        # A scan larger than the cap reaches the subsample path, whose
        # contract is that the renderer never pays for more than MAX_EDGES.
        big = trimesh.creation.icosahedron()
        for _ in range(6):  # 20 * 4**6 == 81920 faces
            big = big.subdivide()
        _, edges, _ = load_and_prepare_mesh(_export(big))
        assert edges.shape[0] <= MAX_EDGES


class TestPointCloud:
    def test_falls_back_to_the_cube_without_raising(self) -> None:
        path = _export_pointcloud()
        vertices, edges, face_count = load_and_prepare_mesh(path)

        cube_vertices, cube_edges = make_cube()
        assert vertices.shape == cube_vertices.shape
        assert edges.shape == cube_edges.shape
        assert face_count == 0


class TestDegenerateAndMissingInput:
    def test_missing_file_falls_back_to_the_cube(self) -> None:
        directory = tempfile.mkdtemp()
        vertices, edges, face_count = load_and_prepare_mesh(
            os.path.join(directory, "does-not-exist.obj")
        )
        cube_vertices, cube_edges = make_cube()
        assert vertices.shape == cube_vertices.shape
        assert edges.shape == cube_edges.shape
        assert face_count == 0

    def test_scene_is_flattened_into_one_mesh(self) -> None:
        part = _subdivided_icosahedron()
        scene = trimesh.Scene(geometry=[part.copy(), part.copy()])
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "scene.obj")
        scene.export(path, file_type="obj")

        vertices, edges, face_count = load_and_prepare_mesh(path)

        # Two icospheres, so the face count must exceed one part's.
        assert face_count > len(part.faces)
        assert np.isclose(float(vertices.ptp(axis=0).max()), 150.0, atol=1e-6)
        assert edges.shape[0] > 0

    def test_zero_face_mesh_falls_back_to_the_cube(self) -> None:
        empty = trimesh.Trimesh(vertices=np.zeros((3, 3), dtype=np.float64))
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "empty.obj")
        empty.export(path, file_type="obj")

        vertices, edges, face_count = load_and_prepare_mesh(path)
        cube_vertices, cube_edges = make_cube()
        assert vertices.shape == cube_vertices.shape
        assert edges.shape == cube_edges.shape
        assert face_count == 0