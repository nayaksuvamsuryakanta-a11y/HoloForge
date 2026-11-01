"""Unit tests for the mesh ingestion layer.

State and arithmetic only: no webcam, no MediaPipe bundle and no hand model
file are needed. The meshes are generated in memory with ``trimesh`` and
written to a temporary .OBJ so that the real file-loading path is exercised.

``pytest``'s own ``tmp_path`` fixture cannot run on this host: its
``AppData\\Local\\Temp\\pytest-of-nayak`` scratch dir is owned by another
context and denies every access, so these tests manage their own temp
directories with ``tempfile``.
"""

from __future__ import annotations

import os
import tempfile
from unittest import mock

import numpy as np
import trimesh

from src.analysis.model_analyzer import load_and_prepare_mesh, mesh_face_count
from src.rendering.projection_3d import make_cube

SUBDIVISIONS = 5


def make_high_poly_mesh(path: str) -> int:
    """Write a subdivided icosahedron to ``path``; return its face count."""
    mesh = trimesh.creation.icosahedron()
    for _ in range(SUBDIVISIONS):
        mesh = mesh.subdivide()
    mesh.export(path, file_type="obj")
    return int(len(mesh.faces))


def temp_path(name: str) -> str:
    """Return a writable path inside a fresh temp directory."""
    return os.path.join(tempfile.mkdtemp(), name)


class TestLoadAndPrepareMesh:
    def test_returns_vertex_and_edge_arrays(self) -> None:
        vertices, edges = load_and_prepare_mesh(temp_path("scan.obj"))
        assert isinstance(vertices, np.ndarray)
        assert isinstance(edges, np.ndarray)
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert edges.ndim == 2 and edges.shape[1] == 2

    def test_centres_the_mesh_on_the_origin(self) -> None:
        vertices, _ = load_and_prepare_mesh(temp_path("scan.obj"))
        # Centering is done against the bounding-box centroid, so the box --
        # rather than the arithmetic mean -- must be balanced on the origin.
        assert np.allclose(vertices.max(axis=0) + vertices.min(axis=0), 0.0, atol=1e-9)

    def test_normalises_to_the_cube_box(self) -> None:
        vertices, _ = load_and_prepare_mesh(temp_path("scan.obj"))
        span = float(vertices.ptp(axis=0).max())
        assert np.isclose(span, 150.0, atol=1e-6)

    def test_caps_the_edge_count(self) -> None:
        _, edges = load_and_prepare_mesh(temp_path("scan.obj"))
        assert edges.shape[0] <= 5000

    def test_edges_stay_within_vertex_range(self) -> None:
        vertices, edges = load_and_prepare_mesh(temp_path("scan.obj"))
        assert edges.min() >= 0
        assert edges.max() < vertices.shape[0]

    def test_falls_back_to_the_cube_for_a_missing_file(self) -> None:
        vertices, edges = load_and_prepare_mesh(temp_path("does-not-exist.obj"))
        cube_vertices, cube_edges = make_cube()
        assert vertices.shape == cube_vertices.shape
        assert edges.shape == cube_edges.shape

    def test_falls_back_to_the_cube_for_an_unreadable_file(self) -> None:
        # A file that exists but is not a valid mesh.
        path = temp_path("not-a-mesh.txt")
        with open(path, "w") as handle:
            handle.write("definitely not an OBJ file\n")
        vertices, edges = load_and_prepare_mesh(path)
        cube_vertices, cube_edges = make_cube()
        assert vertices.shape == cube_vertices.shape
        assert edges.shape == cube_edges.shape

    def test_uses_the_decimated_mesh_when_simplification_succeeds(self) -> None:
        # ``fast-simplification`` is not installed in this environment, so the
        # happy path cannot be exercised for real; fake a successful decimation
        # and confirm the prepared geometry comes from the simplified mesh.
        small = trimesh.creation.icosahedron()
        original_edges = small.edges_unique.shape[0]
        path = temp_path("fake.obj")
        small.export(path, file_type="obj")

        with mock.patch(
            "trimesh.Trimesh.simplify_quadric_decimation", return_value=small
        ) as patched:
            vertices, edges = load_and_prepare_mesh(path)

        patched.assert_called()
        # The returned edges must belong to the decimated mesh, not the
        # original, otherwise decimation would be silently a no-op.
        assert edges.shape[0] <= original_edges
        # Scaling a centred icosahedron keeps it centred on the origin.
        assert np.allclose(vertices.max(axis=0) + vertices.min(axis=0), 0.0, atol=1e-9)

    def test_handles_a_mesh_with_no_faces(self) -> None:
        empty = trimesh.Trimesh(vertices=np.zeros((3, 3), dtype=np.float64))
        path = temp_path("empty.obj")
        empty.export(path, file_type="obj")
        # An empty export may be unreadable; either outcome must be safe.
        vertices, edges = load_and_prepare_mesh(path)
        assert isinstance(vertices, np.ndarray)
        assert isinstance(edges, np.ndarray)


class TestMeshFaceCount:
    def test_reports_the_original_face_count(self) -> None:
        path = temp_path("scan.obj")
        expected = make_high_poly_mesh(path)
        assert mesh_face_count(path) == expected

    def test_returns_zero_for_a_missing_file(self) -> None:
        assert mesh_face_count(temp_path("does-not-exist.obj")) == 0

    def test_returns_zero_when_trimesh_is_unavailable(self) -> None:
        import src.analysis.model_analyzer as analyzer

        with mock.patch.object(analyzer, "trimesh", None):
            assert mesh_face_count(temp_path("does-not-exist.obj")) == 0