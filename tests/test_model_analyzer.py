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
from importlib.util import find_spec
from unittest import mock

import numpy as np
import pytest
import trimesh

from src.analysis.model_analyzer import (
    MAX_EDGES,
    compute_metrics,
    export_transformed_mesh,
    load_and_prepare_mesh,
    mesh_face_count,
)
from src.rendering.projection_3d import make_cube

SUBDIVISIONS = 4  # icosphere: 20 * 4**4 == 5120 faces

# trimesh loads the simplification backend lazily inside the decimation call,
# so the happy path is only reachable when that optional extra is installed.
HAS_FAST_SIMPLIFICATION = find_spec("fast_simplification") is not None


def _subdivided_icosahedron() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosahedron()
    for _ in range(SUBDIVISIONS):
        mesh = mesh.subdivide()
    return mesh


def export(mesh: trimesh.Trimesh, name: str = "scan.obj") -> str:
    """Write ``mesh`` to a fresh temp .obj and return its path."""
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, name)
    mesh.export(path, file_type="obj")
    return path


def temp_path(name: str) -> str:
    """Return a writable path inside a fresh temp directory."""
    return os.path.join(tempfile.mkdtemp(), name)


class TestLoadAndPrepareMesh:
    def test_returns_vertex_and_edge_arrays(self) -> None:
        vertices, edges, face_count = load_and_prepare_mesh(export(_subdivided_icosahedron()))
        assert isinstance(vertices, np.ndarray)
        assert isinstance(edges, np.ndarray)
        assert isinstance(face_count, int)
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert edges.ndim == 2 and edges.shape[1] == 2

    def test_reports_the_original_face_count(self) -> None:
        _, _, face_count = load_and_prepare_mesh(export(_subdivided_icosahedron()))
        assert face_count == 20 * 4 ** SUBDIVISIONS

    def test_centres_the_mesh_on_the_origin(self) -> None:
        vertices, _, _ = load_and_prepare_mesh(export(_subdivided_icosahedron()))
        # Centering is done against the bounding-box centroid, so the box --
        # rather than the arithmetic mean -- must be balanced on the origin.
        # Decimated geometry is asymmetric, so allow for scaling noise.
        assert np.allclose(vertices.max(axis=0) + vertices.min(axis=0), 0.0, atol=1e-6)

    def test_normalises_to_the_cube_box(self) -> None:
        vertices, _, _ = load_and_prepare_mesh(export(_subdivided_icosahedron()))
        span = float(vertices.ptp(axis=0).max())
        assert np.isclose(span, 150.0, atol=1e-6)

    def test_caps_the_edge_count(self) -> None:
        _, edges, _ = load_and_prepare_mesh(export(_subdivided_icosahedron()))
        assert edges.shape[0] <= MAX_EDGES

    @pytest.mark.skipif(
        not HAS_FAST_SIMPLIFICATION, reason="fast-simplification is not installed"
    )
    def test_decimation_path_caps_edges_to_max_edges(self) -> None:
        # With the optional backend present, simplification replaces the
        # subsample fallback, so this is the only test that reaches it. The
        # frame budget must hold there too.
        mesh = _subdivided_icosahedron()
        vertices, edges, face_count = load_and_prepare_mesh(export(mesh))
        # The returned geometry must actually be the simplified one: otherwise
        # this assertion would be satisfied by the fallback alone.
        assert vertices.shape[0] < len(mesh.vertices)
        assert face_count == 20 * 4 ** SUBDIVISIONS
        assert edges.shape[0] <= MAX_EDGES

    def test_edges_stay_within_vertex_range(self) -> None:
        vertices, edges, _ = load_and_prepare_mesh(export(_subdivided_icosahedron()))
        assert edges.min() >= 0
        assert edges.max() < vertices.shape[0]

    def test_falls_back_to_the_cube_for_a_missing_file(self) -> None:
        vertices, edges, face_count = load_and_prepare_mesh(temp_path("does-not-exist.obj"))
        cube_vertices, cube_edges = make_cube()
        assert vertices.shape == cube_vertices.shape
        assert edges.shape == cube_edges.shape
        assert face_count == 0

    def test_falls_back_to_the_cube_for_an_unreadable_file(self) -> None:
        # A file that exists but is not a valid mesh.
        path = temp_path("not-a-mesh.txt")
        with open(path, "w") as handle:
            handle.write("definitely not an OBJ file\n")
        vertices, edges, face_count = load_and_prepare_mesh(path)
        cube_vertices, cube_edges = make_cube()
        assert vertices.shape == cube_vertices.shape
        assert edges.shape == cube_edges.shape
        assert face_count == 0

    def test_uses_the_decimated_mesh_when_simplification_succeeds(self) -> None:
        # ``fast-simplification`` is not installed in this environment, so the
        # happy path cannot be exercised for real; fake a successful decimation
        # and confirm the prepared geometry comes from the simplified mesh.
        small = trimesh.creation.icosahedron()
        original_edges = small.edges_unique.shape[0]
        # A box is smaller than the icosahedron it replaces, so a strict
        # equality check proves the decimated mesh -- not the original --
        # supplies the returned edges.
        fake_decimation = trimesh.creation.box()
        path = export(small, "fake.obj")

        with mock.patch(
            "trimesh.Trimesh.simplify_quadric_decimation", return_value=fake_decimation
        ) as patched:
            vertices, edges, face_count = load_and_prepare_mesh(path)

        patched.assert_called()
        assert edges.shape[0] == fake_decimation.edges_unique.shape[0]
        assert edges.shape[0] < original_edges
        assert face_count == len(small.faces)
        # The returned edges index the returned vertices, so the renderer
        # never wires up the wrong geometry.
        assert edges.max() < vertices.shape[0]
        # Scaling a centred icosahedron keeps it centred on the origin.
        assert np.allclose(vertices.max(axis=0) + vertices.min(axis=0), 0.0, atol=1e-9)

    def test_handles_a_mesh_with_no_faces(self) -> None:
        empty = trimesh.Trimesh(vertices=np.zeros((3, 3), dtype=np.float64))
        path = export(empty, "empty.obj")
        # An empty export may be unreadable; either outcome must be safe.
        vertices, edges, face_count = load_and_prepare_mesh(path)
        cube_vertices, cube_edges = make_cube()
        assert isinstance(vertices, np.ndarray)
        assert isinstance(edges, np.ndarray)
        assert face_count == 0
        assert vertices.shape == cube_vertices.shape
        assert edges.shape == cube_edges.shape


class TestMeshFaceCount:
    def test_reports_the_original_face_count(self) -> None:
        path = export(_subdivided_icosahedron())
        expected = int(len(_subdivided_icosahedron().faces))
        assert mesh_face_count(path) == expected

    def test_returns_zero_for_a_missing_file(self) -> None:
        assert mesh_face_count(temp_path("does-not-exist.obj")) == 0

    def test_returns_zero_when_trimesh_is_unavailable(self) -> None:
        import src.analysis.model_analyzer as analyzer

        with mock.patch.object(analyzer, "trimesh", None):
            assert mesh_face_count(temp_path("does-not-exist.obj")) == 0


class TestMeshMetricsAndExport:
    def test_box_metrics_are_exact(self) -> None:
        metrics = compute_metrics(trimesh.creation.box(extents=(1.0, 2.0, 3.0)))
        assert metrics["volume"] == pytest.approx(6.0)
        assert metrics["surface_area"] == pytest.approx(22.0)
        assert np.array_equal(metrics["extents"], np.array([1.0, 2.0, 3.0]))
        assert np.allclose(metrics["center_mass"], (0.0, 0.0, 0.0))
        assert metrics["is_watertight"] is True
        assert metrics["is_winding_consistent"] is True
        assert metrics["vertex_count"] == 8
        assert metrics["face_count"] == 12
        assert metrics["estimated_mass_g"] == pytest.approx(6.0 * 1.24 / 1000.0)

    def test_open_mesh_has_no_volume(self) -> None:
        mesh = trimesh.creation.box()
        mesh.update_faces(np.arange(len(mesh.faces)) != 0)
        metrics = compute_metrics(mesh)
        assert metrics["is_watertight"] is False
        assert metrics["volume"] is None
        assert metrics["estimated_mass_g"] is None

    def test_exported_y_rotation_swaps_mesh_extents(self) -> None:
        mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
        with tempfile.TemporaryDirectory() as output_dir:
            path = export_transformed_mesh(mesh, 0.0, np.pi / 2.0, 1.0, output_dir)
            reloaded = trimesh.load(path, force="mesh")
        assert np.allclose(reloaded.bounding_box.extents, (3.0, 2.0, 1.0), atol=1e-6)
