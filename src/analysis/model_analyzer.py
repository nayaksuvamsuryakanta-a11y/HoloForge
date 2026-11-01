"""Mesh ingestion and pipeline profiling.

``load_and_prepare_mesh`` turns external photogrammetry scans (.OBJ, .PLY,
.STL ...) into a wireframe the real-time loop can project at frame rate.
Scans are decimated to a few thousand faces; when the optional
``fast-simplification`` backend is absent, the unique-edge table is
subsampled instead so the renderer never pays for more than a handful of
draw calls. Geometry is centred on the origin and normalised to the
``TARGET_BOX_SIZE`` used by the default cube.

``ModelAnalyzer`` is still a Day-1 stub: it profiles latency and accuracy of
the landmark detection pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.rendering.projection_3d import make_cube, rotation_matrix

__all__ = [
    "DENSITY_G_CM3",
    "load_and_prepare_mesh",
    "load_source_mesh",
    "mesh_face_count",
    "compute_metrics",
    "export_transformed_mesh",
    "ModelAnalyzer",
]

logger = logging.getLogger("HoloForge")

# Mesh preparation -------------------------------------------------------
DECIMATION_TARGET = 5000      # faces to aim for when decimating
MAX_EDGES = 5000              # hard cap on edges drawn each frame
TARGET_BOX_SIZE = 150.0       # model-space span, matching make_cube()
SEED = 42                     # reproducible edge subsampling
# trimesh renamed the decimation kwarg across releases; try each spelling.
DECIMATION_KWARGS = ("face_count", "target_face_count")
DENSITY_G_CM3 = 1.24  # PLA density; mesh dimensions are assumed to be mm.

# Trimesh is a declared dependency but may be missing in a minimal install.
try:
    import trimesh
except ImportError:  # pragma: no cover - verified at runtime
    trimesh = None


def _flatten_scene(
    obj: "trimesh.Trimesh" | "trimesh.Scene" | None,
) -> Optional["trimesh.Trimesh"]:
    """Collapse a ``trimesh.Scene`` into a single ``Trimesh``.

    Photogrammetry exports frequently arrive as a scene holding several
    sub-meshes; the renderer needs one connected edge table, so the parts are
    concatenated. Non-mesh geometry (point clouds, lines) is dropped rather
    than breaking the concatenation.

    Parameters
    ----------
    obj : trimesh.Trimesh | trimesh.Scene | None
        Object returned by ``trimesh.load``.

    Returns
    -------
    trimesh.Trimesh | None
        A single mesh, or ``None`` when the scene contains no usable geometry.
    """
    if isinstance(obj, trimesh.Scene):
        meshes = [
            geometry
            for geometry in obj.geometry
            if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0
        ]
        if not meshes:
            return None
        return trimesh.util.concatenate(meshes)
    if isinstance(obj, trimesh.Trimesh):
        return obj
    return None


def _prepare(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Merge coincident vertices and drop degenerate faces before decimation.

    A scan whose vertices were duplicated at patch seams otherwise decimates
    unevenly, and zero-area faces can make the quadratic error metric diverge.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to clean.

    Returns
    -------
    trimesh.Trimesh
        The cleaned mesh, mutated in place.
    """
    mesh.merge_vertices()
    nondegenerate = mesh.nondegenerate_faces()
    if not bool(np.all(nondegenerate)):
        mesh.update_faces(nondegenerate)
    return mesh


def _decimate(mesh: "trimesh.Trimesh", target: int) -> Optional["trimesh.Trimesh"]:
    """Decimate ``mesh`` to ``target`` faces, tolerating trimesh versions.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to simplify.
    target : int
        Desired face count.

    Returns
    -------
    trimesh.Trimesh | None
        The decimated mesh, or ``None`` if simplification is unavailable.
    """
    for kwarg in DECIMATION_KWARGS:
        try:
            return mesh.simplify_quadric_decimation(**{kwarg: target})
        except (ImportError, ModuleNotFoundError, ValueError, TypeError, AttributeError):
            # Backend missing, unsupported kwarg, or degenerate geometry:
            # fall back to subsampling the edge table instead.
            continue
    return None


def _subsample_edges(mesh: "trimesh.Trimesh", max_edges: int) -> np.ndarray:
    """Return at most ``max_edges`` unique edges, sampled without replacement.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to sample edges from.
    max_edges : int
        Maximum number of edges to keep.

    Returns
    -------
    np.ndarray
        ``(m, 2)`` index table, ``m <= max_edges``.
    """
    edges = np.asarray(mesh.edges_unique, dtype=np.intp)
    if edges.ndim != 2 or edges.shape[1] != 2:
        return np.empty((0, 2), dtype=np.intp)

    total = edges.shape[0]
    if total <= max_edges:
        return edges

    rng = np.random.default_rng(SEED)
    chosen = rng.choice(total, max_edges, replace=False)
    return edges[chosen]


def _normalise_to_box(vertices: np.ndarray, mesh: "trimesh.Trimesh") -> np.ndarray:
    """Centre ``vertices`` and scale them to fit ``TARGET_BOX_SIZE``.

    Parameters
    ----------
    vertices : np.ndarray
        ``(n, 3)`` vertex positions in model space.
    mesh : trimesh.Trimesh
        Source mesh; its bounding-box centroid is the centre of mass.

    Returns
    -------
    np.ndarray
        ``(n, 3)`` vertices centred on the origin and scaled.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.size == 0:
        return vertices

    centroid = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    centred = vertices - centroid

    half_span = float(np.ptp(centred, axis=0).max()) / 2.0
    if half_span > 0.0:
        scale = (TARGET_BOX_SIZE / 2.0) / half_span
        return centred * scale
    return centred


def _fallback(
    filepath: str, exc: Optional[Exception] = None, reason: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Fall back to the default cube and record why the scan was rejected.

    Parameters
    ----------
    filepath : str
        Path of the scan that was rejected, used in the warning.
    exc : Exception | None
        The exception that triggered the fallback, if any.
    reason : str | None
        Short human-readable reason when no exception was raised.

    Returns
    -------
    vertices : np.ndarray
        ``(8, 3)`` cube vertices.
    edges : np.ndarray
        ``(12, 2)`` cube edges.
    face_count : int
        ``0``, because no mesh faces were available to report.
    """
    if exc is not None:
        logger.warning(
            "Rejected %r: %s: %s; falling back to the default cube",
            filepath,
            type(exc).__name__,
            exc,
        )
    if reason is not None:
        logger.warning("Rejected %r: %s; falling back to the default cube", filepath, reason)
    vertices, edges = make_cube()
    return vertices, edges, 0


def load_and_prepare_mesh(filepath: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load a photogrammetry scan and prepare it for wireframe rendering.

    The mesh is decimated to ``DECIMATION_TARGET`` faces when the
    simplification backend is present, otherwise its unique edges are
    subsampled to ``MAX_EDGES``. It is then centred on the origin and scaled
    to ``TARGET_BOX_SIZE`` so the default cube and imported scans share one
    pose space.

    Parameters
    ----------
    filepath : str
        Path to a mesh file (``.obj``, ``.ply``, ``.stl`` ...).

    Returns
    -------
    vertices : np.ndarray
        ``(n, 3)`` float64 vertex positions, centred and normalised.
    edges : np.ndarray
        ``(m, 2)`` intp vertex-index pairs, ``m <= MAX_EDGES``.
    face_count : int
        Original face count of the loaded mesh, or ``0`` when the scan could
        not be used and the default cube was substituted.

    Notes
    -----
    Any load or simplification failure falls back to the default cube rather
    than aborting the run, so the application keeps working without an asset.
    """
    if trimesh is None:
        return _fallback(filepath, reason="trimesh is not installed")

    try:
        raw = trimesh.load(str(filepath), force="mesh")
    except Exception as exc:  # noqa: BLE001 - every failure becomes a fallback
        return _fallback(filepath, exc)

    mesh = _flatten_scene(raw)
    if mesh is None:
        return _fallback(filepath, reason="loaded object contains no usable mesh")

    # Reported before cleaning: it is the count the scan actually shipped, and
    # it is what mesh_face_count sees for the same file.
    original_face_count = len(mesh.faces)

    try:
        mesh = _prepare(mesh)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a fallback
        return _fallback(filepath, exc)

    if len(mesh.faces) == 0:
        return _fallback(filepath, reason="mesh has no faces after cleaning")

    decimated = _decimate(mesh, DECIMATION_TARGET)
    if decimated is not None:
        # The decimation path must honour the same budget as the fallback,
        # otherwise a simplified scan could exceed MAX_EDGES and stall the
        # frame with more draw calls than the renderer expects.
        edges = _subsample_edges(decimated, MAX_EDGES)
        logger.info(
            "Decimated %s from %d faces to %d faces; retaining %d edges",
            filepath,
            original_face_count,
            len(decimated.faces),
            edges.shape[0],
        )
        # The edges index decimated.vertices, so the returned vertex array
        # must be the decimated one as well -- otherwise the renderer would
        # wire up the wrong vertices.
        return _normalise_to_box(decimated.vertices, decimated), edges, original_face_count

    edges = _subsample_edges(mesh, MAX_EDGES)
    logger.warning(
        "Decimation unavailable for %s; subsampled %d edges at random",
        filepath,
        edges.shape[0],
    )

    return _normalise_to_box(mesh.vertices, mesh), edges, original_face_count


def mesh_face_count(filepath: str) -> int:
    """Return the original face count of a mesh file.

    Used by the HUD to report how much geometry was discarded. Independent of
    ``load_and_prepare_mesh`` so the count is available even when the prepared
    mesh is not.

    Parameters
    ----------
    filepath : str
        Path to a mesh file.

    Returns
    -------
    int
        Face count of the loaded mesh, or ``0`` if it could not be read.
    """
    if trimesh is None:
        return 0

    try:
        mesh = trimesh.load(str(filepath), force="mesh")
    except (FileNotFoundError, OSError, ValueError, RuntimeError):
        return 0

    if not isinstance(mesh, trimesh.Trimesh):
        return 0
    return int(len(mesh.faces))


def load_source_mesh(filepath: str):
    """Load and flatten the original mesh for analysis and STL export."""
    if trimesh is None:
        raise RuntimeError("trimesh is not installed")
    raw = trimesh.load(str(filepath), force="mesh")
    mesh = _flatten_scene(raw)
    if mesh is None:
        raise ValueError(f"{filepath!r} contains no usable mesh")
    return mesh


def compute_metrics(mesh: "trimesh.Trimesh") -> dict:
    """Return geometry, topology, and PLA print-mass estimates for a mesh.

    Volume and mass use the mesh's coordinate units as millimetres. Surface
    area and extents are returned in the mesh's units and square units.
    """
    watertight = bool(mesh.is_watertight)
    volume_mm3 = abs(float(mesh.volume)) if watertight else None
    return {
        "volume": volume_mm3,
        "surface_area": float(mesh.area),
        "extents": np.asarray(mesh.bounding_box.extents, dtype=np.float64),
        "center_mass": np.asarray(mesh.center_mass, dtype=np.float64),
        "is_watertight": watertight,
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "estimated_mass_g": (
            volume_mm3 * DENSITY_G_CM3 / 1000.0 if volume_mm3 is not None else None
        ),
    }


def export_transformed_mesh(
    mesh: "trimesh.Trimesh",
    rot_x: float,
    rot_y: float,
    scale: float,
    out_dir: str | Path,
) -> Path:
    """Export a transformed mesh as a timestamped STL and return its path.

    The controller's screen-space pan is intentionally ignored: pan changes
    the view placement, while this export applies object rotation and scale.
    """
    transformed = mesh.copy()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_matrix(rot_x, rot_y, 0.0) * float(scale)
    transformed.apply_transform(matrix)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"holoforge_{timestamp}.stl"
    transformed.export(output_path, file_type="stl")
    logger.info("Exported transformed mesh to %s", output_path)
    return output_path


# Analysis ----------------------------------------------------------------
class ModelAnalyzer:
    """Profiles latency and accuracy of the landmark detection pipeline.

    Parameters
    ----------
    window_size : int, default 60
        Number of frames to retain in the rolling latency window.

    Attributes
    ----------
    latency_samples : np.ndarray
        ``(n,)`` array of per-frame inference durations in milliseconds.
    """

    def __init__(self, window_size: int = 60) -> None:
        """Initialize the rolling analysis buffers."""
        self.window_size = window_size
        self.latency_samples: np.ndarray = np.array([], dtype=np.float64)

    def record(self, elapsed_ms: float) -> None:
        """Append a new latency measurement, dropping the oldest if needed.

        TODO (Day 2): keep a circular buffer so memory stays bounded during
        long sessions instead of growing forever.
        """
        raise NotImplementedError("buffer management is Day 2 work")

    def mean_latency(self) -> Optional[float]:
        """Return the mean inference latency over the recorded window.

        Returns
        -------
        float | None
            Mean duration in milliseconds, or ``None`` with no samples.
        """
        raise NotImplementedError("statistics are Day 2 work")

    def accuracy_against(self, ground_truth: np.ndarray) -> Optional[float]:
        """Compare predicted landmarks to a ground-truth sequence.

        Parameters
        ----------
        ground_truth : np.ndarray
            ``(frames, num_landmarks, 3)`` reference landmark tensor.

        Returns
        -------
        float | None
            Mean per-joint accuracy, or ``None`` before any frames are seen.
        """
        raise NotImplementedError("accuracy scoring is Day 2 work")
