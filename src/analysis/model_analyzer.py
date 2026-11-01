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
from typing import Optional, Tuple

import numpy as np

from src.rendering.projection_3d import make_cube

__all__ = ["load_and_prepare_mesh", "mesh_face_count", "ModelAnalyzer"]

logger = logging.getLogger("HoloForge")

# Mesh preparation -------------------------------------------------------
DECIMATION_TARGET = 5000      # faces to aim for when decimating
MAX_EDGES = 5000              # hard cap on edges drawn each frame
TARGET_BOX_SIZE = 150.0       # model-space span, matching make_cube()
SEED = 42                     # reproducible edge subsampling
# trimesh renamed the decimation kwarg across releases; try each spelling.
DECIMATION_KWARGS = ("face_count", "target_face_count")

# Trimesh is a declared dependency but may be missing in a minimal install.
try:
    import trimesh
except ImportError:  # pragma: no cover - verified at runtime
    trimesh = None


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


def load_and_prepare_mesh(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
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

    Notes
    -----
    Any load or simplification failure falls back to the default cube rather
    than aborting the run, so the application keeps working without an asset.
    """
    if trimesh is None:
        logger.warning("trimesh is not installed; falling back to the default cube")
        return make_cube()

    try:
        mesh = trimesh.load(str(filepath), force="mesh")
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        logger.warning("Could not load %r: %s; falling back to the default cube", filepath, exc)
        return make_cube()

    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0:
        logger.warning("Loaded object is not a usable mesh; falling back to the default cube")
        return make_cube()

    decimated = _decimate(mesh, DECIMATION_TARGET)
    if decimated is not None:
        edges = np.asarray(decimated.edges_unique, dtype=np.intp)
        logger.info(
            "Decimated %s from %d faces to %d faces; retaining %d edges",
            filepath,
            len(mesh.faces),
            len(decimated.faces),
            edges.shape[0],
        )
    else:
        edges = _subsample_edges(mesh, MAX_EDGES)
        logger.warning(
            "Decimation unavailable for %s; subsampled %d edges at random",
            filepath,
            edges.shape[0],
        )

    return _normalise_to_box(mesh.vertices, mesh), edges


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