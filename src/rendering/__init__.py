"""3D projection and wireframe rendering."""

from .projection_3d import (
    FOCAL_LENGTH,
    MIN_Z,
    Z_OFFSET,
    make_cube,
    project_points,
    rotation_matrix,
)
from .wireframe_renderer import (
    CORE_THICKNESS,
    GLOW_THICKNESS,
    HOLO_FAR,
    HOLO_GLOW_OUTER,
    HOLO_MID,
    HOLO_NEAR,
    WIREFRAME_COLOR,
    WIREFRAME_THICKNESS,
    WireframeRenderer,
)

__all__ = [
    "CORE_THICKNESS",
    "FOCAL_LENGTH",
    "GLOW_THICKNESS",
    "HOLO_FAR",
    "HOLO_GLOW_OUTER",
    "HOLO_MID",
    "HOLO_NEAR",
    "MIN_Z",
    "WIREFRAME_COLOR",
    "WIREFRAME_THICKNESS",
    "Z_OFFSET",
    "WireframeRenderer",
    "make_cube",
    "project_points",
    "rotation_matrix",
]