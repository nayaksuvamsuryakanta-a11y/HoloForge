"""3D projection of hand landmarks onto a virtual scene.

Day 1 status: module skeleton. Day 2 will implement the world/view/projection
matrix chain and hand it a ``pygame`` surface to draw onto.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["Projection3D"]


class Projection3D:
    """Projects normalized hand landmarks into screen space.

    Parameters
    ----------
    focal_length : float, default 1.0
        Focal length in normalized device units.
    aspect : float, default 1.0
        Viewport aspect ratio.
    near : float, default 0.1
        Near clipping plane distance.
    far : float, default 100.0
        Far clipping plane distance.

    Attributes
    ----------
    projection : np.ndarray
        ``(4, 4)`` projection matrix, built at construction.
    """

    def __init__(
        self,
        focal_length: float = 1.0,
        aspect: float = 1.0,
        near: float = 0.1,
        far: float = 100.0,
    ) -> None:
        """Build the projection matrix from the camera parameters."""
        self.focal_length = focal_length
        self.aspect = aspect
        self.near = near
        self.far = far
        self.projection: np.ndarray = self._build_projection()

    def _build_projection(self) -> np.ndarray:
        """Construct a perspective projection matrix.

        TODO (Day 2): replace the stub matrix with the standard perspective
        matrix built from field of view, aspect and clipping planes.
        """
        return np.eye(4, dtype=np.float64)

    def project(self, landmarks: np.ndarray) -> np.ndarray:
        """Map normalized landmarks into 2D screen coordinates.

        Parameters
        ----------
        landmarks : np.ndarray
            ``(num_landmarks, 3)`` normalized landmark array.

        Returns
        -------
        np.ndarray
            ``(num_landmarks, 2)`` screen-space ``(x, y)`` coordinates.
        """
        raise NotImplementedError("projection math is Day 2 work")

    def bind_surface(self, width: int, height: int) -> None:
        """Attach a rendering surface of the given pixel dimensions.

        TODO (Day 2): create the ``pygame`` display and store the viewport
        transform that converts normalized device coordinates to pixels.
        """
        raise NotImplementedError("render surface wiring is Day 2 work")