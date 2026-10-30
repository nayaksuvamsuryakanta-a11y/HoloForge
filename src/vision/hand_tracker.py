"""Real-time hand pose detection through the MediaPipe hand landmarker.

Day 1 status: module skeleton only. Day 2 will bind the MediaPipe
``HandLandmarker``, convert detections into normalized landmark frames and
expose a small gesture vocabulary to the analysis package.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["HandTracker"]


class HandTracker:
    """Detects hands and finger landmarks in a webcam video stream.

    Parameters
    ----------
    model_path : str, default ``"models/hand_landmarker.task"``
        Path to the MediaPipe hand landmarker task bundle.
    num_hands : int, default 2
        Maximum number of hands to detect in a single frame.
    min_confidence : float, default 0.5
        Minimum confidence a detection must have to be accepted.

    Attributes
    ----------
    landmarker : object | None
        Initialized MediaPipe landmarker, or ``None`` until ``start()``.
    running : bool
        Whether the camera stream is active.
    """

    def __init__(
        self,
        model_path: str = "models/hand_landmarker.task",
        num_hands: int = 2,
        min_confidence: float = 0.5,
    ) -> None:
        """Configure the tracker without allocating any native resources."""
        self.model_path = model_path
        self.num_hands = num_hands
        self.min_confidence = min_confidence
        self.landmarker: Optional[object] = None
        self.running = False

    def start(self) -> None:
        """Create the landmarker and begin capturing frames.

        TODO (Day 2): build ``HandLandmarkerRunningOptions`` and
        ``HandLandmarkerOptions`` with the correct model asset path, then
        create the ``HandLandmarker`` from options so that ``tick`` receives
        real landmark data instead of raising.
        """
        raise NotImplementedError("hand landmarker wiring is Day 2 work")

    def stop(self) -> None:
        """Release the camera and the native landmarker handle."""
        raise NotImplementedError("hand landmarker wiring is Day 2 work")

    def tick(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run detection on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            ``(height, width, 3)`` BGR image captured from the camera.

        Returns
        -------
        np.ndarray | None
            ``(num_landmarks, 3)`` array of normalized ``(x, y, z)``
            landmarks, or ``None`` when no hand is detected.
        """
        raise NotImplementedError("detection logic is Day 2 work")