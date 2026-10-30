"""Profiling and accuracy analysis of the hand-landmark pipeline.

Day 1 status: module skeleton. Day 2 will consume landmark streams from
``vision`` and emit latency distributions plus gesture classification metrics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["ModelAnalyzer"]


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