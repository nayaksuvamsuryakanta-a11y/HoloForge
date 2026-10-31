"""Real-time hand pose detection through the MediaPipe hand landmarker.

Detects hands in a BGR webcam frame and returns a normalised 21-point
landmark set for the most confident hand. Two geometry helpers sit on top of
the raw landmarks: a pinch ratio, normalised against the hand's own size so
that it stays stable as the hand moves toward and away from the camera, and
the palm centre taken from the middle metacarpal.

The landmarker is created in ``RUNNING_MODE_VIDEO`` because the caller must
supply a monotonically increasing per-frame timestamp; image mode rejects
that argument outright.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

if TYPE_CHECKING:
    # Imported only so static type checkers can resolve the return type of
    # ``detect``; mediapipe ships this generated proto alongside the runtime.
    from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark

__all__ = ["GrabDetector", "HandTracker", "Smoother"]

# MediaPipe hand-landmarker bundle. Kept relative so the model can live
# anywhere under the workspace; ``scripts/setup_windows.bat`` populates it.
MODEL_PATH = "models/hand_landmarker.task"

# Webcam frame geometry. Must match the CAP_PROP values set in main.py.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Detection confidence floor. Applied to detection, presence and tracking so
# that a hand is only reported when all three gates agree.
MIN_CONFIDENCE = 0.7

# Hands tracked per frame. The landmarker returns the most confident hands
# first, which is what lets the 3D engine drive two-hand zoom/pan.
NUM_HANDS = 2

# Landmark indices used by the geometry helpers. MediaPipe hands are 21
# points ordered wrist, thumb, index, middle, ring, pinky.
LANDMARK_WRIST = 0
LANDMARK_THUMB_TIP = 4
LANDMARK_INDEX_TIP = 8
LANDMARK_MIDDLE_MCP = 9

# Standard 21-point hand skeleton, expressed as pairs of landmark indices.
# Mirrors the MediaPipe canonical topology; drawn by hand rather than through
# ``mp.solutions.drawing_utils`` so the module never imports the solutions
# package.
HAND_CONNECTIONS: Sequence[Tuple[int, int]] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (5, 9),
    (9, 13),
    (13, 17),  # knuckle arc
)

# OpenCV drawing primitives for the skeleton overlay.
SKELETON_LINE_COLOR = (0, 255, 0)  # BGR green
SKELETON_LINE_THICKNESS = 2
SKELETON_POINT_COLOR = (0, 255, 255)  # BGR yellow
SKELETON_POINT_RADIUS = 3

# Grab gesture thresholds, expressed in pinch-ratio units.
GRAB_THRESHOLD = 0.35   # below this the hand is considered grabbed
RELEASE_THRESHOLD = 0.50  # above this the hand is considered released
# Values between the two thresholds hold the previous state, giving the
# hysteresis band that stops noise from toggling the gesture repeatedly.

# Smoothing applied to scalar streams (pinch ratio, palm position).
SMOOTHING_ALPHA = 0.2


class Smoother:
    """Exponential moving average filter for a single scalar stream.

    Each call blends the new sample toward the running value by
    ``SMOOTHING_ALPHA``; a low alpha favours stability over responsiveness,
    which is what keeps a tracked palm position from jittering frame to frame.
    """

    def __init__(self, initial: float = 0.0) -> None:
        """Seed the filter with an initial value.

        Parameters
        ----------
        initial : float, default ``0.0``
            Starting value of the running average.
        """
        self._value = initial

    def update(self, value: float) -> float:
        """Blend ``value`` into the running average and return the result."""
        self._value += SMOOTHING_ALPHA * (value - self._value)
        return self._value


class GrabDetector:
    """Hysteretic grab state machine driven by a pinch ratio.

    Closing the thumb below ``GRAB_THRESHOLD`` grabs, and opening back above
    ``RELEASE_THRESHOLD`` releases; values in between hold the previous state
    so that noise around the boundary cannot toggle it repeatedly.
    """

    def __init__(self) -> None:
        """Start in the released state."""
        self._grabbed = False

    def update(self, pinch_ratio: float) -> bool:
        """Advance the state machine and return the new grabbed state.

        Parameters
        ----------
        pinch_ratio : float
            Current thumb-to-index gap, normalised by hand size.

        Returns
        -------
        bool
            True if the hand is now grabbed, False otherwise.
        """
        if not self._grabbed and pinch_ratio < GRAB_THRESHOLD:
            self._grabbed = True
        elif self._grabbed and pinch_ratio > RELEASE_THRESHOLD:
            self._grabbed = False
        return self._grabbed

    @property
    def is_grabbed(self) -> bool:
        """True while the hand is currently holding a grab."""
        return self._grabbed


class HandTracker:
    """Detects hands and finger landmarks in a webcam video stream.

    Parameters
    ----------
    model_path : str, default ``MODEL_PATH``
        Path to the MediaPipe hand landmarker task bundle.
    num_hands : int, default ``NUM_HANDS``
        Maximum number of hands to detect in a single frame.

    Attributes
    ----------
    landmarker : vision.HandLandmarker
        The MediaPipe landmarker, created eagerly at construction.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        num_hands: int = NUM_HANDS,
    ) -> None:
        """Build the landmarker from the model bundle on disk.

        Parameters
        ----------
        model_path : str
            Path to the ``.task`` bundle.
        num_hands : int
            Hands to track per frame.

        Raises
        ------
        FileNotFoundError
            If ``model_path`` does not point at an existing file.
        RuntimeError
            If MediaPipe cannot construct the landmarker from the bundle.
        """
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"MediaPipe hand landmarker model not found at {model_path!r}. "
                "Run scripts/setup_windows.bat to download "
                "hand_landmarker.task into models/, or pass an existing "
                "bundle path as model_path."
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=MIN_CONFIDENCE,
            min_hand_presence_confidence=MIN_CONFIDENCE,
            min_tracking_confidence=MIN_CONFIDENCE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._num_hands = num_hands

    def detect(
        self, frame_bgr: np.ndarray, timestamp_ms: int
    ) -> Optional[List[List[NormalizedLandmark]]]:
        """Detect hands in a single BGR frame and return every hand found.

        Parameters
        ----------
        frame_bgr : np.ndarray
            ``(height, width, 3)`` BGR image captured from the webcam.
        timestamp_ms : int
            Monotonic frame timestamp in milliseconds. Must increase
            strictly across consecutive calls while in video mode.

        Returns
        -------
        list[list[NormalizedLandmark]] | None
            One 21-normalised-landmark list per detected hand, most confident
            first, or ``None`` when no hand is present.
        """
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks

    def get_pinch_ratio(
        self, landmarks: Optional[List[NormalizedLandmark]]
    ) -> float:
        """Return the thumb-to-index gap normalised by the hand size.

        The numerator is the distance between the thumb tip and the index
        tip (landmarks 4 and 8). The denominator is the wrist-to-middle-MCP
        distance (landmarks 0 and 9), which scales with the hand's size in
        the image and keeps the ratio roughly invariant to how close the
        hand sits to the camera.

        Parameters
        ----------
        landmarks : list[NormalizedLandmark] | None
            Landmarks returned by ``detect``, or ``None``.

        Returns
        -------
        float
            Pinch ratio, or ``0.0`` when no hand is detected or the
            reference distance is degenerate.
        """
        if landmarks is None:
            return 0.0

        thumb_tip = self._point(landmarks, LANDMARK_THUMB_TIP)
        index_tip = self._point(landmarks, LANDMARK_INDEX_TIP)
        wrist = self._point(landmarks, LANDMARK_WRIST)
        middle_mcp = self._point(landmarks, LANDMARK_MIDDLE_MCP)

        if thumb_tip is None or index_tip is None or wrist is None or middle_mcp is None:
            return 0.0

        pinch = np.linalg.norm(np.asarray(thumb_tip) - np.asarray(index_tip))
        span = np.linalg.norm(np.asarray(wrist) - np.asarray(middle_mcp))
        if span < 1e-9:
            return 0.0
        return float(pinch / span)

    def get_palm_center(
        self, landmarks: Optional[List[NormalizedLandmark]]
    ) -> Optional[Tuple[float, float]]:
        """Return the normalised (x, y) of the middle metacarpal (landmark 9).

        Landmark 9 sits at the base of the middle finger, close to the centre
        of the palm, so it makes a stable reference point for placing
        projected content.

        Parameters
        ----------
        landmarks : list[NormalizedLandmark] | None
            Landmarks returned by ``detect``, or ``None``.

        Returns
        -------
        tuple[float, float] | None
            Normalised ``(x, y)``, or ``None`` when no hand is detected.
        """
        if landmarks is None:
            return None
        return self._point(landmarks, LANDMARK_MIDDLE_MCP)

    def get_two_hand_stats(
        self, hands: Optional[List[List[NormalizedLandmark]]]
    ) -> Optional[dict]:
        """Return the midpoint and span of a two-hand configuration.

        The midpoint is the mean of the two palm centres and the span is the
        Euclidean distance between them. Together they drive two-hand zoom
        (span ratio) and pan (midpoint delta).

        Parameters
        ----------
        hands : list[list[NormalizedLandmark]] | None
            One normalised landmark list per hand, as returned by
            :meth:`detect`.

        Returns
        -------
        dict | None
            ``{"midpoint": (x, y), "span": distance, "num_hands": 2}`` in
            normalised image coordinates, or ``None`` when fewer than two
            hands are detected.
        """
        if hands is None or len(hands) < 2:
            return None

        first_palm = self._point(hands[0], LANDMARK_MIDDLE_MCP)
        second_palm = self._point(hands[1], LANDMARK_MIDDLE_MCP)
        if first_palm is None or second_palm is None:
            return None

        midpoint = tuple(
            (first_palm[axis] + second_palm[axis]) / 2.0 for axis in range(2)
        )
        span = float(
            np.linalg.norm(np.asarray(first_palm) - np.asarray(second_palm))
        )
        return {"midpoint": midpoint, "span": span, "num_hands": 2}

    def draw_skeleton(
        self,
        frame_bgr: np.ndarray,
        hands: Optional[List[List[NormalizedLandmark]]],
    ) -> np.ndarray:
        """Draw every hand's skeleton and landmarks onto a copy of the frame.

        Bone connections are rendered in green and each landmark in yellow.
        Normalised coordinates are scaled to the frame dimensions before
        drawing. The input frame is never modified.

        Parameters
        ----------
        frame_bgr : np.ndarray
            ``(height, width, 3)`` BGR image to annotate.
        hands : list[list[NormalizedLandmark]] | None
            One normalised landmark list per hand, as returned by
            :meth:`detect`.

        Returns
        -------
        np.ndarray
            A new frame carrying the skeleton overlay, or an unannotated
            copy when no hand was detected.
        """
        if hands is None:
            return frame_bgr.copy()

        height, width = frame_bgr.shape[:2]
        drawing = frame_bgr.copy()

        for landmarks in hands:
            for start, end in HAND_CONNECTIONS:
                first = self._scaled(landmarks, start, width, height)
                second = self._scaled(landmarks, end, width, height)
                if first is not None and second is not None:
                    cv2.line(
                        drawing,
                        first,
                        second,
                        SKELETON_LINE_COLOR,
                        SKELETON_LINE_THICKNESS,
                    )

            for index in range(len(landmarks)):
                point = self._scaled(landmarks, index, width, height)
                if point is not None:
                    cv2.circle(
                        drawing,
                        point,
                        SKELETON_POINT_RADIUS,
                        SKELETON_POINT_COLOR,
                        -1,
                    )

        return drawing

    def close(self) -> None:
        """Release the native resources held by the landmarker.

        Safe to call more than once, so teardown paths can close freely
        without checking whether construction succeeded.
        """
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    @staticmethod
    def _point(
        landmarks: List[NormalizedLandmark], index: int
    ) -> Optional[Tuple[float, float]]:
        """Return the normalised (x, y) of a landmark, or None if absent."""
        try:
            landmark = landmarks[index]
        except IndexError:
            return None
        if landmark is None:
            return None
        return float(landmark.x), float(landmark.y)

    @staticmethod
    def _scaled(
        landmarks: List[NormalizedLandmark], index: int, width: int, height: int
    ) -> Optional[Tuple[int, int]]:
        """Return the pixel coordinates of a landmark, or None if absent."""
        point = HandTracker._point(landmarks, index)
        if point is None:
            return None
        return int(round(point[0] * width)), int(round(point[1] * height))
