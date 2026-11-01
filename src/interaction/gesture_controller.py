"""Gesture-driven interaction with the 3D wireframe."""

from __future__ import annotations

__all__ = ["Mode", "GestureController"]

from enum import auto, Enum
from typing import Optional, Tuple

import math

import numpy as np


class Mode(Enum):
    """Interaction mode of the ``GestureController``."""

    IDLE = auto()  # no hand, no gesture
    ROTATE = auto()  # one hand is grabbed, driving rotation
    TRANSFORM = auto()  # two hands are driving scale and pan
    MEASURE = auto()  # a pinch episode is selecting measurement vertices


class GestureController:
    """Maps hand gestures onto the pose and scale of the 3D model.

    The controller owns the model's rotation (``rot_x``, ``rot_y``), uniform
    ``scale`` and screen ``offset`` (in pixels), plus an ``angular_velocity``
    that gives the released model inertia.

    Grab-rotate
    -----------
    A single grabbed hand sets ``angular_velocity`` from the palm's frame-to
    frame delta scaled by ``ROTATION_SENSITIVITY``, and integrates that into
    the rotation. Once the hand opens, the velocity is not zeroed: it is
    decayed by a frame-rate-independent exponential ``exp(-dt / TAU)`` so the
    model keeps spinning and gradually comes to rest -- the "spin-and-catch"
    behaviour.

    Catch
    -----
    Re-grabbing re-sources the velocity from the current hand motion rather
    than resuming the stale, decayed one. Because the delta is recomputed on
    the grab frame, the object follows the hand immediately instead of
    overshooting from leftover velocity.

    Two-hand zoom/pan
    -----------------
    Two palms enter ``TRANSFORM`` mode only after
    ``TWO_HAND_ENTER_FRAMES`` consecutive frames, and leave it after
    ``TWO_HAND_EXIT_FRAMES`` absent frames. That hysteresis prevents the mode
    from flickering when a detection briefly drops to one hand. While in
    ``TRANSFORM`` the scale tracks the ratio of the current palm span to the
    previous span, clamped to ``[SCALE_MIN, SCALE_MAX]``, and the offset tracks
    the palm midpoint delta scaled by ``PAN_SENSITIVITY``. One-hand
    grab-rotate is suspended while ``TRANSFORM`` is active.

    Attributes
    ----------
    rot_x, rot_y : float
        Model rotation about the x and y axes, radians.
    scale : float
        Uniform model scale, clamped to ``[SCALE_MIN, SCALE_MAX]``.
    offset : np.ndarray
        ``(2,)`` screen offset in pixels.
    angular_velocity : np.ndarray
        ``(2,)`` angular velocity ``(d(rot_x)/dt, d(rot_y)/dt)`` in rad/s.
    mode : Mode
        Current interaction mode.
    """

    ROTATION_SENSITIVITY = 3.0  # rad/s per normalised palm pixel
    TAU = 0.35  # s; friction time constant for the spin decay
    TWO_HAND_ENTER_FRAMES = 3  # consecutive two-hand frames before TRANSFORM
    TWO_HAND_EXIT_FRAMES = 5  # consecutive absent frames before leaving TRANSFORM
    PAN_SENSITIVITY = 1.0  # pixels of offset per normalised midpoint pixel
    SCALE_MIN = 0.5
    SCALE_MAX = 3.0

    def __init__(self, rotation_sensitivity: float = ROTATION_SENSITIVITY,
                 pan_sensitivity: float = PAN_SENSITIVITY) -> None:
        """Start the model unrotated, unscaled and centred."""
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.scale = 1.0
        self.offset: np.ndarray = np.zeros(2, dtype=np.float64)
        self.angular_velocity: np.ndarray = np.zeros(2, dtype=np.float64)
        self.mode = Mode.IDLE
        self.rotation_sensitivity = float(rotation_sensitivity)
        self.pan_sensitivity = float(pan_sensitivity)

        self._last_palm: Optional[np.ndarray] = None
        self._two_hand_frames = 0
        self._no_two_hand_frames = 0
        self._entry_span = 1.0
        self._prev_stats: Optional[dict] = None

    def update_one_hand(
        self, palm_smoothed: Tuple[float, float], is_grabbed: bool, dt: float
    ) -> None:
        """Advance the one-hand grab-rotate state by ``dt`` seconds.

        Parameters
        ----------
        palm_smoothed : tuple[float, float]
            Smoothed normalised palm centre ``(x, y)``.
        is_grabbed : bool
            Whether the hand is currently pinched.
        dt : float
            Elapsed wall-clock time since the previous frame, seconds.
        """
        if self.mode == Mode.TRANSFORM:
            # Two-hand zoom/pan owns the model; ignore one-hand input.
            self._last_palm = np.asarray(palm_smoothed, dtype=np.float64)
            return

        palm = np.asarray(palm_smoothed, dtype=np.float64)
        if self._last_palm is not None:
            delta = palm - self._last_palm
            if is_grabbed:
                # Catch: velocity is sourced from the current hand motion.
                self.angular_velocity = delta * self.rotation_sensitivity
            else:
                self.angular_velocity *= math.exp(-dt / self.TAU)

            self.rot_x += self.angular_velocity[0] * dt
            self.rot_y += self.angular_velocity[1] * dt

        self._last_palm = palm
        self.mode = Mode.ROTATE if is_grabbed else Mode.IDLE

    def update_two_hands(self, stats: Optional[dict], dt: float) -> None:
        """Advance the two-hand zoom/pan state by ``dt`` seconds.

        Parameters
        ----------
        stats : dict | None
            ``{"midpoint": (x, y), "span": distance}`` for the current frame,
            or ``None`` when fewer than two hands were detected.
        dt : float
            Elapsed wall-clock time since the previous frame, seconds.
        """
        if stats is not None and stats.get("num_hands", 0) >= 2:
            self._two_hand_frames += 1
            self._no_two_hand_frames = 0

            midpoint = np.asarray(stats["midpoint"], dtype=np.float64)
            span = float(stats["span"])

            if self._two_hand_frames >= self.TWO_HAND_ENTER_FRAMES:
                if self.mode != Mode.TRANSFORM:
                    self._entry_span = span
                    self.mode = Mode.TRANSFORM

                prev_stats = self._prev_stats
                if prev_stats is not None:
                    prev_span = float(prev_stats["span"])
                    prev_midpoint = np.asarray(prev_stats["midpoint"], dtype=np.float64)
                    if prev_span > 0.0:
                        self.scale = np.clip(
                            self.scale * (span / prev_span),
                            self.SCALE_MIN,
                            self.SCALE_MAX,
                        )
                    self.offset += (midpoint - prev_midpoint) * self.pan_sensitivity

                self._prev_stats = {"midpoint": midpoint.copy(), "span": span}
            return

        self._two_hand_frames = 0
        self._no_two_hand_frames += 1
        self._prev_stats = None

        if self.mode == Mode.TRANSFORM and self._no_two_hand_frames >= self.TWO_HAND_EXIT_FRAMES:
            self.mode = Mode.IDLE

    def reset(self) -> None:
        """Return the model to its unrotated, unscaled, centred pose."""
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.scale = 1.0
        self.offset.fill(0.0)
        self.angular_velocity.fill(0.0)
        self.mode = Mode.IDLE
        self._last_palm = None
        self._two_hand_frames = 0
        self._no_two_hand_frames = 0
        self._entry_span = 1.0
        self._prev_stats = None
