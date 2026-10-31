"""Entry point for the HoloForge real-time hand-tracking pipeline.

Day 2 drives a live webcam loop that detects hands, draws the skeleton
overlay and reports a smoothed FPS, a pinch ratio and a palm position. Day 5
adds a 3D wireframe cube that the hand gestures manipulate, with a keyboard
fallback for the same transforms. Days 6-7 bring grab-rotate with spin-and-catch
inertia, and Day 8 adds two-hand zoom/pan. Press ``q`` to shut down.
"""

from __future__ import annotations

import logging
import math
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.interaction.gesture_controller import GestureController, Mode
from src.rendering.projection_3d import make_cube, project_points, rotation_matrix
from src.rendering.wireframe_renderer import WireframeRenderer
from src.utils.helpers import setup_logging
from src.vision.hand_tracker import GrabDetector, HandTracker, Smoother

__all__ = ["main"]

# Camera --------------------------------------------------------------------
CAMERA_INDICES = (0, 1)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30

# HUD -----------------------------------------------------------------------
HUD_ORIGIN = (10, 30)  # baseline of the first HUD line, pixels from top-left
HUD_LINE_GAP = 24  # baseline-to-baseline spacing between HUD lines, pixels
HUD_PADDING = 6  # space between the text and the backing rectangle, pixels
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_SCALE = 0.55
HUD_FONT_THICKNESS = 1
HUD_TEXT_COLOR = (220, 220, 220)  # BGR light gray
HUD_BACKGROUND = (20, 20, 40)  # BGR dark blue-gray

# Per-line HUD colours for the gesture state indicator.
HUD_STATE_IDLE_COLOR = (0, 255, 0)    # BGR green
HUD_STATE_GRAB_COLOR = (0, 0, 255)    # BGR red
HUD_STATE_HOLD_COLOR = (0, 165, 255)  # BGR amber, used while a hand is lost

# Per-line HUD colours for the interaction mode indicator.
HUD_MODE_IDLE_COLOR = (0, 255, 0)      # BGR green
HUD_MODE_ROTATE_COLOR = (0, 255, 255)  # BGR yellow
HUD_MODE_TRANSFORM_COLOR = (255, 165, 0)  # BGR orange
HUD_SCALE_COLOR = (200, 200, 255)  # BGR pale blue

# FPS estimation ------------------------------------------------------------
# Weight given to the previous smoothed value; a high smoothing factor keeps
# stable readings on a laptop webcam whose per-frame latency jitters.
FPS_SMOOTHING = 0.9
FPS_INIT = 30.0

# Window --------------------------------------------------------------------
WINDOW_NAME = "HoloForge - Hand Tracking"
EXIT_KEY = ord("q")

# Keyboard fallback ---------------------------------------------------------
# OpenCV returns these key codes for the arrow keys on Windows.
KEY_LEFT = 2424832
KEY_UP = 2555904
KEY_RIGHT = 2621440
KEY_DOWN = 2490368

ROTATE_STEP = 5.0  # degrees per arrow press
SCALE_STEP = 0.1  # scale delta per +/- press


def open_webcam() -> cv2.VideoCapture:
    """Open a built-in laptop webcam at 640x480.

    Probes the common built-in camera indices in order, so a camera already
    occupied by another process falls through to the next index instead of
    failing outright.

    Returns
    -------
    cv2.VideoCapture
        Open capture handle configured for 640x480.

    Raises
    ------
    RuntimeError
        If none of the probed indices yields an open stream.
    """
    logger = logging.getLogger("HoloForge")

    for index in CAMERA_INDICES:
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            capture.set(cv2.CAP_PROP_FPS, FRAME_FPS)
            actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(
                "Opened camera index %s at %sx%s",
                index,
                actual_width,
                actual_height,
            )
            return capture
        capture.release()

    raise RuntimeError(
        f"No webcam opened at indices {list(CAMERA_INDICES)}. "
        "Close any other application using the camera and try again."
    )


def draw_hud(
    frame: np.ndarray,
    lines: list[str],
    colors: Optional[list[tuple[int, int, int]]] = None,
) -> None:
    """Render a block of status text into the top-left corner of ``frame``.

    A dark backing rectangle is drawn first so the text stays legible over
    bright or busy backgrounds. The frame is modified in place.

    Parameters
    ----------
    frame : np.ndarray
        ``(height, width, 3)`` BGR image to annotate.
    lines : list[str]
        Status strings, rendered top to bottom in the order given.
    colors : list[tuple[int, int, int]] | None, default ``None``
        Per-line BGR text colours. If ``None`` every line uses
        ``HUD_TEXT_COLOR``; a shorter list than ``lines`` is padded with
        that colour.
    """
    if not lines:
        return

    (text_width, text_height), _ = cv2.getTextSize(
        lines[0], HUD_FONT, HUD_FONT_SCALE, HUD_FONT_THICKNESS
    )
    origin_x, origin_y = HUD_ORIGIN
    last_baseline = origin_y + (len(lines) - 1) * HUD_LINE_GAP
    block_width = text_width + 2 * HUD_PADDING
    per_line = colors if colors is not None else [HUD_TEXT_COLOR] * len(lines)

    cv2.rectangle(
        frame,
        (origin_x, origin_y - text_height - HUD_PADDING),
        (origin_x + block_width, last_baseline + text_height + HUD_PADDING),
        HUD_BACKGROUND,
        -1,
    )

    for index, line in enumerate(lines):
        colour = per_line[index] if index < len(per_line) else HUD_TEXT_COLOR
        cv2.putText(
            frame,
            line,
            (origin_x + HUD_PADDING, origin_y + index * HUD_LINE_GAP),
            HUD_FONT,
            HUD_FONT_SCALE,
            colour,
            HUD_FONT_THICKNESS,
        )


def build_hud(
    smoothed_fps: float,
    pinch_ratio: Optional[float],
    palm_center: Optional[Tuple[float, float]],
    smoothed_palm: Optional[Tuple[float, float]],
    grabbed: bool,
    mode: Mode,
    scale: float,
    hand_present: bool,
) -> Tuple[List[str], List[tuple[int, int, int]]]:
    """Assemble the HUD status lines and their per-line colours.

    The first three lines are the Day 2 telemetry block (FPS, pinch, palm).
    A hand-independent ``State`` line follows, coloured to show whether the
    reading is live, grabbed or stale. The last two lines report the
    interaction mode and the current model scale.

    Parameters
    ----------
    smoothed_fps : float
        Exponentially smoothed frames per second.
    pinch_ratio : float | None
        Current thumb-to-index gap normalised by hand size, or ``None``.
    palm_center : tuple[float, float] | None
        Raw normalised palm centre, or ``None``.
    smoothed_palm : tuple[float, float] | None
        Smoothed normalised palm centre, or ``None``.
    grabbed : bool
        Whether a hand is currently pinched.
    mode : Mode
        Current interaction mode.
    scale : float
        Current model scale.
    hand_present : bool
        Whether the landmarker reported a hand this frame.

    Returns
    -------
    lines : list[str]
        HUD strings, top to bottom.
    colors : list[tuple[int, int, int]]
        One BGR colour per line.
    """
    state_text = "State: GRABBED" if grabbed else "State: IDLE"
    state_colour = (
        HUD_STATE_GRAB_COLOR if grabbed else HUD_STATE_HOLD_COLOR
    ) if not hand_present else (
        HUD_STATE_GRAB_COLOR if grabbed else HUD_STATE_IDLE_COLOR
    )

    if hand_present:
        lines = [
            f"FPS: {smoothed_fps:.1f}",
            f"Pinch: {pinch_ratio:.3f}",
            f"Palm: ({smoothed_palm[0]:.3f}, {smoothed_palm[1]:.3f})",
            state_text,
        ]
        colors = [
            HUD_TEXT_COLOR,
            HUD_TEXT_COLOR,
            HUD_TEXT_COLOR,
            state_colour,
        ]
    else:
        lines = [
            f"FPS: {smoothed_fps:.1f}",
            "Pinch: N/A",
            "No hand detected",
            state_text,
        ]
        colors = [
            HUD_TEXT_COLOR,
            HUD_TEXT_COLOR,
            HUD_TEXT_COLOR,
            state_colour,
        ]

    mode_colour = {
        Mode.IDLE: HUD_MODE_IDLE_COLOR,
        Mode.ROTATE: HUD_MODE_ROTATE_COLOR,
        Mode.TRANSFORM: HUD_MODE_TRANSFORM_COLOR,
    }[mode]
    lines.append(f"Mode: {mode.name}")
    colors.append(mode_colour)
    lines.append(f"Scale: {scale:.2f}x")
    colors.append(HUD_SCALE_COLOR)

    return lines, colors


def apply_keyboard(key: int, controller: GestureController) -> None:
    """Apply a single key press to the controller's transform state.

    Arrow keys rotate the model by ``ROTATE_STEP`` degrees, ``+`` and ``-``
    change the uniform scale by ``SCALE_STEP`` clamped to the controller
    bounds, and ``r`` resets the pose. Keys outside this set are ignored.

    Parameters
    ----------
    key : int
        Key code returned by ``cv2.waitKey``.
    controller : GestureController
        Controller whose state is mutated.
    """
    if key == ord("r"):
        controller.reset()
        return

    degrees = math.radians(ROTATE_STEP)
    if key == KEY_LEFT:
        controller.rot_y -= degrees
    elif key == KEY_RIGHT:
        controller.rot_y += degrees
    elif key == KEY_UP:
        controller.rot_x -= degrees
    elif key == KEY_DOWN:
        controller.rot_x += degrees
    elif key == ord("+") or key == ord("="):
        controller.scale = min(controller.SCALE_MAX, controller.scale + SCALE_STEP)
    elif key == ord("-"):
        controller.scale = max(controller.SCALE_MIN, controller.scale - SCALE_STEP)


def run_loop(capture: cv2.VideoCapture, tracker: HandTracker) -> int:
    """Process frames until the user presses the exit key.

    Each frame mirrors the webcam image, detects hands, draws the skeleton
    overlay, projects and draws the 3D cube, and paints a telemetry HUD.
    Gesture input feeds the ``GestureController``; arrow keys, ``+/-`` and
    ``r`` offer the same transforms without a hand.

    Parameters
    ----------
    capture : cv2.VideoCapture
        Open webcam stream.
    tracker : HandTracker
        Initialized hand landmarker.

    Returns
    -------
    int
        Always zero; resource release is handled by the caller.
    """
    logger = logging.getLogger("HoloForge")
    smoothed_fps = FPS_INIT
    last_frame = time.monotonic()

    grab_detector = GrabDetector()
    palm_x_smoother = Smoother()
    palm_y_smoother = Smoother()

    # 3D model and renderer are allocated once, outside the hot loop.
    cube_vertices, cube_edges = make_cube()
    renderer = WireframeRenderer()
    controller = GestureController()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    logger.info("Hand-tracking loop running on camera; press '%s' to exit", chr(EXIT_KEY))

    while True:
        ok, frame = capture.read()
        if not ok:
            logger.warning("Camera stream dropped a frame; stopping loop")
            break

        frame = cv2.flip(frame, 1)
        timestamp_ms = int(time.monotonic() * 1000)

        now = time.monotonic()
        delta = now - last_frame
        measured_fps = 1.0 / delta if delta > 0 else FPS_INIT
        smoothed_fps = FPS_SMOOTHING * smoothed_fps + (1 - FPS_SMOOTHING) * measured_fps
        last_frame = now

        hands = tracker.detect(frame, timestamp_ms)
        overlay = tracker.draw_skeleton(frame, hands)

        # 3D pose follows the gesture controller, projected fresh each frame.
        rot = rotation_matrix(controller.rot_x, controller.rot_y, 0.0)
        points2d = project_points(
            cube_vertices,
            rot,
            controller.scale,
            tuple(controller.offset),
            frame.shape,
        )
        overlay = renderer.draw(overlay, points2d, cube_edges)

        key = cv2.waitKey(1) & 0xFF
        if key == EXIT_KEY:
            logger.info("User requested exit")
            break
        apply_keyboard(key, controller)

        if hands is not None:
            pinch_ratio = tracker.get_pinch_ratio(hands[0])
            grab_detector.update(pinch_ratio)
            palm_center = tracker.get_palm_center(hands[0])
            smoothed_palm = (
                palm_x_smoother.update(palm_center[0]),
                palm_y_smoother.update(palm_center[1]),
            )
            controller.update_one_hand(smoothed_palm, grab_detector.is_grabbed, delta)

            stats = tracker.get_two_hand_stats(hands)
            controller.update_two_hands(stats, delta)

            lines, colors = build_hud(
                smoothed_fps=smoothed_fps,
                pinch_ratio=pinch_ratio,
                palm_center=palm_center,
                smoothed_palm=smoothed_palm,
                grabbed=grab_detector.is_grabbed,
                mode=controller.mode,
                scale=controller.scale,
                hand_present=True,
            )
        else:
            # No hand: the detectors keep their last value so a hand that
            # leaves and returns resumes from the previous state instead of
            # resetting. The State line stays visible, coloured amber to
            # signal that the reading is stale.
            controller.update_two_hands(None, delta)
            lines, colors = build_hud(
                smoothed_fps=smoothed_fps,
                pinch_ratio=None,
                palm_center=None,
                smoothed_palm=None,
                grabbed=grab_detector.is_grabbed,
                mode=controller.mode,
                scale=controller.scale,
                hand_present=False,
            )

        draw_hud(overlay, lines, colors)
        cv2.imshow(WINDOW_NAME, overlay)

    return 0


def main() -> int:
    """Run the real-time hand-tracking pipeline until the user quits.

    Returns
    -------
    int
        Zero on clean shutdown, non-zero if the camera or model could not be
        brought online.
    """
    setup_logging()
    logger = logging.getLogger("HoloForge")

    logger.info("HoloForge v0.1.0 - Initializing...")
    logger.info("[PENDING] Load hand-landmarker model and camera stream")
    logger.info("[PENDING] Initialize 3D projection and rendering surface")
    logger.info("[PENDING] Start gesture command analysis pipeline")

    capture: Optional[cv2.VideoCapture] = None
    tracker: Optional[HandTracker] = None
    try:
        capture = open_webcam()
        tracker = HandTracker()
        logger.info("Hand landmarker and camera stream online")
        return run_loop(capture, tracker)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    raise SystemExit(main())