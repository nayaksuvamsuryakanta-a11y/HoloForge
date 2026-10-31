"""Entry point for the HoloForge real-time hand-tracking pipeline.

Day 2 replaces the bootstrap banner with a live webcam loop that opens a
built-in laptop camera, detects hands with the MediaPipe landmarker, draws
the skeleton overlay and reports a smoothed FPS, a pinch ratio and a palm
position in a top-left HUD. Press ``q`` to shut down.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2

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

# FPS estimation ------------------------------------------------------------
# Weight given to the previous smoothed value; a high smoothing factor keeps
# stable readings on a laptop webcam whose per-frame latency jitters.
FPS_SMOOTHING = 0.9
FPS_INIT = 30.0

# Window --------------------------------------------------------------------
WINDOW_NAME = "HoloForge - Hand Tracking"
EXIT_KEY = ord("q")


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


def run_loop(capture: cv2.VideoCapture, tracker: HandTracker) -> int:
    """Process frames until the user presses the exit key.

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

        landmarks = tracker.detect(frame, timestamp_ms)
        overlay = tracker.draw_skeleton(frame, landmarks)

        if landmarks is not None:
            pinch_ratio = tracker.get_pinch_ratio(landmarks)
            grab_detector.update(pinch_ratio)
            palm_center = tracker.get_palm_center(landmarks)
            smoothed_palm = (
                palm_x_smoother.update(palm_center[0]),
                palm_y_smoother.update(palm_center[1]),
            )
            state_text = "State: GRABBED" if grab_detector.is_grabbed else "State: IDLE"
            state_colour = (
                HUD_STATE_GRAB_COLOR if grab_detector.is_grabbed else HUD_STATE_IDLE_COLOR
            )
            lines = [
                f"FPS: {smoothed_fps:.1f}",
                f"Pinch: {pinch_ratio:.3f}",
                f"Palm: ({smoothed_palm[0]:.3f}, {smoothed_palm[1]:.3f})",
                state_text,
            ]
            colors = [HUD_TEXT_COLOR, HUD_TEXT_COLOR, HUD_TEXT_COLOR, state_colour]
        else:
            # No hand: the detectors keep their last value so a hand that
            # leaves and returns resumes from the previous state instead of
            # resetting. The State line stays visible, coloured amber to
            # signal that the reading is stale.
            state_text = "State: GRABBED" if grab_detector.is_grabbed else "State: IDLE"
            lines = [
                f"FPS: {smoothed_fps:.1f}",
                "Pinch: N/A",
                "No hand detected",
                state_text,
            ]
            colors = [HUD_TEXT_COLOR, HUD_TEXT_COLOR, HUD_TEXT_COLOR, HUD_STATE_HOLD_COLOR]

        draw_hud(overlay, lines, colors)
        cv2.imshow(WINDOW_NAME, overlay)

        if cv2.waitKey(1) & 0xFF == EXIT_KEY:
            logger.info("User requested exit")
            break

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
