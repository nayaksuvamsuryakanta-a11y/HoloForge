"""Entry point for the HoloForge real-time hand-tracking pipeline.

Day 2 drives a live webcam loop that detects hands, draws the skeleton
overlay and reports a smoothed FPS, a pinch ratio and a palm position. Day 5
adds a 3D wireframe cube that the hand gestures manipulate, with a keyboard
fallback for the same transforms. Days 6-7 bring grab-rotate with spin-and-catch
inertia, and Day 8 adds two-hand zoom/pan. Press ``q`` to shut down.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.analysis.model_analyzer import (
    compute_metrics,
    export_transformed_mesh,
    load_and_prepare_mesh,
    load_source_mesh,
)
from src.interaction.gesture_controller import GestureController, Mode
from src.rendering.projection_3d import MIN_Z, Z_OFFSET, make_cube, project_points, rotation_matrix
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

# Colours for the imported-model lines of the HUD.
HUD_MODEL_NAME_COLOR = (200, 255, 255)  # BGR pale cyan
HUD_MODEL_COUNT_COLOR = (180, 180, 180)  # BGR light gray

# FPS estimation ------------------------------------------------------------
# Weight given to the previous smoothed value; a high smoothing factor keeps
# stable readings on a laptop webcam whose per-frame latency jitters.
FPS_SMOOTHING = 0.9
FPS_INIT = 30.0

# Stage timing --------------------------------------------------------------
# Exponential smoothing applied to each stage's latency, in milliseconds.
INFERENCE_SMOOTHING = 0.95
RENDER_SMOOTHING = 0.95
TIMING_LOG_INTERVAL = 5.0  # seconds between stage-latency log lines

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
    model_name: str,
    face_count: int,
    edge_count: int,
    lod_edges: int = 0,
    lod_mean_len: float = 0.0,
    analysis_metrics: Optional[dict] = None,
    analysis_visible: bool = False,
) -> Tuple[List[str], List[tuple[int, int, int]]]:
    """Assemble the HUD status lines and their per-line colours.

    The first three lines are the Day 2 telemetry block (FPS, pinch, palm).
    A hand-independent ``State`` line follows, coloured to show whether the
    reading is live, grabbed or stale. The next two lines report the
    interaction mode and the current model scale. When an external mesh is
    loaded, the final block names the file and reports how much of its
    geometry survived decimation.

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
    model_name : str
        Display name of the loaded model, ``"cube"`` when none was supplied.
    face_count : int
        Original face count of the loaded mesh, ``0`` when the cube was rendered.
    edge_count : int
        Number of edges actually drawn this frame.
    lod_edges : int
        Number of edges rendered after Level-of-Detail selection, ``0`` when
        the mesh was too small to build a ladder.
    lod_mean_len : float
        Mean projected length of the prepared edges, in pixels.

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

    lines.append(f"Model: {model_name}")
    colors.append(HUD_MODEL_NAME_COLOR)
    lines.append(f"Faces: {face_count:,}" if face_count > 0 else "Faces: N/A")
    colors.append(HUD_MODEL_COUNT_COLOR)
    lines.append(f"Rendered Edges: {edge_count:,}")
    colors.append(HUD_MODEL_COUNT_COLOR)
    lines.append(f"LOD: {lod_edges} @ {lod_mean_len:.1f}px")
    colors.append(HUD_MODEL_COUNT_COLOR)

    if analysis_visible and analysis_metrics is not None:
        volume = analysis_metrics["volume"]
        lines.append(
            "Volume: N/A (open mesh)" if volume is None else f"Volume: {volume:.2f} mm³"
        )
        colors.append(HUD_MODEL_COUNT_COLOR)
        lines.append(f"Surface Area: {analysis_metrics['surface_area']:.2f} mm²")
        colors.append(HUD_MODEL_COUNT_COLOR)
        x_size, y_size, z_size = analysis_metrics["extents"]
        lines.append(f"Dimensions (X, Y, Z): {x_size:.2f}, {y_size:.2f}, {z_size:.2f} mm")
        colors.append(HUD_MODEL_COUNT_COLOR)
        watertight = analysis_metrics["is_watertight"]
        lines.append(f"Watertight: {'YES' if watertight else 'NO'}")
        colors.append(HUD_MODEL_COUNT_COLOR if watertight else HUD_STATE_GRAB_COLOR)
        winding = analysis_metrics["is_winding_consistent"]
        lines.append(f"Winding: {'OK' if winding else 'FAIL'}")
        colors.append(HUD_MODEL_COUNT_COLOR)
        mass = analysis_metrics["estimated_mass_g"]
        lines.append("Estimated PLA mass: N/A" if mass is None else f"Estimated PLA mass: {mass:.2f} g")
        colors.append(HUD_MODEL_COUNT_COLOR)

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


def _load_model(model_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray, str, int, int]:
    """Load an external mesh, falling back to the default cube.

    Parameters
    ----------
    model_path : str | None
        Path to a mesh file, or ``None`` to render the cube.

    Returns
    -------
    vertices : np.ndarray
        ``(n, 3)`` model-space vertex positions.
    edges : np.ndarray
        ``(m, 2)`` vertex-index pairs, capped to a few thousand edges.
    model_name : str
        Display name for the HUD.
    face_count : int
        Original face count of the mesh, ``0`` when the cube was rendered.
    edge_count : int
        Number of edges that will be drawn.
    """
    logger = logging.getLogger("HoloForge")
    if model_path is None:
        logger.info("No model supplied; rendering the default cube")
        vertices, edges = make_cube()
        return vertices, edges, "cube", 0, edges.shape[0]

    display_name = os.path.basename(model_path)
    vertices, edges, face_count = load_and_prepare_mesh(model_path)
    logger.info(
        "Rendering %s: %d original faces -> %d edges (%d vertices)",
        display_name,
        face_count,
        edges.shape[0],
        vertices.shape[0],
    )
    return vertices, edges, display_name, face_count, edges.shape[0]


def _inference_stage(
    tracker: HandTracker,
    frame: np.ndarray,
    timestamp_ms: int,
    frame_index: int,
    frame_skip: int,
    cached_hands,
):
    """Run or reuse hand detection for one frame."""
    if frame_index % frame_skip == 0:
        cached_hands = tracker.detect(frame, timestamp_ms)
    return cached_hands, cached_hands


def _render_stage(
    frame: np.ndarray,
    renderer: WireframeRenderer,
    model_vertices: np.ndarray,
    model_edges: np.ndarray,
    controller: GestureController,
) -> np.ndarray:
    """Project and draw the model for one frame, without UI operations."""
    rot = rotation_matrix(controller.rot_x, controller.rot_y, 0.0)
    transformed = (model_vertices @ rot.T) * controller.scale
    depth = transformed[:, 2] + Z_OFFSET
    depth = np.where(depth < MIN_Z, MIN_Z, depth)
    points2d = project_points(
        model_vertices,
        rot,
        controller.scale,
        tuple(controller.offset),
        frame.shape,
        transformed,
    )
    return renderer.draw(frame, points2d, model_edges, depth=depth)


def run_loop(
    capture: cv2.VideoCapture,
    tracker: HandTracker,
    model_path: Optional[str] = None,
    frame_skip: int = 1,
) -> int:
    """Process frames until the user presses the exit key.

    Each frame mirrors the webcam image, detects hands, draws the skeleton
    overlay, projects and draws the 3D model, and paints a telemetry HUD.
    Gesture input feeds the ``GestureController``; arrow keys, ``+/-`` and
    ``r`` offer the same transforms without a hand.

    Parameters
    ----------
    capture : cv2.VideoCapture
        Open webcam stream.
    tracker : HandTracker
        Initialized hand landmarker.
    model_path : str | None
        Path to an external mesh to render, or ``None`` for the default cube.
    frame_skip : int, default 1
        Run the landmarker every ``frame_skip`` frames and reuse the cached
        landmarks on the frames in between; 2 halves the inference cost.

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

    # Stage latencies and frame-skip state, tracked across the loop.
    inference_ema = 0.0
    render_ema = 0.0
    last_timing = time.perf_counter()
    frame_index = 0
    cached_hands = None

    # Model, renderer and controller are allocated once, outside the hot loop.
    model_vertices, model_edges, model_name, face_count, edge_count = _load_model(model_path)
    model_mesh = load_source_mesh(model_path) if model_path is not None else None
    analysis_metrics = compute_metrics(model_mesh) if model_mesh is not None else None
    renderer = WireframeRenderer(holographic=model_path is not None)
    renderer.prepare_lod(model_edges)
    controller = GestureController()
    analysis_visible = False
    exported_message: Optional[str] = None
    exported_message_until = 0.0

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

        # Inference stage: the landmarker. On a skipped frame the previous detection
        # is reused, so the EMA genuinely records the work saved by frame skip.
        t0 = time.perf_counter()
        hands, cached_hands = _inference_stage(
            tracker, frame, timestamp_ms, frame_index, frame_skip, cached_hands
        )
        inference_ema = INFERENCE_SMOOTHING * inference_ema + (1 - INFERENCE_SMOOTHING) * (
            time.perf_counter() - t0
        ) * 1000.0

        overlay = tracker.draw_skeleton(frame, hands)

        # Render stage: projection plus the wireframe pass.
        t0 = time.perf_counter()
        overlay = _render_stage(overlay, renderer, model_vertices, model_edges, controller)
        render_ema = RENDER_SMOOTHING * render_ema + (1 - RENDER_SMOOTHING) * (
            time.perf_counter() - t0
        ) * 1000.0

        key = cv2.waitKey(1) & 0xFF
        if key == EXIT_KEY:
            logger.info("User requested exit")
            break
        if key == ord("a"):
            analysis_visible = not analysis_visible
        elif key == ord("e"):
            if model_mesh is None:
                logger.info("Export requires --model")
                exported_message = None
            else:
                export_path = export_transformed_mesh(
                    model_mesh,
                    controller.rot_x,
                    controller.rot_y,
                    controller.scale,
                    "output",
                )
                exported_message = f"Exported: {export_path.name}"
                exported_message_until = time.monotonic() + 2.0
        else:
            apply_keyboard(key, controller)

        # Stage latencies are logged on a separate cadence from the FPS so the
        # console is not flooded by a per-frame number.
        if time.perf_counter() - last_timing >= TIMING_LOG_INTERVAL:
            logger.info(
                "Stage latencies (%.0f s EMA): inference %.2f ms, render %.2f ms",
                TIMING_LOG_INTERVAL,
                inference_ema,
                render_ema,
            )
            last_timing = time.perf_counter()

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
                model_name=model_name,
                face_count=face_count,
                edge_count=edge_count,
                lod_edges=renderer.active_edge_count,
                lod_mean_len=renderer.mean_edge_length,
                analysis_metrics=analysis_metrics,
                analysis_visible=analysis_visible,
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
                model_name=model_name,
                face_count=face_count,
                edge_count=edge_count,
                lod_edges=renderer.active_edge_count,
                lod_mean_len=renderer.mean_edge_length,
                analysis_metrics=analysis_metrics,
                analysis_visible=analysis_visible,
            )

        if exported_message and time.monotonic() < exported_message_until:
            lines.append(exported_message)
            colors.append(HUD_MODEL_NAME_COLOR)
        elif time.monotonic() >= exported_message_until:
            exported_message = None

        draw_hud(overlay, lines, colors)
        cv2.imshow(WINDOW_NAME, overlay)
        frame_index += 1

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the real-time hand-tracking pipeline until the user quits.

    Parameters
    ----------
    argv : sequence[str] | None
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Zero on clean shutdown, non-zero if the camera or model could not be
        brought online.
    """
    parser = argparse.ArgumentParser(
        description="HoloForge: hand-tracked 3D wireframe overlay."
    )
    parser.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help="Photogrammetry scan (.obj, .ply, .stl ...) to render instead of "
             "the default cube. Large scans are decimated to a few thousand "
             "edges automatically.",
    )
    parser.add_argument(
        "--frame-skip",
        metavar="N",
        type=int,
        default=1,
        help="Run hand inference every N frames (default: 1).",
    )
    args = parser.parse_args(argv)
    if args.frame_skip < 1:
        parser.error("--frame-skip must be at least 1")

    setup_logging()
    logger = logging.getLogger("HoloForge")

    logger.info("HoloForge v1.0.0 - Initializing...")
    logger.info("Phase 1: Vision pipeline [COMPLETE]")
    logger.info("Phase 2: 3D rendering engine [COMPLETE]")
    logger.info("Phase 3: Model analysis [COMPLETE]")

    capture: Optional[cv2.VideoCapture] = None
    tracker: Optional[HandTracker] = None
    try:
        capture = open_webcam()
        tracker = HandTracker()
        logger.info("Hand landmarker and camera stream online")
        return run_loop(capture, tracker, model_path=args.model, frame_skip=args.frame_skip)
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
