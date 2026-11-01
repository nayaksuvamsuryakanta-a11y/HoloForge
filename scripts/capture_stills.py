"""Generate canonical demonstration frames without camera or display access."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import trimesh

from src.interaction.gesture_controller import GestureController, Mode
from src.interaction.vertex_measure import VertexMeasure
from src.main import _draw_measurement, build_hud, draw_hud
from src.rendering.projection_3d import project_points, rotation_matrix
from src.rendering.wireframe_renderer import WireframeRenderer


def capture(output_dir: str | Path = "output/stills") -> list[Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=(150, 150, 150))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    edges = np.asarray(mesh.edges_unique, dtype=np.intp)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    renderer = WireframeRenderer(holographic=True)
    renderer.prepare_lod(edges)
    metrics = {
        "volume": 3375000.0, "surface_area": 135000.0,
        "extents": np.array([150.0, 150.0, 150.0]),
        "center_mass": np.zeros(3), "is_watertight": True,
        "is_winding_consistent": True, "estimated_mass_g": 4185.0,
    }
    states = ["idle", "grabbed_rotate", "transform_zoom", "analysis_on",
              "exploded_full", "frozen_measurement"]
    outputs = []
    for state in states:
        controller = GestureController()
        grabbed, analysis, explode, measure = False, False, 0.0, VertexMeasure()
        if state == "grabbed_rotate":
            controller.rot_x, controller.rot_y = 0.35, 0.55
            controller.mode, grabbed = Mode.ROTATE, True
        elif state == "transform_zoom":
            controller.scale, controller.mode = 1.6, Mode.TRANSFORM
            controller.offset[:] = (35, -15)
        elif state == "analysis_on":
            analysis = True
        elif state == "exploded_full":
            explode = 1.0
        elif state == "frozen_measurement":
            measure.begin(0)
            measure.update(6)
            measure.release()
        shown = vertices + normals * (40.0 * explode)
        rotation = rotation_matrix(controller.rot_x, controller.rot_y, 0.0)
        points = project_points(shown, rotation, controller.scale,
                                tuple(controller.offset), (480, 640, 3))
        depth = (shown @ rotation.T)[:, 2] + 1500
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (18, 20, 27)
        frame = renderer.draw(frame, points, edges, depth=depth)
        _draw_measurement(frame, points, measure, vertices)
        lines, colors = build_hud(
            35.0, 0.22 if grabbed else None, (0.5, 0.5) if grabbed else None,
            (0.5, 0.5) if grabbed else None, grabbed, controller.mode,
            controller.scale, grabbed, "demo cube", len(mesh.faces), edges.shape[0],
            renderer.active_edge_count, renderer.mean_edge_length, metrics, analysis,
            explode_f=explode, voice_status="off", measurement_count=measure.count,
        )
        draw_hud(frame, lines, colors)
        path = target / f"{state}.png"
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"Could not write {path}")
        outputs.append(path)
    return outputs


if __name__ == "__main__":
    for path in capture():
        print(path)
