"""Smoke test the LOD ladder on a subdivided icosphere and a decimated scan."""
import os
import tempfile

import numpy as np
import trimesh

from src.analysis.model_analyzer import load_and_prepare_mesh
from src.rendering.projection_3d import make_cube, project_points, rotation_matrix
from src.rendering.wireframe_renderer import WireframeRenderer, build_ladder

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def project_mesh(vertices, rot, scale, offset=(0.0, 0.0)):
    transformed = (vertices @ rot.T) * scale
    depth = transformed[:, 2] + 1000.0
    points = project_points(vertices, rot, scale, offset, FRAME.shape, transformed)
    return points, depth


def sweep(renderer, vertices, edges, label):
    rot = rotation_matrix(0.3, 0.5, 0.0)
    print(f"--- {label}: {edges.shape[0]} prepared edges ---")
    for scale in (0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0, 4.0):
        points, depth = project_mesh(vertices, rot, scale)
        renderer.draw(FRAME, points, edges, depth=depth)
        print(f"  scale={scale:<5} active={renderer.active_edge_count:<6} "
              f"mean={renderer.mean_edge_length:7.2f}px")


ico = trimesh.creation.icosahedron()
for _ in range(3):
    ico = ico.subdivide()

# raw icosphere: 1280 edges, ladder capped at 1280/600
sweep(WireframeRenderer(holographic=True), ico.vertices, np.asarray(ico.edges_unique), "raw icosphere (1280 faces)")

# decimated scan path
directory = tempfile.mkdtemp()
path = os.path.join(directory, "icosphere.obj")
ico.export(path, file_type="obj")
v, e, fc = load_and_prepare_mesh(path)
print("loaded:", fc, "faces,", e.shape[0], "edges,", v.shape[0], "vertices")
sweep(WireframeRenderer(holographic=True), v, e, "prepared scan")

# ladder subset check
idx, sub = build_ladder(e)
print("ladder:", [s.shape[0] for s in sub],
      "| strict subsets:", all(
          set(map(tuple, sub[i])) < set(map(tuple, sub[i - 1])) for i in range(1, len(sub))))