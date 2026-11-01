"""Confirm the decimation path actually reports decimation on a large scan."""
import logging
import os
import tempfile
import trimesh

from src.analysis.model_analyzer import load_and_prepare_mesh

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s:%(message)s")

mesh = trimesh.creation.icosahedron()
for _ in range(5):  # 20 * 4**5 == 20480 faces
    mesh = mesh.subdivide()
print("original faces:", len(mesh.faces))

directory = tempfile.mkdtemp()
path = os.path.join(directory, "big.obj")
mesh.export(path, file_type="obj")

v, e, fc = load_and_prepare_mesh(path)
print("loaded:", fc, "faces,", e.shape[0], "edges,", v.shape[0], "vertices")