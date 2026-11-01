from scripts.benchmark import run_benchmark
from src.rendering.wireframe_renderer import WireframeRenderer

import numpy as np
import trimesh


def test_forced_lod_selects_requested_density():
    mesh = trimesh.creation.icosphere(subdivisions=5)
    edges = np.asarray(mesh.edges_unique)
    points = np.asarray(mesh.vertices[:, :2]) * 2 + 200
    renderer = WireframeRenderer()
    renderer.prepare_lod(edges)
    renderer.draw(np.zeros((480, 640, 3), dtype=np.uint8), points, edges,
                  force_lod=1200)
    assert renderer.active_edge_count == 1200
    renderer.draw(np.zeros((480, 640, 3), dtype=np.uint8), points, edges,
                  force_lod=5000)
    assert renderer.active_edge_count == 5000


def test_benchmark_uses_environment_frame_count_and_writes_report(tmp_path, monkeypatch):
    monkeypatch.setenv("HOLOFORGE_BENCH_FRAMES", "5")
    output = tmp_path / "BENCHMARK.md"
    report = run_benchmark(output)
    assert output.is_file()
    assert "measured: 5" in report
    assert "icosphere-3" in report and "icosphere-5" in report
