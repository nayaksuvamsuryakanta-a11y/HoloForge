import json

import numpy as np

from src.interaction.vertex_measure import VertexMeasure
from src.rendering.projection_3d import make_cube
from src.utils.config import DEFAULTS, load_config
from src.utils.telemetry import SessionTelemetry


def test_measurement_uses_base_vertices_and_counts_frozen_pairs():
    vertices, _ = make_cube()
    measure = VertexMeasure()
    measure.begin(0)
    measure.update(6)
    expected = float(np.linalg.norm(vertices[6] - vertices[0]))
    offset = np.ones_like(vertices) * 40
    assert measure.distance(vertices) == expected
    assert measure.distance(vertices) == measure.distance(vertices + offset)
    measure.release()
    assert measure.count == 1
    measure.begin(1)
    measure.update(7)
    measure.release()
    assert measure.count == 2
    measure.clear()
    assert measure.count == 0


def test_vertex_pick_radius_is_inclusive():
    measure = VertexMeasure(radius=12)
    points = np.array([[10.0, 10.0], [30.0, 30.0]])
    assert measure.nearest(points, (22.0, 10.0)) == 0
    assert measure.nearest(points, (22.01, 10.0)) is None


def test_telemetry_header_ring_buffer_and_flush_cadence(tmp_path):
    telemetry = SessionTelemetry("cube", tmp_path, flush_every=2, buffer_size=2)
    assert telemetry.toggle()
    path = next(tmp_path.glob("session_*.jsonl"))
    telemetry.record({"t": 0, "mode": "IDLE"})
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    telemetry.record({"t": 1, "mode": "ROTATE"})
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3
    telemetry.record({"t": 2, "mode": "IDLE"})
    telemetry.stop()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["schema_version"] == 1
    assert rows[0]["model"] == "cube"
    assert "started_iso" in rows[0]
    assert [row["t"] for row in rows[1:]] == [0, 1, 2]
    assert len(telemetry.records) == 2


def test_config_validation_defaults_and_cli_precedence(tmp_path, caplog):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "grab_threshold": 0.7,
        "release_threshold": 0.6,
        "smoothing_alpha": 0,
        "rotation_sensitivity": -1,
        "pan_sensitivity": 2.5,
    }), encoding="utf-8")
    values = load_config(path, {"rotation_sensitivity": 4.0})
    assert values["grab_threshold"] == DEFAULTS["grab_threshold"]
    assert values["release_threshold"] == DEFAULTS["release_threshold"]
    assert values["smoothing_alpha"] == DEFAULTS["smoothing_alpha"]
    assert values["rotation_sensitivity"] == 4.0
    assert values["pan_sensitivity"] == 2.5
    assert "smoothing_alpha" in caplog.text
    assert "rotation_sensitivity" in caplog.text
