import time
import wave
import numpy as np
import trimesh

from src.interaction.voice_commands import VoiceLayer
from src.utils.audio_feedback import SAMPLE_RATE, SoundBoard, _wav
from src.analysis.model_analyzer import export_transformed_mesh, load_prepared_mesh


def test_synthesized_cues_have_requested_pcm_duration_and_rate():
    for freqs, seconds, envelope in [
        ((880,), 0.04, "raised"),
        ((660, 440), 0.06, "chirp"),
        ((523, 784), 0.12, "hard"),
        ((180,), 0.09, "hard"),
    ]:
        stream = _wav(freqs, seconds, envelope)
        with wave.open(stream, "rb") as wav:
            assert wav.getframerate() == SAMPLE_RATE
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getnframes() == round(SAMPLE_RATE * seconds)


def test_soundboard_uses_silent_fallback_once_on_mixer_error(caplog):
    board = SoundBoard(mixer_factory=lambda: (_ for _ in ()).throw(RuntimeError("no device")))
    assert "Audio unavailable" in caplog.text
    board.play("grab")
    assert board.toggle_mute()
    board.play("export")
    board.close()


class _Microphone:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Recognizer:
    def __init__(self):
        self.dynamic_energy_threshold = True
        self.calls = 0

    def listen(self, source, timeout, phrase_time_limit):
        assert (timeout, phrase_time_limit) == (5, 2)
        self.calls += 1
        if self.calls > 2:
            raise RuntimeError("end fake stream")
        return object()

    def recognize_google(self, audio):
        return "explode"


def test_voice_maps_commands_debounces_and_disables_after_failures():
    recognizer = _Recognizer()
    voice = VoiceLayer(True, lambda: recognizer, _Microphone)
    deadline = time.monotonic() + 2
    while voice.enabled and time.monotonic() < deadline:
        time.sleep(0.01)
    assert recognizer.dynamic_energy_threshold is False
    assert voice.drain() == ["explode"]
    voice.stop()


def test_voice_disabled_is_inert():
    voice = VoiceLayer(False)
    assert voice.status == "off"
    assert voice.drain() == []
    voice.stop()


def test_prepared_mesh_normals_drive_monotonic_explode_and_export_ignores_it(tmp_path):
    source = trimesh.creation.icosphere(subdivisions=1, radius=2)
    path = tmp_path / "sphere.obj"
    source.export(path)
    mesh, _, _ = load_prepared_mesh(str(path))
    base = np.asarray(mesh.vertices).copy()
    offset = np.asarray(mesh.vertex_normals) * 40.0
    radii = [np.linalg.norm(base + offset * f, axis=1).mean() for f in (0, 0.5, 1)]
    assert radii[0] < radii[1] < radii[2]
    output = export_transformed_mesh(source, 0, 0, 1, tmp_path / "out")
    exported = trimesh.load(output, force="mesh")
    assert np.allclose(np.sort(exported.vertices, axis=0), np.sort(source.vertices, axis=0))
