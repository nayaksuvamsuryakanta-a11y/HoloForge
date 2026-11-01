"""Short synthesized sound cues with a silent fallback."""

from __future__ import annotations

import io
import logging
import math
import wave
from typing import Callable

import numpy as np

logger = logging.getLogger("HoloForge")
VOLUME = 0.5
SAMPLE_RATE = 44100


def _wav(frequencies: tuple[float, ...], duration: float, envelope: str) -> io.BytesIO:
    count = int(SAMPLE_RATE * duration)
    samples = np.zeros(count, dtype=np.float64)
    boundaries = np.linspace(0, count, len(frequencies) + 1, dtype=int)
    for index, frequency in enumerate(frequencies):
        start, end = boundaries[index:index + 2]
        t = np.arange(end - start) / SAMPLE_RATE
        if envelope == "chirp":
            end_frequency = frequencies[-1]
            phase = 2 * math.pi * (
                frequencies[0] * t
                + 0.5 * (end_frequency - frequencies[0]) * t * t / duration
            )
        elif len(frequencies) == 1:
            phase = 2 * math.pi * frequency * t
        else:
            total = max(duration / len(frequencies), 1 / SAMPLE_RATE)
            phase = 2 * math.pi * frequency * t
        samples[start:end] = np.sin(phase)
    if envelope == "raised":
        ramp = int(0.005 * SAMPLE_RATE)
        if ramp:
            curve = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, ramp))
            samples[:ramp] *= curve
            samples[-ramp:] *= curve[::-1]
    elif envelope == "linear":
        samples *= np.linspace(1.0, 0.0, count)
    pcm = np.clip(samples * 32767, -32768, 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm)
    buffer.seek(0)
    return buffer


class _SilentSound:
    def play(self) -> None:
        return None

    def set_volume(self, volume: float) -> None:
        return None


class _SilentMixer:
    def Sound(self, file) -> _SilentSound:
        return _SilentSound()

    def quit(self) -> None:
        return None


class SoundBoard:
    """Own synthesized cues and degrade to silence if audio is unavailable."""

    def __init__(self, mixer_factory: Callable | None = None) -> None:
        self.muted = False
        self._mixer = None
        self._sounds = {}
        try:
            if mixer_factory is None:
                import pygame
                mixer_factory = lambda: pygame.mixer
            self._mixer = mixer_factory()
            self._mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1)
            definitions = {
                "grab": ((880,), 0.04, "raised"),
                "release": ((660, 440), 0.06, "chirp"),
                "export": ((523, 784), 0.12, "hard"),
                "error": ((180,), 0.09, "hard"),
            }
            self._sounds = {
                name: self._mixer.Sound(file=_wav(*spec))
                for name, spec in definitions.items()
            }
            for sound in self._sounds.values():
                sound.set_volume(VOLUME)
        except Exception as exc:  # audio must never prevent interaction
            self._mixer = _SilentMixer()
            self._sounds = {name: _SilentSound() for name in ("grab", "release", "export", "error")}
            logger.warning("Audio unavailable; continuing silently: %s", exc)

    def play(self, cue: str) -> None:
        if not self.muted and cue in self._sounds:
            self._sounds[cue].play()

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        return self.muted

    def close(self) -> None:
        if self._mixer is not None:
            self._mixer.quit()
            self._mixer = None
