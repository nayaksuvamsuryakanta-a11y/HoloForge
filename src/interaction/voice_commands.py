"""Optional speech command worker; results are drained by the UI thread."""

from __future__ import annotations

import importlib.util
import logging
import threading
import time
from collections import deque
from typing import Callable

logger = logging.getLogger("HoloForge")
COMMANDS = {"analyze", "export", "reset", "explode", "quit"}


class VoiceLayer:
    def __init__(self, enabled: bool = False, recognizer_factory: Callable | None = None,
                 microphone_factory: Callable | None = None) -> None:
        self.enabled = bool(enabled)
        self._recognizer_factory = recognizer_factory
        self._microphone_factory = microphone_factory
        self._commands: deque[str] = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failures = 0
        self._last_command = ""
        self._last_time = 0.0
        self._warned = False
        if self.enabled:
            if microphone_factory is None and importlib.util.find_spec("pyaudio") is None:
                self._disable("Voice unavailable: install optional pyaudio")
            else:
                self._thread = threading.Thread(target=self._listen_loop,
                                                name="HoloForgeVoice", daemon=True)
                self._thread.start()

    def _disable(self, reason: str) -> None:
        self.enabled = False
        if not self._warned:
            logger.warning(reason)
            self._warned = True

    def _listen_loop(self) -> None:
        try:
            import speech_recognition as sr
            recognizer = (self._recognizer_factory or sr.Recognizer)()
            recognizer.dynamic_energy_threshold = False
            microphone = self._microphone_factory or sr.Microphone
        except Exception as exc:
            self._disable(f"Voice unavailable: {exc}")
            return
        while not self._stop.is_set() and self.enabled:
            try:
                with microphone() as source:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=2)
                command = recognizer.recognize_google(audio).strip().lower()
                if command not in COMMANDS:
                    continue
                now = time.monotonic()
                if command == self._last_command and now - self._last_time < 2.0:
                    continue
                self._last_command, self._last_time = command, now
                with self._lock:
                    self._commands.append(command)
                self._failures = 0
            except Exception:
                self._failures += 1
                if self._failures >= 5:
                    self._disable("Voice disabled after five consecutive recognition failures")

    def drain(self) -> list[str]:
        with self._lock:
            commands = list(self._commands)
            self._commands.clear()
        return commands

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.25)

    @property
    def status(self) -> str:
        return "on" if self.enabled else "off"
