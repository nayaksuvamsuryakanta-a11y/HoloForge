"""Frame-synchronous JSONL session telemetry."""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


class SessionTelemetry:
    def __init__(self, model: str, output_dir: str | Path = "output",
                 flush_every: int = 150, buffer_size: int = 150) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.flush_every = int(flush_every)
        self.records = deque(maxlen=buffer_size)
        self._pending: list[dict] = []
        self._active = False
        self._path: Path | None = None
        self._frames = 0

    @property
    def active(self) -> bool:
        return self._active

    def toggle(self) -> bool:
        if self._active:
            self.stop()
            return False
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._path = self.output_dir / f"session_{stamp}.jsonl"
        header = {"schema_version": 1, "model": self.model,
                  "started_iso": datetime.now(timezone.utc).isoformat()}
        self._path.write_text(json.dumps(header) + "\n", encoding="utf-8")
        self._active = True
        self._frames = 0
        return True

    def record(self, record: dict) -> None:
        if not self._active:
            return
        self.records.append(dict(record))
        self._pending.append(dict(record))
        self._frames += 1
        if self._frames % self.flush_every == 0:
            self.flush()

    def flush(self) -> None:
        if self._path is None or not self._pending:
            return
        with self._path.open("a", encoding="utf-8") as output:
            output.writelines(json.dumps(row) + "\n" for row in self._pending)
        self._pending.clear()

    def stop(self) -> None:
        self.flush()
        self._active = False

