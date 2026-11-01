"""Selection and distance state for vertex-to-vertex measurements."""

from __future__ import annotations

import numpy as np

PICK_RADIUS = 12.0


class VertexMeasure:
    def __init__(self, radius: float = PICK_RADIUS) -> None:
        self.radius = float(radius)
        self.highlight: int | None = None
        self.anchor: int | None = None
        self.target: int | None = None
        self.active = False
        self.measurements: list[tuple[int, int]] = []

    def nearest(self, points2d: np.ndarray, cursor: tuple[float, float]) -> int | None:
        if points2d.size == 0:
            return None
        distances = np.linalg.norm(points2d - np.asarray(cursor), axis=1)
        index = int(np.argmin(distances))
        return index if distances[index] <= self.radius else None

    def begin(self, index: int) -> None:
        self.anchor = self.target = int(index)
        self.active = True

    def update(self, index: int | None) -> None:
        if self.active and index is not None:
            self.target = int(index)

    def release(self) -> None:
        if self.active and self.anchor is not None and self.target is not None:
            self.measurements.append((self.anchor, self.target))
        self.active = False

    def clear(self) -> None:
        self.highlight = self.anchor = self.target = None
        self.active = False
        self.measurements.clear()

    def distance(self, base_vertices: np.ndarray) -> float | None:
        if self.anchor is None or self.target is None:
            return None
        return float(np.linalg.norm(base_vertices[self.target] - base_vertices[self.anchor]))

    @property
    def count(self) -> int:
        return len(self.measurements) + int(self.active)
