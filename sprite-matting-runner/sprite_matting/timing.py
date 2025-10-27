"""Timing helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class Timer:
    """Collects named duration measurements."""

    records: Dict[str, float] = field(default_factory=dict)

    @contextmanager
    def time(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.records[name] = self.records.get(name, 0.0) + (time.perf_counter() - start)

    def add(self, name: str, value: float) -> None:
        self.records[name] = self.records.get(name, 0.0) + value

    def get(self, name: str) -> float:
        return self.records.get(name, 0.0)


@contextmanager
def stopwatch() -> Callable[[], float]:
    start = time.perf_counter()
    duration = [0.0]

    try:
        yield lambda: duration[0]
    finally:
        duration[0] = time.perf_counter() - start
