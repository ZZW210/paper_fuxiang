"""Small, process-local counters; timings are inclusive and may overlap."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import time


class PerformanceCounters:
    def __init__(self):
        self.values = defaultdict(float)

    @contextmanager
    def measure(self, component):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.values[component + "_calls"] += 1
            self.values[component + "_seconds"] += time.perf_counter() - start

    def snapshot(self):
        return dict(self.values)

    def add(self, values):
        for key, value in values.items():
            self.values[key] += value

    def delta(self, before):
        return {k: v - before.get(k, 0) for k, v in self.values.items()}
