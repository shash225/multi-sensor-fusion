"""Per-stage latency measurement (CUDA-synchronised so the timings are real)."""
from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

import torch


class StageTimer:
    """Collect wall-clock time per named pipeline stage.

    ``callback(stage_name, ms)`` runs after every stage; the dashboard uses it to tick
    off pipeline stages live while a frame is being processed.
    """

    def __init__(self, enabled: bool = True, sync_cuda: bool = True,
                 callback: Optional[Callable[[str, float], None]] = None) -> None:
        self.enabled = enabled
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.callback = callback
        self.times: "OrderedDict[str, float]" = OrderedDict()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        if self.sync_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if self.sync_cuda:
            torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000.0
        self.times[name] = self.times.get(name, 0.0) + ms
        if self.callback is not None:
            self.callback(name, ms)

    @property
    def total_ms(self) -> float:
        return float(sum(self.times.values()))

    def as_dict(self) -> dict[str, float]:
        return dict(self.times)


class NullTimer(StageTimer):
    """A timer that records nothing (used in training)."""

    def __init__(self) -> None:
        super().__init__(enabled=False)
