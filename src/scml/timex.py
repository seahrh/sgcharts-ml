__all__ = ["Timer"]

import math
import time
from datetime import timedelta
from typing import Optional


class Timer:
    def __init__(self, func=time.perf_counter_ns):
        self.elapsed: Optional[timedelta] = None
        self._func = func
        self._start: Optional[int] = None

    def start(self):
        if self._start is not None:
            raise RuntimeError("Already started")
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError("Not started")
        ns = self._func() - self._start
        self.elapsed = timedelta(microseconds=math.ceil(ns / 1000))
        self._start = None

    @property
    def is_running(self) -> bool:
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
