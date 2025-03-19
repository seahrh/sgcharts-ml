__all__ = ["Timer", "days_ago", "last_month"]

import math
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple


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


def days_ago(days: int, reference: Optional[datetime] = None) -> datetime:
    if reference is None:
        reference = datetime.now(timezone.utc)
    return reference - timedelta(days=days)


def last_month(reference: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    if reference is None:
        reference = datetime.now(timezone.utc)
    last_day_of_month = reference.replace(day=1) - timedelta(days=1)
    first_day_of_month = last_day_of_month.replace(day=1)
    return first_day_of_month, last_day_of_month
