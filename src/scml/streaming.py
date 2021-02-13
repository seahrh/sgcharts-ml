__all__ = ["RollingWindow", "IterativeMean"]

from collections import deque
from typing import Deque, Optional


class RollingWindow:
    def __init__(self, capacity: int):
        self._capacity: int = capacity
        self._sum: float = 0
        self._buf: Deque[float] = deque()

    def append(self, x: float) -> None:
        if len(self._buf) == self._capacity:
            self._sum -= self._buf.popleft()
        self._buf.append(x)
        self._sum += x

    def mean(self) -> Optional[float]:
        if len(self._buf) == 0:
            return None
        return self._sum / len(self._buf)


class IterativeMean:
    """Process each value just once, and the variables never get larger than the largest value in the stream,
    so you won't get an overflow.
    Adapted from @url https://stackoverflow.com/a/1934266/519951
    """

    def __init__(self):
        self._res: float = 0
        self._n: int = 1

    def add(self, x: float) -> None:
        self._res += (x - self._res) / self._n
        self._n += 1

    def get(self) -> float:
        return self._res
