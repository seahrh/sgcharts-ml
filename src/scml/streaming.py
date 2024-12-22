__all__ = ["RollingWindow", "RunningMean"]

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


class RunningMean:
    """Process each value just once, and the variables never get larger than the largest value in the stream,
    so you won't get an overflow.
    Adapted from @url https://stackoverflow.com/a/1934266/519951
    """

    def __init__(self, initial_value: float = 0, size_on_next_addition: int = 1):
        self.mean: float = initial_value
        self.size: int = size_on_next_addition

    def add(self, x: float) -> None:
        self.mean += (x - self.mean) / self.size
        self.size += 1

    def get(self) -> float:
        return self.mean
