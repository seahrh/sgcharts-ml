__all__ = ["RollingWindow", "RunningMean"]

from collections import deque
from typing import Deque


class RollingWindow:
    def __init__(self, capacity: int, initial_value: float = 0):
        self.capacity: int = capacity
        self.sum: float = initial_value
        self.buf: Deque[float] = deque()

    def append(self, x: float) -> None:
        if len(self.buf) == self.capacity:
            self.sum -= self.buf.popleft()
        self.buf.append(x)
        self.sum += x

    def mean(self) -> float:
        if len(self.buf) == 0:
            return self.sum
        return self.sum / len(self.buf)

    def __len__(self):
        return len(self.buf)


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
