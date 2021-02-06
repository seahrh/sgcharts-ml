__all__ = ["RollingWindow"]

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
