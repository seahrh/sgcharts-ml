import logging
import os
import random
import re
import sys
from bisect import bisect
from typing import FrozenSet, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
from numba import njit

__all__ = [
    "getboolean",
    "seed_everything",
    "weighted_choice",
    "contiguous_ranges",
    "var_name",
    "fillna",
]

TRUTHY: FrozenSet[str] = frozenset({"1", "yes", "true", "on"})


def getboolean(s: str, truthy: Union[Set[str], FrozenSet[str]] = TRUTHY) -> bool:
    """
    Inspired by configparser's implementation but there is one key difference:
    This method does not throw exceptions, it simply returns False by default.

    A convenience method which coerces the option in the specified section to a Boolean value.
    Note that the accepted values for the option are '1', 'yes', 'true', and 'on',
    which cause this method to return True.
    These string values are checked in a case-insensitive manner. Any other value will return False.
    :return:
    """
    if s.lower() in truthy:
        return True
    return False


def get_logger(name: Optional[str] = None):
    # suppress matplotlib logging
    logging.getLogger(name="matplotlib").setLevel(logging.WARNING)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s"
    )
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def seed_everything(seed: int = 31) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def var_name(s: str) -> str:
    res = s.strip().lower()
    res = re.sub(r"[\s]+", "_", res)
    res = re.sub(r"[\W]", "", res)
    return res


def weighted_choice(weights: Iterable[float]) -> int:
    """
    1. Arrange the weights into a cumulative distribution.
    2. Use random.random() to pick a random float 0.0 <= x < total.
    3. Binary search on the cumulative distribution

    Time complexity O(N) to build the cumulative distribution.
    Based on https://stackoverflow.com/a/4322940/519951

    :param weights:
    :return:
    """
    total: float = 0
    cum_weights: List[float] = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = random.random() * total
    # bisect_right
    # The return value i is such that all e in a[:i] have e <= x,
    # and all e in a[i:] have e > x.
    return bisect(cum_weights, x)


def contiguous_ranges(a: List[int]) -> List[Tuple[int, int]]:
    if len(a) == 0:
        raise ValueError("a must not be empty")
    a = sorted(a)
    los = [a[0]]
    his = []
    for i in range(len(a) - 1):
        if a[i + 1] - a[i] > 1:
            his.append(a[i])
            los.append(a[i + 1])
    if len(his) < len(los):
        his.append(a[-1])
    return list(zip(los, his))


@njit
def fillna(
    arr: np.ndarray, values: np.ndarray, add_flag: bool = False, dtype=np.float32
) -> np.ndarray:
    mask = np.isnan(arr)
    res = np.where(mask, values, arr).astype(dtype)
    if not add_flag:
        return res  # type: ignore
    flags = np.where(mask, np.full(arr.shape, 1), np.full(arr.shape, 0)).astype(dtype)
    return np.hstack((res, flags))


from .ml_stratifiers import *

__all__ += ml_stratifiers.__all__  # type: ignore  # module name is not defined

from ._smote import *

__all__ += _smote.__all__  # type: ignore  # module name is not defined

from .streaming import *

__all__ += streaming.__all__  # type: ignore  # module name is not defined
from .timex import *

__all__ += timex.__all__  # type: ignore  # module name is not defined
