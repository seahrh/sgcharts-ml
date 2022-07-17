import logging
import os
import random
import re
import sys
from typing import Sequence

import numpy as np
from numba import njit

__all__ = [
    "seed_everything",
    "var_name",
    "fillna",
    "uncertainty_weighted_loss",
]


def get_logger(name: str = None):
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


def uncertainty_weighted_loss(
    losses: Sequence[float], log_variances: Sequence[float]
) -> float:
    """Based on Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (Kendall 2018).
    Log variance represents the uncertainty. The higher the uncertainty, the smaller the weight.
    To prevent the model from simply suppressing all weights to zero, add the uncertainty to final loss.

    https://github.com/yaringal/multi-task-learning-example
    """
    if len(losses) == 0:
        raise ValueError("losses must not be empty")
    if len(losses) != len(log_variances):
        raise ValueError("Length of losses must equal log_variances")
    sm = 0
    for i in range(len(losses)):
        # weight ("precision") is a positive number between 0 and 1
        w = np.exp(-log_variances[i])
        sm += w * losses[i] + log_variances[i]
    return sm / len(losses)


from .ml_stratifiers import *

__all__ += ml_stratifiers.__all__  # type: ignore  # module name is not defined

from ._smote import *

__all__ += _smote.__all__  # type: ignore  # module name is not defined

from .streaming import *

__all__ += streaming.__all__  # type: ignore  # module name is not defined
from .timex import *

__all__ += timex.__all__  # type: ignore  # module name is not defined
