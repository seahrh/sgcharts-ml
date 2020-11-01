import os
import random
import re
from typing import List

import numpy as np

from .ml_stratifiers import *

__all__ = [
    "file_paths",
    "seed_everything",
    "var_name",
]
__all__ += ml_stratifiers.__all__  # type: ignore  # Name 'ml_stratifiers' is not defined


def file_paths(root_directory: str) -> List[str]:
    """Returns a list of file paths rooted at the given directory."""
    res = []
    for dirname, _, filenames in os.walk(root_directory):
        for filename in filenames:
            res.append(os.path.join(dirname, filename))
    return res


def seed_everything(seed: int = 31):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def var_name(s: str) -> str:
    res = s.strip().lower()
    res = re.sub(r"[\s]+", "_", res)
    res = re.sub(r"[\W]", "", res)
    return res
