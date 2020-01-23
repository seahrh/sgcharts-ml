__all__ = [
    'file_paths',
    'seed_everything'
]

import os
import random
from typing import Iterable
import numpy as np


def file_paths(root_directory: str) -> Iterable[str]:
    """Returns a list of file paths rooted at the given directory."""
    res = []
    for dirname, _, filenames in os.walk(root_directory):
        for filename in filenames:
            res.append(os.path.join(dirname, filename))
    return res


def seed_everything(seed: int = 31):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
