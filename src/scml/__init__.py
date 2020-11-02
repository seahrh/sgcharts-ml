import os
import random
import re
import pandas as pd
import tensorflow as tf
from typing import List, Dict

import numpy as np

from .ml_stratifiers import *
from .model_checkpoint import *

__all__ = [
    "file_paths",
    "seed_everything",
    "var_name",
    "normalized_counts",
    "freq_encode",
]
__all__ += ml_stratifiers.__all__  # type: ignore  # module name is not defined
__all__ += model_checkpoint.__all__  # type: ignore  # module name is not defined


def file_paths(root_directory: str) -> List[str]:
    """Returns a list of file paths rooted at the given directory."""
    res = []
    for dirname, _, filenames in os.walk(root_directory):
        for filename in filenames:
            res.append(os.path.join(dirname, filename))
    return res


def seed_everything(seed: int = 31) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def var_name(s: str) -> str:
    res = s.strip().lower()
    res = re.sub(r"[\s]+", "_", res)
    res = re.sub(r"[\W]", "", res)
    return res


def normalized_counts(s: pd.Series) -> Dict[str, float]:
    return dict(s.value_counts(normalize=True))


def freq_encode(
    s: pd.Series,
    dtype=np.float64,
    encoding_map: Dict[str, float] = None,
    default: float = 0,
) -> pd.Series:
    if encoding_map is None:
        encoding_map = normalized_counts(s)
    return s.map(encoding_map).astype(dtype).fillna(default)
