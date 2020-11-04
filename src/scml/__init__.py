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
    "quantize",
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


def quantize(df: pd.DataFrame, verbose: bool = True) -> None:
    """Reduce memory usage of pandas dataframe by quantization (i.e. use smaller data types).

    Removed float16 because precision is too low and risk of overflow is high.
    If a column is already in float16, it will remain in float16.

    :param df: pandas dataframe
    :param verbose: True to enable print output
    :return: None
    """
    numerics = ["int16", "int32", "int64", "float32", "float64"]
    start_mem: float = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    if verbose:
        end_mem: float = df.memory_usage().sum() / 1024 ** 2
        percent: float = 100 * (start_mem - end_mem) / start_mem
        print(f"Mem. usage decreased to {end_mem:5.2f} Mb ({percent:.1f}% reduction)")
