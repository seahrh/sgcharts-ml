import os
import random
import re
import warnings
from typing import List

import numpy as np
import pandas as pd
from numba import njit

from ._smote import *
from .encoders import *
from .ml_stratifiers import *
from .streaming import *

__all__ = [
    "file_paths",
    "seed_everything",
    "var_name",
    "quantize",
    "find_missing_values",
    "rescale_as_int",
    "fillna",
]
__all__ += ml_stratifiers.__all__  # type: ignore  # module name is not defined
__all__ += _smote.__all__  # type: ignore  # module name is not defined
__all__ += encoders.__all__  # type: ignore  # module name is not defined
__all__ += streaming.__all__  # type: ignore  # module name is not defined

try:
    import tensorflow as tf
    from .model_checkpoint import *

    __all__ += model_checkpoint.__all__  # type: ignore  # module name is not defined
except ImportError:
    tf = None
    warnings.warn("Install tensorflow to use this feature", ImportWarning)


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
    if tf is not None:
        tf.random.set_seed(seed)


def var_name(s: str) -> str:
    res = s.strip().lower()
    res = re.sub(r"[\s]+", "_", res)
    res = re.sub(r"[\W]", "", res)
    return res


def rescale_as_int(
    s: pd.Series, min_value: float = None, max_value: float = None, dtype=np.int16
) -> pd.Series:
    """Cannot be converted to njit because np.clip is unsupported."""
    valid_dtypes = {np.int8, np.int16, np.int32}
    if dtype not in valid_dtypes:
        raise ValueError(f"dtype: expecting [{valid_dtypes}] but found [{dtype}]")
    if min_value is None:
        min_value = min(s)
    if max_value is None:
        max_value = max(s)
    if min_value == 0 and max_value == 0:
        raise ValueError("Both min_value and max_value must not be zero")
    limit = max(abs(min_value), abs(max_value))
    res = np.clip(s / limit, 0, 1) * np.iinfo(dtype).max
    return res.astype(dtype)


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
            _min = df[col].min()
            _max = df[col].max()
            dtype = np.float64  # type: ignore
            if str(col_type)[:3] == "int":
                if _min >= np.iinfo(np.int8).min and _max <= np.iinfo(np.int8).max:
                    dtype = np.int8  # type: ignore
                elif _min >= np.iinfo(np.int16).min and _max <= np.iinfo(np.int16).max:
                    dtype = np.int16  # type: ignore
                elif _min >= np.iinfo(np.int32).min and _max <= np.iinfo(np.int32).max:
                    dtype = np.int32  # type: ignore
                else:
                    dtype = np.int64  # type: ignore
            elif _min >= np.finfo(np.float32).min and _max <= np.finfo(np.float32).max:
                dtype = np.float32  # type: ignore
            df[col] = df[col].astype(dtype)
    if verbose:
        end_mem: float = df.memory_usage().sum() / 1024 ** 2
        percent: float = 100 * (start_mem - end_mem) / start_mem
        print(f"Mem. usage decreased to {end_mem:5.2f} Mb ({percent:.1f}% reduction)")


def find_missing_values(
    df: pd.DataFrame, blank_strings_as_null: bool = True
) -> pd.DataFrame:
    if blank_strings_as_null:
        df = df.replace(r"^\s*$", np.nan, regex=True)
    total = df.isna().sum()
    percent = df.isna().sum() / df.isna().count()
    res = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    types = [str(df[col].dtype) for col in df.columns]
    res["Type"] = types
    return res


@njit
def fillna(
    arr: np.ndarray,
    values: np.ndarray,
    add_flag: bool = False,
    dtype: np.dtype = np.float32,  # type: ignore
) -> np.ndarray:
    mask = np.isnan(arr)
    res = np.where(mask, values, arr).astype(dtype)
    if not add_flag:
        return res  # type: ignore
    flags = np.where(mask, np.full(arr.shape, 1), np.full(arr.shape, 0)).astype(dtype)
    return np.hstack((res, flags))
