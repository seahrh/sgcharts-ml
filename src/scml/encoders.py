__all__ = [
    "FrequencyEncoder",
    "TargetEncoder",
    "cyclical_encode",
    "group_statistics",
    "group_features",
]
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from numba import njit


class FrequencyEncoder:
    def __init__(self, encoding_map: Dict[str, float] = None):
        self._map = encoding_map

    def encoding_map(self, s: pd.Series) -> Dict[str, float]:
        if self._map is None:
            self._map = dict(s.value_counts(normalize=True))
        return self._map

    def encode(self, s: pd.Series, dtype=np.float32, default: float = 0) -> pd.Series:
        if self._map is None:
            self.encoding_map(s)
        return s.map(self._map).astype(dtype).fillna(default)


class TargetEncoder:
    def __init__(self, encoding_map: Dict[str, float] = None):
        self._map = encoding_map

    def encoding_map(
        self, target: pd.Series, categorical: pd.Series, method: str
    ) -> Dict[str, float]:
        if len(target) == 0:
            raise ValueError("target must not be empty")
        if len(categorical) == 0:
            raise ValueError("categorical must not be empty")
        if len(target) != len(categorical):
            raise ValueError("target and categorical must have the same length")
        if self._map is None:
            df = pd.DataFrame({"target": target, "categorical": categorical})
            self._map = df.groupby(categorical)["target"].agg(method).to_dict()
        return self._map

    def encode(
        self,
        categorical: pd.Series,
        method: str = "mean",
        dtype=np.float32,
        default: float = 0,
        target: pd.Series = None,
    ) -> pd.Series:
        if self._map is None:
            self.encoding_map(target, categorical, method=method)
        return categorical.map(self._map).astype(dtype).fillna(default)


@njit
def cyclical_encode(
    a: np.ndarray,
    interval: Tuple[float, float],
    dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    rg = interval[1] - interval[0]
    frac = (a - interval[0]) / rg
    cos = np.cos(2 * np.pi * frac).astype(dtype)
    sin = np.sin(2 * np.pi * frac).astype(dtype)
    return cos, sin


def group_statistics(
    data: pd.DataFrame,
    column: str,
    group_columns: Iterable[str],
    dtype=np.float32,
) -> pd.DataFrame:
    columns = [
        f"{column}_p50",
        f"{column}_mean",
        f"{column}_min",
        f"{column}_max",
        f"{column}_std",
        f"{column}_p25",
        f"{column}_p75",
    ]
    grouped = data.groupby(group_columns, sort=False)
    res = grouped[column].agg(["median", "mean", "min", "max"])
    res.rename(
        columns={
            "median": columns[0],
            "mean": columns[1],
            "min": columns[2],
            "max": columns[3],
        },
        inplace=True,
    )
    # population standard deviation to prevent NaN
    agg = grouped[column].std(ddof=0)
    agg.rename(columns[4], inplace=True)
    res = res.merge(agg, left_index=True, right_index=True)
    agg = grouped[column].quantile(0.25)
    agg.rename(columns[5], inplace=True)
    res = res.merge(agg, left_index=True, right_index=True)
    agg = grouped[column].quantile(0.75)
    agg.rename(columns[6], inplace=True)
    res = res.merge(agg, left_index=True, right_index=True)
    for col in columns:
        res[col] = res[col].astype(dtype)
    return res


def group_features(
    df: pd.DataFrame,
    statistics: pd.DataFrame,
    column: str,
    group_columns: Iterable[str],
    dtype=np.float32,
) -> pd.DataFrame:
    res = df.merge(statistics, how="left", left_on=group_columns, right_index=True)
    eps = np.finfo(dtype).eps
    for statistic_column in statistics.columns:
        ratio_col = f"{statistic_column}_ratio"
        diff_col = f"{statistic_column}_diff"
        # Prevent division-by-zero error
        res[ratio_col] = res[column] / res[statistic_column].replace(0, eps)
        res[diff_col] = res[column] - res[statistic_column]
        res[statistic_column] = res[statistic_column].astype(dtype)
        res[ratio_col] = res[ratio_col].astype(dtype)
        res[diff_col] = res[diff_col].astype(dtype)
    return res
