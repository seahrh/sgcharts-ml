__all__ = [
    "FrequencyEncoder",
    "TargetEncoder",
    "cyclical_encode",
    "group_statistics",
    "group_features",
]
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

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
    group_columns: Sequence[str],
    aggregates: Optional[Union[Set[str], FrozenSet[str]]] = frozenset(
        ["mean", "std", "min", "max"]
    ),
    percentiles: Optional[Sequence[int]] = None,
    dtype=np.float32,
) -> pd.DataFrame:
    if len(group_columns) == 0:
        raise ValueError("groupby must be at least 1 column")
    if (aggregates is None or len(aggregates) == 0) and (
        percentiles is None or len(percentiles) == 0
    ):
        raise ValueError(
            "No statistics to compute. Both `aggregates` and `percentiles` are None or empty."
        )
    c_map: Dict[str, str] = {}
    res = None
    grouped = data.groupby(group_columns, sort=True)
    index = [k for k in grouped.groups.keys()]
    if len(group_columns) > 1:
        # noinspection PyTypeChecker
        index = pd.MultiIndex.from_tuples(index, names=group_columns)
    if aggregates is not None:
        for a in aggregates:
            c_map[a] = f"{column}_{a}"
        a_list = list(aggregates - {"std"})
        if len(a_list) != 0:
            res = grouped[column].agg(a_list)
        if "std" in aggregates:
            # population standard deviation to prevent NaN
            sr = grouped[column].std(ddof=0)
            sr.name = "std"
            if res is None:
                res = sr.to_frame()
            else:
                res = pd.concat([res, sr], axis=1)
    if percentiles is not None and len(percentiles) != 0:
        for p in percentiles:
            c_map[f"p{p}"] = f"{column}_p{p}"
        quantiles: List[float] = [p / 100 for p in percentiles]
        df = grouped[column].quantile(quantiles).to_frame()
        df = df.reset_index()
        cols = list(df.columns)
        i = len(group_columns)
        quantiles_ls: List[List[float]] = [[] for _ in range(len(percentiles))]
        for t in df.itertuples():
            p = int(getattr(t, cols[i]) * 100)
            quantiles_ls[percentiles.index(p)].append(getattr(t, cols[i + 1]))
        quantiles_sr = []
        for i, ls in enumerate(quantiles_ls):
            # align series index with res dataframe
            quantiles_sr.append(
                pd.Series(
                    ls,
                    index=index if res is None else res.index,
                    name=f"p{percentiles[i]}",
                )
            )
        if res is None:
            res = pd.concat([sr.to_frame() for sr in quantiles_sr], axis=1)
        else:
            res = pd.concat([res] + quantiles_sr, axis=1)
    if res is not None:
        res = res.rename(columns=c_map, errors="raise")
        res = res.astype(dtype)
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
