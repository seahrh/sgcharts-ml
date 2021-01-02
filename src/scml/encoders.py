__all__ = [
    "normalized_counts",
    "freq_encode",
    "cyclical_encode",
    "group_statistics",
    "group_features",
]
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, Iterable

Numeric = Union[int, float]


def normalized_counts(s: pd.Series) -> Dict[str, float]:
    return dict(s.value_counts(normalize=True))


def freq_encode(
    s: pd.Series,
    dtype=np.float32,
    encoding_map: Dict[str, float] = None,
    default: float = 0,
) -> pd.Series:
    if encoding_map is None:
        encoding_map = normalized_counts(s)
    return s.map(encoding_map).astype(dtype).fillna(default)


def cyclical_encode(
    s: pd.Series, interval: Tuple[Numeric, Numeric], dtype=np.float32,
) -> Tuple[pd.Series, pd.Series]:
    rg = interval[1] - interval[0]
    t = (s.to_numpy() - interval[0]) / rg
    cos = pd.Series(np.cos(2 * np.pi * t)).astype(dtype)
    sin = pd.Series(np.sin(2 * np.pi * t)).astype(dtype)
    return cos, sin


def group_statistics(
    data: pd.DataFrame, column: str, group_columns: Iterable[str], dtype=np.float32,
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