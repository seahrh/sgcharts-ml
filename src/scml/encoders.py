__all__ = ["normalized_counts", "freq_encode", "cyclical_encode", "group_features"]
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


def deprecated_group_features(
    data: pd.DataFrame,
    column: str,
    group_columns: Iterable[str],
    functions: Tuple[str, ...] = ("median", "mean", "min", "max", "std"),
) -> pd.DataFrame:
    grouped = data.groupby(group_columns, sort=False)[column].agg(functions)
    rows = []
    for t in data.itertuples():
        # noinspection PyProtectedMember
        t_row = t._asdict()
        fields = [t_row[col] for col in group_columns]
        g = grouped.loc[tuple(fields)]
        row = {"Index": t_row["Index"]}
        for f in functions:
            row[f] = g[f]
        rows.append(row)
    res = pd.DataFrame.from_records(rows)
    if "std" in res.columns:  # prevent division-by-zero error
        eps = np.finfo(np.float32).eps
        res["std"].fillna(eps, inplace=True)
    res.set_index("Index", drop=True, inplace=True)
    return res


def group_features(
    data: pd.DataFrame,
    column: str,
    group_columns: Iterable[str],
    functions: Tuple[str, ...] = ("median", "mean", "min", "max", "std"),
    dtype=np.float32,
) -> pd.DataFrame:
    grouped = data.groupby(group_columns, sort=False)[column].agg(functions)
    columns = {f: f"{column}_{f}" for f in functions}
    if "median" in columns:
        columns["median"] = f"{column}_p50"
    grouped.rename(columns=columns, inplace=True)
    res = data.merge(grouped, how="left", left_on=group_columns, right_index=True)
    for col in columns.values():
        res[col] = res[col].astype(dtype)
    return res
