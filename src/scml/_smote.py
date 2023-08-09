__all__ = ["smote"]
import logging
import random
from typing import Dict, Iterable, List, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

log = logging.getLogger(__name__)


class Neighbour(NamedTuple):
    distance: float
    position: int


def smote(
    df: pd.DataFrame,
    size: int,
    k_neighbours: int = 5,
    distance_measure: str = "euclidean",
    columns: Optional[Iterable[str]] = None,
    embedding_columns: Optional[Iterable[str]] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    if size < 1:
        raise ValueError("must generate at least 1 synthetic example")
    if k_neighbours < 1:
        raise ValueError("k_neighbours must be at least 1")
    if len(df) < 2:
        raise ValueError("Input dataframe must have at least 2 records")
    _columns: List = list(df.columns)
    if columns is not None:
        _columns = list(columns)
    for col in _columns:
        dtype = str(df[col].dtype)
        if not dtype.startswith("int") and not dtype.startswith("float"):
            raise ValueError(
                f"column must be integer or float. Found col={col}, dtype={dtype}"
            )
    _embedding_columns: List = _columns
    if embedding_columns is not None:
        _embedding_columns = list(embedding_columns)
    if random_state is not None:
        random.seed(random_state)
    log.debug(f"_columns={_columns}, _embedding_columns={_embedding_columns}")
    distances = pairwise_distances(df[_embedding_columns], metric=distance_measure)
    log.debug(f"distances={distances}")
    neighbours: List[List[Neighbour]] = []
    for ds in distances:
        ns: List[Neighbour] = []
        for i, d in enumerate(ds):
            if d != 0:  # discard duplicates
                ns.append(Neighbour(distance=d, position=i))
        ns.sort()
        tmp = ns
        if k_neighbours < len(ns):
            tmp = ns[:k_neighbours]
        neighbours.append(tmp)
    rows: List[Dict[str, Union[int, float]]] = []
    while len(rows) < size:
        row = {}
        i = random.randint(0, len(neighbours) - 1)
        j = random.choice(neighbours[i]).position
        curr = df.iloc[i]
        neighbour = df.iloc[j]
        for col in _columns:
            gap = neighbour[col] - curr[col]
            gap_factor = random.random()
            v = gap_factor * gap + curr[col]
            if isinstance(curr[col], int):
                v = round(v)
            row[col] = v
        rows.append(row)
    res = pd.DataFrame.from_records(rows)
    types: Dict[str, np.dtype] = {}
    for col in _columns:
        types[col] = df[col].dtype
    res = res.astype(types)
    return res
