__all__ = ["smote"]
import pandas as pd
from sklearn.metrics import pairwise_distances
from typing import Iterable, List, Dict


def smote(
    df: pd.DataFrame,
    size: int,
    columns: Iterable[str],
    k_neighbours: int,
    distance_measure: str = "euclidean",
    embedding_columns: Iterable[str] = None,
) -> pd.DataFrame:
    if size < 1:
        raise ValueError("must generate at least 1 synthetic example")
    if k_neighbours < 1:
        raise ValueError("k_neighbours must be at least 1")
    if len(df) == 0:
        raise ValueError("Input dataframe must not be empty")
    _columns = set(columns)
    if len(_columns) == 0:
        raise ValueError("Number of columns to synthesize must be at least 1")
    _embedding_columns = None
    if embedding_columns is not None:
        _embedding_columns = set(embedding_columns)
    if _embedding_columns is None or len(_embedding_columns) == 0:
        _embedding_columns = _columns
    distances = pairwise_distances(df[_embedding_columns], metric=distance_measure)
    rows: List[Dict] = []
    return pd.DataFrame.from_records(rows)
