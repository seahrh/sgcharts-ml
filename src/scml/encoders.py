__all__ = ["normalized_counts", "freq_encode", "cyclical_encode"]
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union

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
