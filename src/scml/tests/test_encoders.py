# noinspection PyUnresolvedReferences
import pandas as pd

# noinspection PyUnresolvedReferences
import numpy as np
from sklearn.metrics import pairwise_distances
from scml import *


class TestNormalizedCounts:
    def test_one_distinct_value(self):
        assert normalized_counts(pd.Series(["a", "a", "a", "a"])) == {"a": 1}

    def test_many_distinct_values(self):
        assert normalized_counts(pd.Series(["a", "b", "c", "d"])) == {
            "a": 0.25,
            "b": 0.25,
            "c": 0.25,
            "d": 0.25,
        }


class TestFreqEncode:
    def test_case_1(self):
        default = 0
        assert freq_encode(
            pd.Series(["a", "b", "c"]),
            encoding_map={"a": 0.75, "b": 0.25},
            default=default,
        ).to_list() == [0.75, 0.25, default]


class TestCyclicalEncode:
    _tol: float = 1e-7

    def test_zero_indexed_cycle_is_equidistant(self):
        _min, _max = 0, 9
        features = pd.DataFrame(columns=["cos", "sin"])
        features["cos"], features["sin"] = cyclical_encode(
            pd.Series(range(_min, _max + 1)), interval=(_min, _max)
        )
        D = pairwise_distances(features, metric="euclidean")
        d = D[0][1]
        assert D[1][2] - d <= abs(self._tol)
        assert D[2][3] - d <= abs(self._tol)
        assert D[3][4] - d <= abs(self._tol)
        assert D[4][5] - d <= abs(self._tol)
        assert D[5][6] - d <= abs(self._tol)
        assert D[6][7] - d <= abs(self._tol)
        assert D[7][8] - d <= abs(self._tol)
        assert D[8][9] - d <= abs(self._tol)
        assert D[9][0] - d <= abs(self._tol)

    def test_one_indexed_cycle_is_equidistant(self):
        _min, _max = 1, 10
        self._tol = 1e-7
        features = pd.DataFrame(columns=["cos", "sin"])
        features["cos"], features["sin"] = cyclical_encode(
            pd.Series(range(_min, _max + 1)), interval=(_min, _max)
        )
        D = pairwise_distances(features, metric="euclidean")
        d = D[0][1]
        assert D[1][2] - d <= abs(self._tol)
        assert D[2][3] - d <= abs(self._tol)
        assert D[3][4] - d <= abs(self._tol)
        assert D[4][5] - d <= abs(self._tol)
        assert D[5][6] - d <= abs(self._tol)
        assert D[6][7] - d <= abs(self._tol)
        assert D[7][8] - d <= abs(self._tol)
        assert D[8][9] - d <= abs(self._tol)
        assert D[9][0] - d <= abs(self._tol)
