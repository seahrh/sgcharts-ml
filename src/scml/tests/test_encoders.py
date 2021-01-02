# noinspection PyUnresolvedReferences
import pandas as pd
import pytest

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


class TestGroupStatistics:
    def test_one_group_column(self):
        dtype = "float32"
        data = pd.DataFrame(
            {
                "a": [1, 2, 7, 4, 13, 6],
                "b": ["foo", "bar", "foo", "bar", "foo", "bar"],
                "c": [1, 2, 1, 2, 1, 2],
            },
            dtype=dtype,
        )
        a = group_statistics(data, column="a", group_columns=["b"])
        assert set(a.index) == {"bar", "foo"}
        assert a.loc["foo"].to_dict() == {
            "a_p50": 7,
            "a_max": 13,
            "a_mean": 7,
            "a_min": 1,
            "a_p25": 4,
            "a_p75": 10,
            "a_std": 4.898979663848877,
        }
        assert a.loc["bar"].to_dict() == {
            "a_max": 6,
            "a_mean": 4,
            "a_min": 2,
            "a_p25": 3,
            "a_p50": 4,
            "a_p75": 5,
            "a_std": 1.632993221282959,
        }
        assert str(a["a_p50"].dtype) == dtype
        assert str(a["a_mean"].dtype) == dtype
        assert str(a["a_min"].dtype) == dtype
        assert str(a["a_max"].dtype) == dtype
        assert str(a["a_std"].dtype) == dtype
        assert str(a["a_p25"].dtype) == dtype
        assert str(a["a_p75"].dtype) == dtype

    def test_two_group_columns(self):
        dtype = "float32"
        data = pd.DataFrame(
            {
                "a": [1, 2, 7, 4, 13, 6],
                "b": ["foo", "bar", "foo", "bar", "foo", "bar"],
                "c": [1, 2, 1, 2, 1, 2],
            },
            dtype=dtype,
        )
        a = group_statistics(data, column="a", group_columns=["b", "c"])
        assert set(a.index) == {("foo", 1), ("bar", 2)}
        assert a.loc[("foo", 1)].to_dict() == {
            "a_p50": 7,
            "a_max": 13,
            "a_mean": 7,
            "a_min": 1,
            "a_p25": 4,
            "a_p75": 10,
            "a_std": 4.898979663848877,
        }
        assert a.loc[("bar", 2)].to_dict() == {
            "a_max": 6,
            "a_mean": 4,
            "a_min": 2,
            "a_p25": 3,
            "a_p50": 4,
            "a_p75": 5,
            "a_std": 1.632993221282959,
        }
        assert str(a["a_p50"].dtype) == dtype
        assert str(a["a_mean"].dtype) == dtype
        assert str(a["a_min"].dtype) == dtype
        assert str(a["a_max"].dtype) == dtype
        assert str(a["a_std"].dtype) == dtype
        assert str(a["a_p25"].dtype) == dtype
        assert str(a["a_p75"].dtype) == dtype


@pytest.mark.skip
class TestGroupFeatures:
    def test_one_group_column(self):
        dtype = "float32"
        data = pd.DataFrame(
            {
                "a": [1, 2, 7, 4, 13, 6],
                "b": ["foo", "bar", "foo", "bar", "foo", "bar"],
                "c": [1, 2, 1, 2, 1, 2],
            },
            dtype=dtype,
        )
        a = group_statistics(data, column="a", group_columns=["b"])
        assert list(data.index) == list(a.index)
        assert list(a["a_p50"]) == [7, 4, 7, 4, 7, 4]
        assert list(a["a_mean"]) == [7, 4, 7, 4, 7, 4]
        assert list(a["a_min"]) == [1, 2, 1, 2, 1, 2]
        assert list(a["a_max"]) == [13, 6, 13, 6, 13, 6]
        assert list(a["a_p25"]) == [4, 3, 4, 3, 4, 3]
        assert list(a["a_p75"]) == [10, 5, 10, 5, 10, 5]
        assert np.allclose(
            a["a_std"],
            [
                4.898979663848877,
                1.632993221282959,
                4.898979663848877,
                1.632993221282959,
                4.898979663848877,
                1.632993221282959,
            ],
        )
        assert str(a["a_p50"].dtype) == dtype
        assert str(a["a_mean"].dtype) == dtype
        assert str(a["a_min"].dtype) == dtype
        assert str(a["a_max"].dtype) == dtype
        assert str(a["a_std"].dtype) == dtype
        assert str(a["a_p25"].dtype) == dtype
        assert str(a["a_p75"].dtype) == dtype

    def test_two_group_columns(self):
        dtype = "float32"
        data = pd.DataFrame(
            {
                "a": [1, 2, 7, 4, 13, 6],
                "b": ["foo", "bar", "foo", "bar", "foo", "bar"],
                "c": [1, 2, 1, 2, 1, 2],
            },
            dtype=dtype,
        )
        a = group_statistics(data, column="a", group_columns=["b", "c"])
        assert list(data.index) == list(a.index)
        assert list(a["a_p50"]) == [7, 4, 7, 4, 7, 4]
        assert list(a["a_mean"]) == [7, 4, 7, 4, 7, 4]
        assert list(a["a_min"]) == [1, 2, 1, 2, 1, 2]
        assert list(a["a_max"]) == [13, 6, 13, 6, 13, 6]
        assert list(a["a_p25"]) == [4, 3, 4, 3, 4, 3]
        assert list(a["a_p75"]) == [10, 5, 10, 5, 10, 5]
        assert np.allclose(
            a["a_std"],
            [
                4.898979663848877,
                1.632993221282959,
                4.898979663848877,
                1.632993221282959,
                4.898979663848877,
                1.632993221282959,
            ],
        )
        assert str(a["a_p50"].dtype) == dtype
        assert str(a["a_mean"].dtype) == dtype
        assert str(a["a_min"].dtype) == dtype
        assert str(a["a_max"].dtype) == dtype
        assert str(a["a_std"].dtype) == dtype
        assert str(a["a_p25"].dtype) == dtype
        assert str(a["a_p75"].dtype) == dtype
