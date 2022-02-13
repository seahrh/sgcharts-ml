import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from scml import (
    FrequencyEncoder,
    TargetEncoder,
    cyclical_encode,
    group_features,
    group_statistics,
)


class TestFrequencyEncoder:
    def test_one_distinct_value(self):
        assert FrequencyEncoder().encoding_map(pd.Series(["a", "a", "a", "a"])) == {
            "a": 1
        }

    def test_many_distinct_values(self):
        assert FrequencyEncoder().encoding_map(pd.Series(["a", "b", "c", "d"])) == {
            "a": 0.25,
            "b": 0.25,
            "c": 0.25,
            "d": 0.25,
        }

    def test_encode(self):
        default = 0
        dtype = np.float32
        a = FrequencyEncoder(encoding_map={"a": 0.75, "b": 0.25}).encode(
            pd.Series(["a", "b", "c"]), default=default, dtype=dtype
        )
        assert list(a) == [0.75, 0.25, default]
        assert a.dtype == dtype


class TestTargetEncoder:
    def test_one_group_one_target(self):
        assert TargetEncoder().encoding_map(
            target=pd.Series([1, 1, 1, 1, 1]),
            categorical=pd.Series(["a", "a", "a", "a", "a"]),
            method="mean",
        ) == {"a": 1}
        assert TargetEncoder().encoding_map(
            target=pd.Series([1, 1, 1, 1, 1]),
            categorical=pd.Series(["a", "a", "a", "a", "a"]),
            method="median",
        ) == {"a": 1}

    def test_one_group_many_targets(self):
        assert TargetEncoder().encoding_map(
            target=pd.Series([0, 0, 0, 1, 1]),
            categorical=pd.Series(["a", "a", "a", "a", "a"]),
            method="mean",
        ) == {"a": 0.4}
        assert TargetEncoder().encoding_map(
            target=pd.Series([0, 0, 0, 1, 1]),
            categorical=pd.Series(["a", "a", "a", "a", "a"]),
            method="median",
        ) == {"a": 0}

    def test_many_groups_one_target(self):
        assert TargetEncoder().encoding_map(
            target=pd.Series([1, 1, 1, 1, 1]),
            categorical=pd.Series(["a", "a", "a", "b", "b"]),
            method="mean",
        ) == {"a": 1, "b": 1}
        assert TargetEncoder().encoding_map(
            target=pd.Series([1, 1, 1, 1, 1]),
            categorical=pd.Series(["a", "a", "a", "b", "b"]),
            method="median",
        ) == {"a": 1, "b": 1}

    def test_many_groups_many_targets(self):
        assert TargetEncoder().encoding_map(
            target=pd.Series([0, 0, 0, 1, 1, 0, 1]),
            categorical=pd.Series(["a", "a", "a", "a", "a", "b", "b"]),
            method="mean",
        ) == {"a": 0.4, "b": 0.5}
        assert TargetEncoder().encoding_map(
            target=pd.Series([0, 0, 0, 1, 1, 0, 1]),
            categorical=pd.Series(["a", "a", "a", "a", "a", "b", "b"]),
            method="median",
        ) == {"a": 0, "b": 0.5}

    def test_encode(self):
        default = 0
        dtype = np.float32
        a = TargetEncoder(encoding_map={"a": 0.75, "b": 0.25}).encode(
            pd.Series(["a", "b", "c"]), default=default, dtype=dtype
        )
        assert list(a) == [0.75, 0.25, default]
        assert a.dtype == dtype


class TestCyclicalEncode:
    _tol: float = 1e-7

    def test_zero_indexed_cycle_is_equidistant(self):
        _min, _max = 0, 9
        features = pd.DataFrame(columns=["cos", "sin"])
        features["cos"], features["sin"] = cyclical_encode(
            np.arange(_min, _max + 1), interval=(_min, _max)
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
        features = pd.DataFrame(columns=["cos", "sin"])
        features["cos"], features["sin"] = cyclical_encode(
            np.arange(_min, _max + 1), interval=(_min, _max)
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
        a = group_features(
            data,
            statistics=group_statistics(data, column="a", group_columns=["b"]),
            column="a",
            group_columns=["b"],
            dtype=dtype,
        )
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
        assert np.allclose(a["a_p50_ratio"], a["a"] / a["a_p50"])
        assert np.allclose(a["a_mean_ratio"], a["a"] / a["a_mean"])
        assert np.allclose(a["a_min_ratio"], a["a"] / a["a_min"])
        assert np.allclose(a["a_max_ratio"], a["a"] / a["a_max"])
        assert np.allclose(a["a_std_ratio"], a["a"] / a["a_std"])
        assert np.allclose(a["a_p25_ratio"], a["a"] / a["a_p25"])
        assert np.allclose(a["a_p75_ratio"], a["a"] / a["a_p75"])
        assert np.allclose(a["a_p50_diff"], a["a"] - a["a_p50"])
        assert np.allclose(a["a_mean_diff"], a["a"] - a["a_mean"])
        assert np.allclose(a["a_min_diff"], a["a"] - a["a_min"])
        assert np.allclose(a["a_max_diff"], a["a"] - a["a_max"])
        assert np.allclose(a["a_std_diff"], a["a"] - a["a_std"])
        assert np.allclose(a["a_p25_diff"], a["a"] - a["a_p25"])
        assert np.allclose(a["a_p75_diff"], a["a"] - a["a_p75"])
        assert str(a["a_p50"].dtype) == dtype
        assert str(a["a_mean"].dtype) == dtype
        assert str(a["a_min"].dtype) == dtype
        assert str(a["a_max"].dtype) == dtype
        assert str(a["a_std"].dtype) == dtype
        assert str(a["a_p25"].dtype) == dtype
        assert str(a["a_p75"].dtype) == dtype
        assert str(a["a_p50_ratio"].dtype) == dtype
        assert str(a["a_mean_ratio"].dtype) == dtype
        assert str(a["a_min_ratio"].dtype) == dtype
        assert str(a["a_max_ratio"].dtype) == dtype
        assert str(a["a_std_ratio"].dtype) == dtype
        assert str(a["a_p25_ratio"].dtype) == dtype
        assert str(a["a_p75_ratio"].dtype) == dtype
        assert str(a["a_p50_diff"].dtype) == dtype
        assert str(a["a_mean_diff"].dtype) == dtype
        assert str(a["a_min_diff"].dtype) == dtype
        assert str(a["a_max_diff"].dtype) == dtype
        assert str(a["a_std_diff"].dtype) == dtype
        assert str(a["a_p25_diff"].dtype) == dtype
        assert str(a["a_p75_diff"].dtype) == dtype

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
        a = group_features(
            data,
            statistics=group_statistics(data, column="a", group_columns=["b", "c"]),
            column="a",
            group_columns=["b", "c"],
            dtype=dtype,
        )
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
        assert np.allclose(a["a_p50_ratio"], a["a"] / a["a_p50"])
        assert np.allclose(a["a_mean_ratio"], a["a"] / a["a_mean"])
        assert np.allclose(a["a_min_ratio"], a["a"] / a["a_min"])
        assert np.allclose(a["a_max_ratio"], a["a"] / a["a_max"])
        assert np.allclose(a["a_std_ratio"], a["a"] / a["a_std"])
        assert np.allclose(a["a_p25_ratio"], a["a"] / a["a_p25"])
        assert np.allclose(a["a_p75_ratio"], a["a"] / a["a_p75"])
        assert np.allclose(a["a_p50_diff"], a["a"] - a["a_p50"])
        assert np.allclose(a["a_mean_diff"], a["a"] - a["a_mean"])
        assert np.allclose(a["a_min_diff"], a["a"] - a["a_min"])
        assert np.allclose(a["a_max_diff"], a["a"] - a["a_max"])
        assert np.allclose(a["a_std_diff"], a["a"] - a["a_std"])
        assert np.allclose(a["a_p25_diff"], a["a"] - a["a_p25"])
        assert np.allclose(a["a_p75_diff"], a["a"] - a["a_p75"])
        assert str(a["a_p50"].dtype) == dtype
        assert str(a["a_mean"].dtype) == dtype
        assert str(a["a_min"].dtype) == dtype
        assert str(a["a_max"].dtype) == dtype
        assert str(a["a_std"].dtype) == dtype
        assert str(a["a_p25"].dtype) == dtype
        assert str(a["a_p75"].dtype) == dtype
        assert str(a["a_p50_ratio"].dtype) == dtype
        assert str(a["a_mean_ratio"].dtype) == dtype
        assert str(a["a_min_ratio"].dtype) == dtype
        assert str(a["a_max_ratio"].dtype) == dtype
        assert str(a["a_std_ratio"].dtype) == dtype
        assert str(a["a_p25_ratio"].dtype) == dtype
        assert str(a["a_p75_ratio"].dtype) == dtype
        assert str(a["a_p50_diff"].dtype) == dtype
        assert str(a["a_mean_diff"].dtype) == dtype
        assert str(a["a_min_diff"].dtype) == dtype
        assert str(a["a_max_diff"].dtype) == dtype
        assert str(a["a_std_diff"].dtype) == dtype
        assert str(a["a_p25_diff"].dtype) == dtype
        assert str(a["a_p75_diff"].dtype) == dtype
