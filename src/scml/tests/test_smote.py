# noinspection PyUnresolvedReferences
from typing import List, Dict, Union

# noinspection PyUnresolvedReferences
import numpy as np

# noinspection PyUnresolvedReferences
import pandas as pd
import pytest

from scml import *

Numeric = Union[int, float]


def _cluster(vector: Dict[str, Numeric], clusters: List[Dict[str, Numeric]]) -> int:
    for i, c in enumerate(clusters):
        is_member = True
        for k, v in vector.items():
            if v < c[f"{k}_min"] or v > c[f"{k}_max"]:
                is_member = False
                break
        if is_member:
            return i
    return -1


def _input_dataframe(clusters: List[Dict[str, Numeric]], size: int) -> pd.DataFrame:
    rows = []
    for c in clusters:
        for _ in range(size):
            rows.append(
                {
                    "a": random.randint(int(c["a_min"]), int(c["a_max"])),
                    "b": random.uniform(c["b_min"], c["b_max"]),
                }
            )
    return pd.DataFrame.from_records(rows)


class TestSmote:
    def test_target_number_of_synthetic_examples(self):
        df = pd.DataFrame.from_records([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        assert len(smote(df, size=2)) == 2

    def test_data_types_are_preserved(self):
        inp = pd.DataFrame.from_records(
            [{"a": -100, "b": -9e-10}, {"a": 100, "b": 9e-10}]
        )
        e = {"a": "int32", "b": "float32"}
        inp = inp.astype(e)
        out = smote(inp, size=2)
        a = dict(out.dtypes)
        for k, v in a.items():
            a[k] = str(v)
        assert a == e

    def test_synthetic_values_bound_by_neighbours(self):
        df = pd.DataFrame.from_records(
            [{"a": -100, "b": -9e-10, "c": 0}, {"a": 100, "b": 9e-10, "c": 0}]
        )
        a = smote(df, size=1000)
        for row in a.itertuples():
            assert -100 <= row.a <= 100
            assert -9e-10 <= row.b <= 9e-10
            assert row.c == 0

    def test_when_column_is_not_numeric_then_raise_error(self):
        df = pd.DataFrame.from_records(
            [{"a": 1, "b": "string"}, {"a": 2, "b": "string"}]
        )
        with pytest.raises(ValueError, match=r"^column must be integer or float"):
            smote(df, size=2)

    def test_synthetic_point_must_belong_to_a_cluster(self):
        k_neighbours = 2
        clusters = [
            {"a_min": -100, "a_max": -90, "b_min": -0.1, "b_max": -0.001},
            {"a_min": 90, "a_max": 100, "b_min": 0.90, "b_max": 0.9999},
        ]
        inp = _input_dataframe(clusters, size=k_neighbours + 1)
        out = smote(inp, size=1000, k_neighbours=k_neighbours)
        counts = [0 for _ in range(len(clusters))]
        for t in out.itertuples():
            row = t._asdict()
            del row["Index"]
            i = _cluster(row, clusters=clusters)
            assert i >= 0
            counts[i] += 1
        for c in counts:
            assert c > 0

    def test_embedding_columns_are_used_for_nearest_neighbours_but_not_synthesized(
        self,
    ):
        k_neighbours = 2
        columns = {"a"}
        embedding_columns = {"b"}
        clusters = [
            {"a_min": 0, "a_max": 0, "b_min": -0.1, "b_max": -0.001},
            {"a_min": 1, "a_max": 1, "b_min": 0.90, "b_max": 0.9999},
        ]
        inp = _input_dataframe(clusters, size=k_neighbours + 1)
        out = smote(
            inp,
            size=1000,
            k_neighbours=k_neighbours,
            columns=columns,
            embedding_columns=embedding_columns,
        )
        counts = [0 for _ in range(len(clusters))]
        for t in out.itertuples():
            row = t._asdict()
            for col in embedding_columns:
                assert col not in row
            for col in columns:
                i = int(row[col])
                assert 0 <= i <= len(clusters) - 1
                counts[i] += 1
        for c in counts:
            assert c > 0
