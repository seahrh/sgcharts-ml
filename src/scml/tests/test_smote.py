# noinspection PyUnresolvedReferences
import random

# noinspection PyUnresolvedReferences
from typing import Dict, List, Union

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
                    "c": random.uniform(c["c_min"], c["c_max"]),
                }
            )
    return pd.DataFrame.from_records(rows)


class TestSmote:
    DISTANCE_MEASURES: List[str] = ["euclidean", "cosine"]

    def test_target_number_of_synthetic_examples(self):
        size = 2
        df = pd.DataFrame.from_records([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        for distance_measure in self.DISTANCE_MEASURES:
            assert len(smote(df, size=size, distance_measure=distance_measure)) == size

    def test_data_types_are_preserved(self):
        for distance_measure in self.DISTANCE_MEASURES:
            inp = pd.DataFrame.from_records(
                [{"a": -100, "b": -9e-10}, {"a": 100, "b": 9e-10}]
            )
            e = {"a": "int32", "b": "float32"}
            inp = inp.astype(e)
            out = smote(inp, size=2, distance_measure=distance_measure)
            a = dict(out.dtypes)
            for k, v in a.items():
                a[k] = str(v)
            assert a == e

    def test_synthetic_values_bound_by_neighbours(self):
        for distance_measure in self.DISTANCE_MEASURES:
            df = pd.DataFrame.from_records(
                [{"a": -100, "b": -9e-10, "c": 0}, {"a": 100, "b": 9e-10, "c": 0}]
            )
            a = smote(df, size=1000, distance_measure=distance_measure)
            for row in a.itertuples():
                assert -100 <= row.a <= 100
                assert -9e-10 <= row.b <= 9e-10
                assert row.c == 0

    def test_when_column_is_not_numeric_then_raise_error(self):
        for distance_measure in self.DISTANCE_MEASURES:
            df = pd.DataFrame.from_records(
                [{"a": 1, "b": "string"}, {"a": 2, "b": "string"}]
            )
            with pytest.raises(ValueError, match=r"^column must be integer or float"):
                smote(df, size=2, distance_measure=distance_measure)

    def test_synthetic_point_must_belong_to_a_cluster(self):
        k_neighbours = 2
        clusters = [
            {
                "a_min": -100,
                "a_max": -90,
                "b_min": -0.99,
                "b_max": -0.01,
                "c_min": -0.99,
                "c_max": -0.01,
            },
            {
                "a_min": 90,
                "a_max": 100,
                "b_min": 10.01,
                "b_max": 10.99,
                "c_min": 10.01,
                "c_max": 10.99,
            },
        ]
        for distance_measure in self.DISTANCE_MEASURES:
            inp = _input_dataframe(clusters, size=k_neighbours + 1)
            out = smote(
                inp,
                size=1000,
                k_neighbours=k_neighbours,
                distance_measure=distance_measure,
            )
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
        # at least 2 vector components are required for cosine distance to work
        embedding_columns = {"b", "c"}
        clusters = [
            {
                "a_min": 0,
                "a_max": 0,
                "b_min": -0.99,
                "b_max": -0.01,
                "c_min": -0.99,
                "c_max": -0.01,
            },
            {
                "a_min": 1,
                "a_max": 1,
                "b_min": 10.01,
                "b_max": 10.99,
                "c_min": 10.01,
                "c_max": 10.99,
            },
        ]
        for distance_measure in self.DISTANCE_MEASURES:
            inp = _input_dataframe(clusters, size=k_neighbours + 1)
            out = smote(
                inp,
                size=1000,
                k_neighbours=k_neighbours,
                columns=columns,
                embedding_columns=embedding_columns,
                distance_measure=distance_measure,
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
            print(f"distance_measure={distance_measure}, counts={counts}")
            for c in counts:
                assert c > 0
