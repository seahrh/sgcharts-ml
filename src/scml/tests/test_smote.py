import random

# noinspection PyUnresolvedReferences
import numpy as np

# noinspection PyUnresolvedReferences
import pandas as pd
import pytest
from scml import *
from typing import List, Dict, Union

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
            [{"a": -100, "b": -9e-10}, {"a": 100, "b": 9e-10}]
        )
        a = smote(df, size=1000)
        for row in a.itertuples():
            assert -100 <= row.a <= 100
            assert -9e-10 <= row.b <= 9e-10

    def test_when_column_is_not_numeric_then_raise_error(self):
        df = pd.DataFrame.from_records(
            [{"a": 1, "b": "string"}, {"a": 2, "b": "string"}]
        )
        with pytest.raises(ValueError, match=r"^column must be integer or float"):
            smote(df, size=2)

    def test_2_clusters(self):
        k_neighbours = 2
        clusters = [
            {"a_min": -100, "a_max": -90, "b_min": -0.1, "b_max": -0.001},
            {"a_min": 90, "a_max": 100, "b_min": 0.90, "b_max": 0.9999},
        ]
        rows = []
        for c in clusters:
            for _ in range(k_neighbours + 1):
                rows.append(
                    {
                        "a": random.randint(c["a_min"], c["a_max"]),
                        "b": random.uniform(c["b_min"], c["b_max"]),
                    }
                )
        inp = pd.DataFrame.from_records(rows)
        out = smote(inp, size=1000, k_neighbours=k_neighbours)
        counts = [0 for _ in range(len(clusters))]
        for row in out.itertuples():
            d = row._asdict()
            del d["Index"]
            i = _cluster(d, clusters=clusters)
            assert i >= 0
            counts[i] += 1
        for c in counts:
            assert c > 0
