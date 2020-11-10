# noinspection PyUnresolvedReferences
import numpy as np

# noinspection PyUnresolvedReferences
import pandas as pd
import pytest
from scml import *


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
