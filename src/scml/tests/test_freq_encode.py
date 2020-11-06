# noinspection PyUnresolvedReferences
import pandas as pd

# noinspection PyUnresolvedReferences
import numpy as np
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
        dtype = np.float32
        assert freq_encode(
            pd.Series(["a", "b", "c"]),
            dtype=dtype,
            encoding_map={"a": 0.75, "b": 0.25},
            default=default,
        ).to_list() == [0.75, 0.25, default]
