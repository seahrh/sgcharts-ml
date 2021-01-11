import numpy as np
import pandas as pd
from scml import rescale_as_int


class TestRescaleAsInt:
    def test_int8_rescale(self):
        dtype = np.int8
        a = rescale_as_int(
            pd.Series(
                [0, 0.101, 0.201, 0.301, 0.401, 0.501, 0.601, 0.701, 0.801, 0.901, 1]
            ),
            dtype=dtype,
        )
        assert list(a) == [0, 12, 25, 38, 50, 63, 76, 89, 101, 114, 127]
        assert a.dtype == dtype
        a = rescale_as_int(
            pd.Series([0, 10.1, 20.1, 30.1, 40.1, 50.1, 60.1, 70.1, 80.1, 90.1, 100]),
            dtype=dtype,
        )
        assert list(a) == [0, 12, 25, 38, 50, 63, 76, 89, 101, 114, 127]
        assert a.dtype == dtype

    def test_int16_rescale(self):
        dtype = np.int16
        a = rescale_as_int(
            pd.Series(
                [0, 0.101, 0.201, 0.301, 0.401, 0.501, 0.601, 0.701, 0.801, 0.901, 1]
            ),
            dtype=dtype,
        )
        assert list(a) == [
            0,
            3309,
            6586,
            9862,
            13139,
            16416,
            19692,
            22969,
            26246,
            29523,
            32767,
        ]
        assert a.dtype == dtype
        a = rescale_as_int(
            pd.Series([0, 10.1, 20.1, 30.1, 40.1, 50.1, 60.1, 70.1, 80.1, 90.1, 100]),
            dtype=dtype,
        )
        assert list(a) == [
            0,
            3309,
            6586,
            9862,
            13139,
            16416,
            19692,
            22969,
            26246,
            29523,
            32767,
        ]
        assert a.dtype == dtype

    def test_int32_rescale(self):
        dtype = np.int32
        a = rescale_as_int(
            pd.Series(
                [0, 0.101, 0.201, 0.301, 0.401, 0.501, 0.601, 0.701, 0.801, 0.901, 1]
            ),
            dtype=dtype,
        )
        assert list(a) == [
            0,
            216895848,
            431644213,
            646392577,
            861140942,
            1075889307,
            1290637671,
            1505386036,
            1720134401,
            1934882765,
            2147483647,
        ]
        assert a.dtype == dtype
        a = rescale_as_int(
            pd.Series([0, 10.1, 20.1, 30.1, 40.1, 50.1, 60.1, 70.1, 80.1, 90.1, 100]),
            dtype=dtype,
        )
        assert list(a) == [
            0,
            216895848,
            431644213,
            646392577,
            861140942,
            1075889307,
            1290637671,
            1505386036,
            1720134401,
            1934882765,
            2147483647,
        ]
        assert a.dtype == dtype

    def test_scaling_is_bounded(self):
        a = rescale_as_int(
            pd.Series([-0.2, 1.2]), min_value=0, max_value=1, dtype=np.int8
        )
        assert list(a) == [0, np.iinfo(np.int8).max]
        a = rescale_as_int(
            pd.Series([-0.2, 1.2]), min_value=0, max_value=1, dtype=np.int16
        )
        assert list(a) == [0, np.iinfo(np.int16).max]
        a = rescale_as_int(
            pd.Series([-0.2, 1.2]), min_value=0, max_value=1, dtype=np.int32
        )
        assert list(a) == [0, np.iinfo(np.int32).max]
