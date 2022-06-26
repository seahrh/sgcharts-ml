import numpy as np
import pandas as pd

from scml.pandasx import quantize


class TestQuantize:
    def test_int8_conversion(self):
        df = pd.DataFrame.from_dict(
            {"col1": [np.iinfo(np.int8).min + 1, np.iinfo(np.int8).max - 1]}
        )
        quantize(df, verbose=False)
        assert df["col1"].dtype == np.int8

    def test_int16_conversion(self):
        df = pd.DataFrame.from_dict(
            {"col1": [np.iinfo(np.int16).min + 1, np.iinfo(np.int16).max - 1]}
        )
        quantize(df, verbose=False)
        assert df["col1"].dtype == np.int16

    def test_int32_conversion(self):
        df = pd.DataFrame.from_dict(
            {"col1": [np.iinfo(np.int32).min + 1, np.iinfo(np.int32).max - 1]}
        )
        quantize(df, verbose=False)
        assert df["col1"].dtype == np.int32

    def test_int64_conversion(self):
        df = pd.DataFrame.from_dict(
            {"col1": [np.iinfo(np.int64).min + 1, np.iinfo(np.int64).max - 1]}
        )
        quantize(df, verbose=False)
        assert df["col1"].dtype == np.int64

    def test_float16_remains_the_same(self):
        df = pd.DataFrame.from_dict({"col1": [-9e-6, 9e-6]}, dtype=np.float16)
        quantize(df, verbose=False)
        assert df["col1"].dtype == np.float16

    def test_float32_conversion(self):
        df = pd.DataFrame.from_dict({"col1": [-9e-20, 9e-20]})
        quantize(df, verbose=False)
        p_max = np.finfo(np.float32).precision
        p = df["col1"].apply(lambda x: np.finfo(x).precision).max()
        print(f"p_max={p_max}, p={p}")
        assert df["col1"].dtype == np.float32
