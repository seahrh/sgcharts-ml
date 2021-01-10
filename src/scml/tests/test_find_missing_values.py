import pandas as pd
import numpy as np
from scml import find_missing_values


class TestFindMissingValues:
    def test_no_missing_values(self):
        df = pd.DataFrame(
            {
                "integers": [1, 2, 3, np.inf],
                "floats": [0.1, 0.2, 0.3, np.inf],
                "strings": ["", " ", "c", "d"],
                "timestamps": [
                    pd.Timestamp("2017-01-01"),
                    pd.Timestamp("2018-01-01"),
                    pd.Timestamp("2019-01-01"),
                    pd.Timestamp("2020-01-01"),
                ],
            }
        )
        a = find_missing_values(df, blank_strings_as_null=False).to_dict()
        assert a["Total"] == {"integers": 0, "floats": 0, "strings": 0, "timestamps": 0}
        assert a["Percent"] == {
            "integers": 0,
            "floats": 0,
            "strings": 0,
            "timestamps": 0,
        }

    def test_missing_values_are_present(self):
        df = pd.DataFrame(
            {
                "integers": [1, None, np.NaN, np.inf],
                "floats": [0.1, None, np.NaN, np.inf],
                "strings": ["", " ", "c", None],
                "timestamps": [
                    pd.Timestamp("2017-01-01"),
                    pd.Timestamp("2018-01-01"),
                    None,
                    pd.NaT,
                ],
            }
        )
        a = find_missing_values(df, blank_strings_as_null=False).to_dict()
        assert a["Total"] == {"integers": 2, "floats": 2, "strings": 1, "timestamps": 2}
        assert a["Percent"] == {
            "integers": 0.5,
            "floats": 0.5,
            "strings": 0.25,
            "timestamps": 0.5,
        }

    def test_treat_empty_strings_and_inf_as_na(self):
        pd.options.mode.use_inf_as_na = True
        df = pd.DataFrame(
            {
                "integers": [1, None, np.NaN, np.inf],
                "floats": [0.1, None, np.NaN, np.inf],
                "strings": ["", " ", "c", None],
                "timestamps": [
                    pd.Timestamp("2017-01-01"),
                    pd.Timestamp("2018-01-01"),
                    None,
                    pd.NaT,
                ],
            }
        )
        a = find_missing_values(df, blank_strings_as_null=True).to_dict()
        assert a["Total"] == {"integers": 3, "floats": 3, "strings": 3, "timestamps": 2}
        assert a["Percent"] == {
            "integers": 0.75,
            "floats": 0.75,
            "strings": 0.75,
            "timestamps": 0.5,
        }
        pd.options.mode.use_inf_as_na = False
