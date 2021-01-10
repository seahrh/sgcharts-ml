import numpy as np
from scml import fillna


class TestFillna:
    def test_when_nan_is_not_present_then_do_not_fill(self):
        np.testing.assert_array_equal(
            fillna(np.asarray([1, 1]), values=np.asarray([0, 0])), [1, 1]
        )
        np.testing.assert_array_equal(
            fillna(np.asarray([[1, 1], [1, 1]]), values=np.asarray([[0, 0], [0, 0]])),
            [[1, 1], [1, 1]],
        )

    def test_when_nan_is_present_then_do_fill(self):
        np.testing.assert_array_equal(
            fillna(np.asarray([1, np.nan]), values=np.asarray([0, 0])), [1, 0]
        )
        np.testing.assert_array_equal(
            fillna(
                np.asarray([[1, np.nan], [np.nan, 1]]),
                values=np.asarray([[0, 0], [0, 0]]),
            ),
            [[1, 0], [0, 1]],
        )
