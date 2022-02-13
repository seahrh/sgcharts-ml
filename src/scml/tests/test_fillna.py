import numpy as np

from scml import fillna


class TestFillna:
    def test_when_nan_is_not_present_then_do_not_fill_1d_array(self):
        np.testing.assert_allclose(
            fillna(np.array([1.2, 1.2]), values=np.array([0, 0]), add_flag=False),
            [1.2, 1.2],
        )
        np.testing.assert_allclose(
            fillna(np.array([1.2, 1.2]), values=np.array([0, 0]), add_flag=True),
            [1.2, 1.2, 0, 0],
        )

    def test_when_nan_is_not_present_then_do_not_fill_2d_array(self):
        np.testing.assert_allclose(
            fillna(
                np.array([[1.2, 1.2], [1.2, 1.2]]),
                values=np.array([[0, 0], [0, 0]]),
                add_flag=False,
            ),
            [[1.2, 1.2], [1.2, 1.2]],
        )
        np.testing.assert_allclose(
            fillna(
                np.array([[1.2, 1.2], [1.2, 1.2]]),
                values=np.array([[0, 0], [0, 0]]),
                add_flag=True,
            ),
            [[1.2, 1.2, 0, 0], [1.2, 1.2, 0, 0]],
        )

    def test_when_nan_is_present_then_do_fill_1d_array(self):
        np.testing.assert_allclose(
            fillna(np.array([1.2, np.nan]), values=np.array([0, 0]), add_flag=False),
            [1.2, 0],
        )
        np.testing.assert_allclose(
            fillna(np.array([1.2, np.nan]), values=np.array([0, 0]), add_flag=True),
            [1.2, 0, 0, 1],
        )

    def test_when_nan_is_present_then_do_fill_2d_array(self):
        np.testing.assert_allclose(
            fillna(
                np.array([[1.2, np.nan], [np.nan, 1.2]]),
                values=np.array([[0, 0], [0, 0]]),
                add_flag=False,
            ),
            [[1.2, 0], [0, 1.2]],
        )
        np.testing.assert_allclose(
            fillna(
                np.array([[1.2, np.nan], [np.nan, 1.2]]),
                values=np.array([[0, 0], [0, 0]]),
                add_flag=True,
            ),
            [[1.2, 0, 0, 1], [0, 1.2, 1, 0]],
        )
