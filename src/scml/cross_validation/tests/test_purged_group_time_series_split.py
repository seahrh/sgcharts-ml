import numpy as np
import pytest
from typing import Tuple
from scml.cross_validation import PurgedGroupTimeSeriesSplit


class TestPurgedGroupTimeSeriesSplit:
    @pytest.fixture(scope="class")
    def even_group_sizes(self) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = 20
        n_groups = 10
        X = np.random.random(size=(n_samples, 2))
        groups = np.repeat(
            np.linspace(0, n_groups - 1, num=n_groups), n_samples / n_groups
        )
        return X, groups

    def test_group_gap_equals_1(self, even_group_sizes):
        X, groups = even_group_sizes
        for fold, (ti, vi) in enumerate(
            PurgedGroupTimeSeriesSplit(
                n_splits=5,
                group_gap=1,
                max_train_group_size=None,
                max_test_group_size=None,
            ).split(X=X, y=None, groups=groups)
        ):
            assert fold != 5
            if fold == 0:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7]
                np.testing.assert_array_equal(groups[ti], [0, 0, 1, 1, 2, 2, 3, 3])
                assert vi == [11]
                np.testing.assert_array_equal(groups[vi], [5])
            if fold == 1:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                np.testing.assert_array_equal(
                    groups[ti], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
                )
                assert vi == [13]
                np.testing.assert_array_equal(groups[vi], [6])
            if fold == 2:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                np.testing.assert_array_equal(
                    groups[ti], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
                )
                assert vi == [15]
                np.testing.assert_array_equal(groups[vi], [7])
            if fold == 3:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                np.testing.assert_array_equal(
                    groups[ti], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
                )
                assert vi == [17]
                np.testing.assert_array_equal(groups[vi], [8])
            if fold == 4:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                np.testing.assert_array_equal(
                    groups[ti], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
                )
                assert vi == [19]
                np.testing.assert_array_equal(groups[vi], [9])
