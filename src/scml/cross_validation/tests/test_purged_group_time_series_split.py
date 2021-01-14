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
                assert set(groups[ti]) == {0, 1, 2, 3}
                assert vi == [10, 11]
                assert set(groups[vi]) == {5}
            if fold == 1:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                assert set(groups[ti]) == {0, 1, 2, 3, 4}
                assert vi == [12, 13]
                assert set(groups[vi]) == {6}
            if fold == 2:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                assert set(groups[ti]) == {0, 1, 2, 3, 4, 5}
                assert vi == [14, 15]
                assert set(groups[vi]) == {7}
            if fold == 3:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                assert set(groups[ti]) == {0, 1, 2, 3, 4, 5, 6}
                assert vi == [16, 17]
                assert set(groups[vi]) == {8}
            if fold == 4:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                assert set(groups[ti]) == {0, 1, 2, 3, 4, 5, 6, 7}
                assert vi == [18, 19]
                assert set(groups[vi]) == {9}

    def test_group_gap_equals_2(self, even_group_sizes):
        X, groups = even_group_sizes
        for fold, (ti, vi) in enumerate(
            PurgedGroupTimeSeriesSplit(
                n_splits=5,
                group_gap=2,
                max_train_group_size=None,
                max_test_group_size=None,
            ).split(X=X, y=None, groups=groups)
        ):
            assert fold != 5
            if fold == 0:
                assert ti == [0, 1, 2, 3, 4, 5]
                assert set(groups[ti]) == {0, 1, 2}
                assert vi == [10, 11]
                assert set(groups[vi]) == {5}
            if fold == 1:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7]
                assert set(groups[ti]) == {0, 1, 2, 3}
                assert vi == [12, 13]
                assert set(groups[vi]) == {6}
            if fold == 2:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                assert set(groups[ti]) == {0, 1, 2, 3, 4}
                assert vi == [14, 15]
                assert set(groups[vi]) == {7}
            if fold == 3:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                assert set(groups[ti]) == {0, 1, 2, 3, 4, 5}
                assert vi == [16, 17]
                assert set(groups[vi]) == {8}
            if fold == 4:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                assert set(groups[ti]) == {0, 1, 2, 3, 4, 5, 6}
                assert vi == [18, 19]
                assert set(groups[vi]) == {9}

    def test_max_train_group_size(self, even_group_sizes):
        X, groups = even_group_sizes
        for fold, (ti, vi) in enumerate(
            PurgedGroupTimeSeriesSplit(
                n_splits=5,
                group_gap=1,
                max_train_group_size=2,
                max_test_group_size=None,
            ).split(X=X, y=None, groups=groups)
        ):
            assert fold != 5
            if fold == 0:
                assert ti == [4, 5, 6, 7]
                assert set(groups[ti]) == {2, 3}
                assert vi == [10, 11]
                assert set(groups[vi]) == {5}
            if fold == 1:
                assert ti == [6, 7, 8, 9]
                assert set(groups[ti]) == {3, 4}
                assert vi == [12, 13]
                assert set(groups[vi]) == {6}
            if fold == 2:
                assert ti == [8, 9, 10, 11]
                assert set(groups[ti]) == {4, 5}
                assert vi == [14, 15]
                assert set(groups[vi]) == {7}
            if fold == 3:
                assert ti == [10, 11, 12, 13]
                assert set(groups[ti]) == {5, 6}
                assert vi == [16, 17]
                assert set(groups[vi]) == {8}
            if fold == 4:
                assert ti == [12, 13, 14, 15]
                assert set(groups[ti]) == {6, 7}
                assert vi == [18, 19]
                assert set(groups[vi]) == {9}
