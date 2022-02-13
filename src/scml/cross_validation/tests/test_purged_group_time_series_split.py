from typing import Tuple

import numpy as np
import pytest

from scml.cross_validation import PurgedGroupTimeSeriesSplit


class TestPurgedGroupTimeSeriesSplit:
    @pytest.fixture(scope="class")
    def even_sized_groups(self) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = 20
        X = np.random.random(size=(n_samples, 2))
        # sorted array except for first 2 groups
        groups = np.array([1, 1, 0, 0, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
        return X, groups

    @pytest.fixture(scope="class")
    def uneven_sized_groups(self) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = 20
        X = np.random.random(size=(n_samples, 2))
        # sorted array except for first 2 groups
        groups = np.array([1, 0, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9])
        return X, groups

    def test_group_gap_equals_1_on_even_sized_groups(self, even_sized_groups):
        X, groups = even_sized_groups
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

    def test_group_gap_equals_1_on_uneven_sized_groups(self, uneven_sized_groups):
        X, groups = uneven_sized_groups
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
                assert ti == [0, 1, 2, 3]
                assert set(groups[ti]) == {0, 1, 2, 3}
                assert vi == [6, 7]
                assert set(groups[vi]) == {5}
            if fold == 1:
                assert ti == [0, 1, 2, 3, 4, 5]
                assert set(groups[ti]) == {0, 1, 2, 3, 4}
                assert vi == [8, 9]
                assert set(groups[vi]) == {6}
            if fold == 2:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7]
                assert set(groups[ti]) == {0, 1, 2, 3, 4, 5}
                assert vi == [10, 11, 12]
                assert set(groups[vi]) == {7}
            if fold == 3:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                assert set(groups[ti]) == {0, 1, 2, 3, 4, 5, 6}
                assert vi == [13, 14, 15]
                assert set(groups[vi]) == {8}
            if fold == 4:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                assert set(groups[ti]) == {0, 1, 2, 3, 4, 5, 6, 7}
                assert vi == [16, 17, 18, 19]
                assert set(groups[vi]) == {9}

    def test_group_gap_equals_2(self, even_sized_groups):
        X, groups = even_sized_groups
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

    def test_group_gap_equals_2_on_uneven_sized_groups(self, uneven_sized_groups):
        X, groups = uneven_sized_groups
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
                assert ti == [0, 1, 2]
                assert set(groups[ti]) == {0, 1, 2}
                assert vi == [6, 7]
                assert set(groups[vi]) == {5}
            if fold == 1:
                assert ti == [0, 1, 2, 3]
                assert set(groups[ti]) == {0, 1, 2, 3}
                assert vi == [8, 9]
                assert set(groups[vi]) == {6}
            if fold == 2:
                assert ti == [0, 1, 2, 3, 4, 5]
                assert set(groups[ti]) == {0, 1, 2, 3, 4}
                assert vi == [10, 11, 12]
                assert set(groups[vi]) == {7}
            if fold == 3:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7]
                assert set(groups[ti]) == {0, 1, 2, 3, 4, 5}
                assert vi == [13, 14, 15]
                assert set(groups[vi]) == {8}
            if fold == 4:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                assert set(groups[ti]) == {0, 1, 2, 3, 4, 5, 6}
                assert vi == [16, 17, 18, 19]
                assert set(groups[vi]) == {9}

    def test_max_train_group_size(self, even_sized_groups):
        X, groups = even_sized_groups
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

    def test_max_test_group_size(self, uneven_sized_groups):
        X, groups = uneven_sized_groups
        for fold, (ti, vi) in enumerate(
            PurgedGroupTimeSeriesSplit(
                n_splits=3,
                group_gap=1,
                max_train_group_size=None,
                max_test_group_size=2,
            ).split(X=X, y=None, groups=groups)
        ):
            assert fold != 3
            if fold == 0:
                assert ti == [0, 1, 2]
                assert set(groups[ti]) == {0, 1, 2}
                assert vi == [4, 5, 6, 7]
                assert set(groups[vi]) == {4, 5}
            if fold == 1:
                assert ti == [0, 1, 2, 3, 4, 5]
                assert set(groups[ti]) == {0, 1, 2, 3, 4}
                assert vi == [8, 9, 10, 11, 12]
                assert set(groups[vi]) == {6, 7}
            if fold == 2:
                assert ti == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                assert set(groups[ti]) == {0, 1, 2, 3, 4, 5, 6}
                assert vi == [13, 14, 15, 16, 17, 18, 19]
                assert set(groups[vi]) == {8, 9}
