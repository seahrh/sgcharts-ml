import math
import numpy as np
import pytest
from krig import MultilabelStratifiedKFold


def _test(y, n_samples, n_splits):
    x = np.zeros((n_samples, 2))  # This is not used in the split
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True)
    for train_index, test_index in mskf.split(x, y):
        assert len(set(train_index) & set(test_index)) == 0
        y_train = np.sum(y[train_index], axis=0)  # sum column-wise
        y_train = y_train / np.sum(y_train)
        y_test = np.sum(y[test_index], axis=0)
        y_test = y_test / np.sum(y_test)
        for r1, r2 in zip(y_train, y_test):
            assert r1 > 0
            assert r2 > 0
            assert math.fabs(r1 - r2) < .08


class TestMultilabelStratifiedKFold:

    def test_label_proportion_must_be_similar_in_each_split(self):
        n_samples = 200
        labels = [3, 10, 30]
        folds = [2, 3, 5, 10]
        for n_labels in labels:
            y = np.random.randint(0, 2, size=(n_samples, n_labels))
            for n_splits in folds:
                _test(y, n_samples=n_samples, n_splits=n_splits)

    def test_when_fold_has_no_positive_examples_then_raise_error(self):
        n_samples = 200
        n_labels = 2
        n_splits = 2
        y = np.zeros(shape=(n_samples, n_labels))
        match = r'at least \d+ positive examples'
        with pytest.raises(ValueError, match=match):
            _test(y, n_samples=n_samples, n_splits=n_splits)
        y[0, 0], y[0, 1] = 1, 1
        with pytest.raises(ValueError, match=match):
            _test(y, n_samples=n_samples, n_splits=n_splits)

    def test_at_least_one_positive_example_in_each_fold(self):
        n_samples = 200
        labels = [3, 10, 30]
        folds = [2, 3, 5, 10]
        for n_labels in labels:
            y = np.zeros(shape=(n_samples, n_labels))
            for n_splits in folds:
                # number of positive examples same as number of folds
                for i in range(n_splits):
                    for j in range(n_labels):
                        y[i, j] = 1
                _test(y, n_samples=n_samples, n_splits=n_splits)
