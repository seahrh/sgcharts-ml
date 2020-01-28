import math
import numpy as np

from krig import MultilabelStratifiedKFold


def _test(n_labels, n_samples, n_splits):
    x = np.zeros((n_samples, 2))  # This is not used in the split
    y = np.random.randint(0, 2, size=(n_samples, n_labels))
    mskf = MultilabelStratifiedKFold(n_splits=n_splits)

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
            for n_splits in folds:
                _test(n_labels=n_labels, n_samples=n_samples, n_splits=n_splits)
