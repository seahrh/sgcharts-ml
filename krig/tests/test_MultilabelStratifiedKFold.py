import math
import numpy as np

from krig import MultilabelStratifiedKFold


class TestMultilabelStratifiedKFold:

    def test_label_proportion_in_each_fold(self):

        # noinspection PyShadowingNames
        def _test(n_labels, n_samples, n_splits, is_proba):
            x = np.zeros((n_samples, 2))  # This is not used in the split
            if is_proba:
                y = np.random.uniform(0.0, 1.0000000001, size=(n_samples, n_labels))
            else:
                y = np.random.randint(0, 2, size=(n_samples, n_labels))
            mskf = MultilabelStratifiedKFold(n_splits=n_splits)

            for train_index, test_index in mskf.split(x, y):
                assert len(set(train_index) & set(test_index)) == 0
                y_train = np.sum(y[train_index], axis=0)  # sum column-wise
                y_train = y_train / np.sum(y_train)
                y_test = np.sum(y[test_index], axis=0)
                y_test = y_test / np.sum(y_test)
                for r1, r2 in zip(y_train, y_test):
                    if is_proba:
                        assert math.fabs(r1 - r2) <= .12
                    else:
                        assert math.fabs(r1 - r2) <= .07

        for is_proba in [False, True]:
            _test(3, 200, n_splits=3, is_proba=is_proba)
            _test(10, 200, n_splits=3, is_proba=is_proba)
            _test(30, 200, n_splits=3, is_proba=is_proba)

            _test(3, 200, n_splits=5, is_proba=is_proba)
            _test(10, 200, n_splits=5, is_proba=is_proba)
            _test(30, 200, n_splits=5, is_proba=is_proba)

            _test(3, 200, n_splits=10, is_proba=is_proba)
            _test(10, 200, n_splits=10, is_proba=is_proba)
            _test(30, 200, n_splits=10, is_proba=is_proba)
