__all__ = ["PurgedGroupTimeSeriesSplit"]

import logging
import sys
from typing import Dict, List

import numpy as np

# noinspection PyProtectedMember
from sklearn.model_selection._split import _BaseKFold, _num_samples, indexable

# noinspection PyProtectedMember
from sklearn.utils.validation import _deprecate_positional_args

log = logging.getLogger(__name__)

# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243


# noinspection PyAbstractClass
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=None
        Maximum size for a single training set, as number of groups
    group_gap : int, default=1
        Gap between train and test
    max_test_group_size : int, default=None
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(
        self,
        n_splits=5,
        *,
        max_train_group_size: int = None,
        max_test_group_size: int = None,
        group_gap: int = 1,
    ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.group_gap = group_gap
        self.max_train_group_size = (
            sys.maxsize if max_train_group_size is None else max_train_group_size
        )
        self.max_test_group_size = (
            sys.maxsize if max_test_group_size is None else max_test_group_size
        )

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_folds = self.n_splits + 1
        # np.unique returns sorted groups
        u, ind = np.unique(groups, return_index=True)
        # re-sort unique groups in order of first occurrence
        unique_groups = u[np.argsort(ind)]
        log.debug(f"u={u}, unique_groups={unique_groups}")
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        if n_folds > n_groups:
            raise ValueError(
                (
                    "Cannot have number of folds={0} greater than"
                    " the number of groups={1}"
                ).format(n_folds, n_groups)
            )
        group_dict: Dict[int, List[int]] = {}
        for idx in np.arange(n_samples):
            if groups[idx] in group_dict:
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        group_test_size = min(n_groups // n_folds, self.max_test_group_size)
        group_test_starts = range(
            n_groups - self.n_splits * group_test_size, n_groups, group_test_size
        )
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []
            group_st = max(
                0, group_test_start - self.group_gap - self.max_train_group_size
            )
            log.debug(
                f"group_st={group_st}, group_test_size={group_test_size}, group_test_starts={group_test_starts}"
            )
            for train_group_idx in unique_groups[
                group_st : (group_test_start - self.group_gap)
            ]:
                tmp = group_dict[train_group_idx]
                train_array = np.sort(
                    np.unique(np.concatenate((train_array, tmp)), axis=None),
                    axis=None,
                )
            for test_group_idx in unique_groups[
                group_test_start : group_test_start + group_test_size
            ]:
                tmp = group_dict[test_group_idx]
                test_array = np.sort(
                    np.unique(np.concatenate((test_array, tmp)), axis=None),
                    axis=None,
                )
            yield [int(i) for i in train_array], [int(i) for i in test_array]
