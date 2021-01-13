from scml.cross_validation import PurgedGroupTimeSeriesSplit


class TestPurgedGroupTimeSeriesSplit:
    def test_it(self):
        PurgedGroupTimeSeriesSplit(
            n_splits=5, group_gap=1, max_train_group_size=None, max_test_group_size=None
        )
