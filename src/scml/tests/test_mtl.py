from scml import uncertainty_weighted_loss


class TestUncertaintyWeightedLoss:
    def test_zero_loss(self):
        assert uncertainty_weighted_loss(losses=[0, 0], log_variances=[0, 0]) == 0
        assert uncertainty_weighted_loss(losses=[0, 0], log_variances=[1, 1]) == 1

    def test_nonzero_loss(self):
        assert uncertainty_weighted_loss(losses=[1, 1], log_variances=[0, 0]) == 1
