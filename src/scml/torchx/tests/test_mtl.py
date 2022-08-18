import torch

from scml.torchx import uncertainty_weighted_loss


class TestUncertaintyWeightedLoss:
    def test_all_inputs_receive_gradient(self):
        l1 = torch.tensor([1], dtype=torch.float32, requires_grad=True)
        l2 = torch.tensor([100], dtype=torch.float32, requires_grad=True)
        v1 = torch.tensor([0], dtype=torch.float32, requires_grad=True)
        v2 = torch.tensor([0], dtype=torch.float32, requires_grad=True)
        a = uncertainty_weighted_loss(losses=[l1, l2], log_variances=[v1, v2])
        assert a.requires_grad
        assert torch.equal(a, torch.tensor([101.0]))
        a.backward()
        assert torch.equal(l1.grad, torch.tensor([1.0]))
        assert torch.equal(l2.grad, torch.tensor([1.0]))
        assert torch.equal(v1.grad, torch.tensor([0.0]))
        assert torch.equal(v2.grad, torch.tensor([-99.0]))

    def test_some_inputs_receive_gradient(self):
        l1 = torch.tensor([1], dtype=torch.float32, requires_grad=True)
        l2 = torch.tensor([100], dtype=torch.float32, requires_grad=False)
        v1 = torch.tensor([0], dtype=torch.float32, requires_grad=True)
        v2 = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        a = uncertainty_weighted_loss(losses=[l1, l2], log_variances=[v1, v2])
        assert a.requires_grad
        assert torch.equal(a, torch.tensor([101.0]))
        a.backward()
        assert torch.equal(l1.grad, torch.tensor([1.0]))
        assert l2.grad is None
        assert torch.equal(v1.grad, torch.tensor([0.0]))
        assert v2.grad is None

    def test_no_inputs_receive_gradient(self):
        l1 = torch.tensor([1], dtype=torch.float32, requires_grad=False)
        l2 = torch.tensor([100], dtype=torch.float32, requires_grad=False)
        v1 = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        v2 = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        a = uncertainty_weighted_loss(losses=[l1, l2], log_variances=[v1, v2])
        # a.backward() throws exception because a does not require gradient
        assert not a.requires_grad
        assert torch.equal(a, torch.tensor([101.0]))
