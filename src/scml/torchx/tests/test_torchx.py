import pytest
import torch

from scml.torchx import uncertainty_weighted_loss


@pytest.fixture
def rtol() -> float:
    return 1e-4


class TestUncertaintyWeightedLoss:
    def test_all_inputs_receive_gradient(self, rtol):
        l1 = torch.tensor([1], dtype=torch.float32, requires_grad=True)
        l2 = torch.tensor([100], dtype=torch.float32, requires_grad=True)
        v1 = torch.tensor([1], dtype=torch.float32, requires_grad=True)
        v2 = torch.tensor([1], dtype=torch.float32, requires_grad=True)
        a = uncertainty_weighted_loss(losses=[l1, l2], log_variances=[v1, v2])
        assert a.requires_grad
        assert torch.allclose(a, torch.tensor([39.1558]), rtol=rtol)
        a.backward()
        assert torch.allclose(l1.grad, torch.tensor([0.3679]), rtol=rtol)
        assert torch.allclose(l2.grad, torch.tensor([0.3679]), rtol=rtol)
        assert torch.allclose(v1.grad, torch.tensor([1.2642]), rtol=rtol)
        assert torch.allclose(v2.grad, torch.tensor([-71.5759]), rtol=rtol)

    def test_some_inputs_receive_gradient(self, rtol):
        l1 = torch.tensor([1], dtype=torch.float32, requires_grad=True)
        l2 = torch.tensor([100], dtype=torch.float32, requires_grad=False)
        v1 = torch.tensor([1], dtype=torch.float32, requires_grad=True)
        v2 = torch.tensor([1], dtype=torch.float32, requires_grad=False)
        a = uncertainty_weighted_loss(losses=[l1, l2], log_variances=[v1, v2])
        assert a.requires_grad
        assert torch.allclose(a, torch.tensor([39.1558]), rtol=rtol)
        a.backward()
        assert torch.allclose(l1.grad, torch.tensor([0.3679]), rtol=rtol)
        assert l2.grad is None
        assert torch.allclose(v1.grad, torch.tensor([1.2642]), rtol=rtol)
        assert v2.grad is None
        l1 = torch.tensor([1], dtype=torch.float32, requires_grad=False)
        l2 = torch.tensor([100], dtype=torch.float32, requires_grad=False)
        v1 = torch.tensor([1], dtype=torch.float32, requires_grad=True)
        v2 = torch.tensor([1], dtype=torch.float32, requires_grad=True)
        a = uncertainty_weighted_loss(losses=[l1, l2], log_variances=[v1, v2])
        assert a.requires_grad
        assert torch.allclose(a, torch.tensor([39.1558]), rtol=rtol)
        a.backward()
        assert l1.grad is None
        assert l2.grad is None
        assert torch.allclose(v1.grad, torch.tensor([1.2642]), rtol=rtol)
        assert torch.allclose(v2.grad, torch.tensor([-71.5759]), rtol=rtol)

    def test_no_inputs_receive_gradient(self, rtol):
        l1 = torch.tensor([1], dtype=torch.float32, requires_grad=False)
        l2 = torch.tensor([100], dtype=torch.float32, requires_grad=False)
        v1 = torch.tensor([1], dtype=torch.float32, requires_grad=False)
        v2 = torch.tensor([1], dtype=torch.float32, requires_grad=False)
        a = uncertainty_weighted_loss(losses=[l1, l2], log_variances=[v1, v2])
        # a.backward() throws exception because a does not require gradient
        assert not a.requires_grad
        assert torch.allclose(a, torch.tensor([39.1558]), rtol=rtol)
