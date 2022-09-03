import copy

import pytest
import torch
from torch import nn

from scml.torchx import noisy_tune, uncertainty_weighted_loss


@pytest.fixture
def rtol() -> float:
    return 1e-4


class SimpleMlp(nn.Module):
    def __init__(self, hidden_units: int = 2):
        super().__init__()
        # seq.0.weight
        self.seq = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.Linear(hidden_units, hidden_units),
            nn.Linear(hidden_units, hidden_units),
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class TestNoisyTune:
    def test_change_all_weights(self):
        model = SimpleMlp(hidden_units=2)
        initial_state = copy.deepcopy(model.state_dict())
        noisy_tune(model=model, noise_intensity=0.15, prefixes=None)
        final_state = model.state_dict()
        assert initial_state.keys() == final_state.keys()
        assert not torch.equal(
            initial_state["seq.0.weight"], final_state["seq.0.weight"]
        )
        assert not torch.equal(initial_state["seq.0.bias"], final_state["seq.0.bias"])
        assert not torch.equal(
            initial_state["seq.1.weight"], final_state["seq.1.weight"]
        )
        assert not torch.equal(initial_state["seq.1.bias"], final_state["seq.1.bias"])
        assert not torch.equal(
            initial_state["seq.2.weight"], final_state["seq.2.weight"]
        )
        assert not torch.equal(initial_state["seq.2.bias"], final_state["seq.2.bias"])

    def test_change_weights_when_name_matches_prefix(self):
        model = SimpleMlp(hidden_units=2)
        initial_state = copy.deepcopy(model.state_dict())
        noisy_tune(
            model=model, noise_intensity=0.15, prefixes={"seq.0", "seq.1.weight"}
        )
        final_state = model.state_dict()
        assert initial_state.keys() == final_state.keys()
        assert not torch.equal(
            initial_state["seq.0.weight"], final_state["seq.0.weight"]
        )
        assert not torch.equal(initial_state["seq.0.bias"], final_state["seq.0.bias"])
        assert not torch.equal(
            initial_state["seq.1.weight"], final_state["seq.1.weight"]
        )
        assert torch.equal(initial_state["seq.1.bias"], final_state["seq.1.bias"])
        assert torch.equal(initial_state["seq.2.weight"], final_state["seq.2.weight"])
        assert torch.equal(initial_state["seq.2.bias"], final_state["seq.2.bias"])

    def test_change_no_weights_when_name_does_not_match_prefix(self):
        model = SimpleMlp(hidden_units=2)
        initial_state = copy.deepcopy(model.state_dict())
        noisy_tune(model=model, noise_intensity=0.15, prefixes={"foo"})
        final_state = model.state_dict()
        assert initial_state.keys() == final_state.keys()
        assert torch.equal(initial_state["seq.0.weight"], final_state["seq.0.weight"])
        assert torch.equal(initial_state["seq.0.bias"], final_state["seq.0.bias"])
        assert torch.equal(initial_state["seq.1.weight"], final_state["seq.1.weight"])
        assert torch.equal(initial_state["seq.1.bias"], final_state["seq.1.bias"])
        assert torch.equal(initial_state["seq.2.weight"], final_state["seq.2.weight"])
        assert torch.equal(initial_state["seq.2.bias"], final_state["seq.2.bias"])

    def test_change_no_weights_when_noise_is_zero(self):
        model = SimpleMlp(hidden_units=2)
        initial_state = copy.deepcopy(model.state_dict())
        noisy_tune(model=model, noise_intensity=0, prefixes=None)
        final_state = model.state_dict()
        assert initial_state.keys() == final_state.keys()
        assert torch.equal(initial_state["seq.0.weight"], final_state["seq.0.weight"])
        assert torch.equal(initial_state["seq.0.bias"], final_state["seq.0.bias"])
        assert torch.equal(initial_state["seq.1.weight"], final_state["seq.1.weight"])
        assert torch.equal(initial_state["seq.1.bias"], final_state["seq.1.bias"])
        assert torch.equal(initial_state["seq.2.weight"], final_state["seq.2.weight"])
        assert torch.equal(initial_state["seq.2.bias"], final_state["seq.2.bias"])


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
