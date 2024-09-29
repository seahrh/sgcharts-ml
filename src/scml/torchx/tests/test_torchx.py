import copy

import torch
from torch import nn

from scml.torchx import noisy_tune


class SimpleMlp(nn.Module):
    def __init__(self, hidden_units: int = 2):
        super().__init__()
        # param names: seq.0.weight, seq.0.bias
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
