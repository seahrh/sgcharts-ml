import logging

import pytest
import torch
import torch.nn.functional as F

from scml.torchx import focal_loss_for_multiclass_classification

log = logging.getLogger(__name__)


class TestFocalLossForMultiClassClassification:

    @staticmethod
    def _logit(p):
        return torch.log(p / (1 - p))

    def _generate_diverse_input_target_pair(self, **kwargs):
        inputs = self._logit(
            torch.tensor(
                [
                    [0.1, 0.2, 0.3],
                    [0.1, 0.2, 0.3],
                    [0.1, 0.2, 0.3],
                    [0.7, 0.8, 0.9],
                    [0.7, 0.8, 0.9],
                    [0.7, 0.8, 0.9],
                    [0.4, 0.5, 0.6],
                    [0.4, 0.5, 0.6],
                    [0.4, 0.5, 0.6],
                ],
                **kwargs,
            )
        )
        targets = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            **kwargs,
        )
        targets = targets.long()
        return inputs, targets

    @pytest.mark.parametrize("gamma", [0, 2])
    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_correct_ratio(self, gamma, device, dtype):
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        # For testing the ratio with manual calculation, we require the reduction to be "none"
        reduction = "none"
        inputs, targets = self._generate_diverse_input_target_pair(
            dtype=dtype, device=device
        )
        targets_indices = torch.argmax(targets, dim=-1)
        ce_loss = F.cross_entropy(
            input=inputs, target=targets_indices, reduction=reduction
        )
        focal_loss = focal_loss_for_multiclass_classification(
            input=inputs, target=targets_indices, gamma=gamma, reduction=reduction
        )
        assert torch.all(
            focal_loss <= ce_loss
        ), "focal loss must be less or equal to cross entropy loss with same input"
        actual_ratio = focal_loss / ce_loss
        log_p = F.log_softmax(inputs, dim=-1)
        # p_t is the model's estimated probability of the ground truth class
        p_t = torch.exp(log_p) * targets  # (samples, classes)
        p_t = p_t.masked_select((p_t > 0))  # (samples,)
        expected_ratio = (1.0 - p_t) ** gamma
        log.debug(
            f"""p_t={p_t}, targets={targets}
p_t={p_t.shape}, focal_loss={focal_loss.shape}, ce_loss={ce_loss.shape}
correct_ratio={expected_ratio.shape}, loss_ratio={actual_ratio.shape}"""
        )
        tol = 1e-3 if dtype is torch.half else 1e-5
        torch.testing.assert_close(expected_ratio, actual_ratio, atol=tol, rtol=tol)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_focal_loss_equals_ce_loss_when_gamma_is_zero(
        self, reduction, device, dtype
    ):
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        inputs, targets = self._generate_diverse_input_target_pair(
            dtype=dtype, device=device
        )
        inputs_fl = inputs.clone().requires_grad_()
        targets_fl = torch.argmax(targets.clone(), dim=-1)
        inputs_ce = inputs.clone().requires_grad_()
        targets_ce = torch.argmax(targets.clone(), dim=-1)
        focal_loss = focal_loss_for_multiclass_classification(
            input=inputs_fl, target=targets_fl, gamma=0, reduction=reduction
        )
        ce_loss = F.cross_entropy(
            input=inputs_ce, target=targets_ce, reduction=reduction
        )
        torch.testing.assert_close(focal_loss, ce_loss)
        focal_loss.backward()
        ce_loss.backward()
        torch.testing.assert_close(inputs_fl.grad, inputs_ce.grad)

    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_reduction_mode(self, device, dtype, reduction="xyz"):
        # Raise ValueError for anonymous reduction mode
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        inputs, targets = self._generate_diverse_input_target_pair(
            device=device, dtype=dtype
        )
        with pytest.raises(ValueError, match="valid value for reduction"):
            focal_loss_for_multiclass_classification(
                input=inputs, target=targets, reduction=reduction
            )
