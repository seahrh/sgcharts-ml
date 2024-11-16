import logging

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scml.torchx.loss import *

log = logging.getLogger(__name__)


def _logit(p):
    return torch.log(p / (1 - p))


def _softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # type: ignore[no-any-return]


class TestUncertaintyWeightedLoss:
    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_all_inputs_receive_gradient(self, device, dtype):
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        l1 = torch.tensor([1], dtype=dtype, device=device, requires_grad=True)
        l2 = torch.tensor([100], dtype=dtype, device=device, requires_grad=True)
        v1 = torch.tensor([1], dtype=dtype, device=device, requires_grad=True)
        v2 = torch.tensor([1], dtype=dtype, device=device, requires_grad=True)
        loss = uncertainty_weighted_loss(losses=[l1, l2], log_variances=[v1, v2])
        assert loss.item() > 0
        assert loss.requires_grad
        loss.backward()
        assert l1.grad.item() != 0
        assert l2.grad.item() != 0
        assert v1.grad.item() != 0
        assert v2.grad.item() != 0

    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_some_inputs_receive_gradient(self, device, dtype):
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        l1 = torch.tensor([1], dtype=dtype, device=device, requires_grad=True)
        l2 = torch.tensor([100], dtype=dtype, device=device, requires_grad=False)
        v1 = torch.tensor([1], dtype=dtype, device=device, requires_grad=True)
        v2 = torch.tensor([1], dtype=dtype, device=device, requires_grad=False)
        loss = uncertainty_weighted_loss(losses=[l1, l2], log_variances=[v1, v2])
        assert loss.requires_grad
        assert loss.item() > 0
        loss.backward()
        assert l1.grad.item() != 0
        assert l2.grad is None
        assert v1.grad.item() != 0
        assert v2.grad is None
        l1 = torch.tensor([1], dtype=dtype, device=device, requires_grad=False)
        l2 = torch.tensor([100], dtype=dtype, device=device, requires_grad=False)
        v1 = torch.tensor([1], dtype=dtype, device=device, requires_grad=True)
        v2 = torch.tensor([1], dtype=dtype, device=device, requires_grad=True)
        loss = uncertainty_weighted_loss(losses=[l1, l2], log_variances=[v1, v2])
        assert loss.requires_grad
        assert loss.item() > 0
        loss.backward()
        assert l1.grad is None
        assert l2.grad is None
        assert v1.grad.item() != 0
        assert v2.grad.item() != 0

    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_no_inputs_receive_gradient(self, device, dtype):
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        l1 = torch.tensor([1], dtype=dtype, device=device, requires_grad=False)
        l2 = torch.tensor([100], dtype=dtype, device=device, requires_grad=False)
        v1 = torch.tensor([1], dtype=dtype, device=device, requires_grad=False)
        v2 = torch.tensor([1], dtype=dtype, device=device, requires_grad=False)
        loss = uncertainty_weighted_loss(losses=[l1, l2], log_variances=[v1, v2])
        # a.backward() throws exception because a does not require gradient
        assert not loss.requires_grad


class TestFocalLossForMultiClassClassification:

    @staticmethod
    def _generate_diverse_input_target_pair(**kwargs):
        inputs = _logit(
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
        torch.testing.assert_close(expected_ratio, actual_ratio)

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


class TestSelfAdjustingDiceLoss:

    @staticmethod
    def loss_numpy(
        input: np.ndarray,
        target: np.ndarray,
        reduction: str,
        alpha: float,
        gamma: float,
    ) -> float:
        loss: float = 0.0
        for curr_logits, curr_target in zip(input, target):
            curr_probs = _softmax(curr_logits)
            curr_prob = curr_probs[int(curr_target)]
            prob_with_factor = ((1 - curr_prob) ** alpha) * curr_prob
            curr_loss = 1 - (2 * prob_with_factor + gamma) / (
                prob_with_factor + 1 + gamma
            )
            loss += curr_loss
        if reduction == "mean":
            return loss / int(input.shape[0])
        return loss

    @pytest.mark.parametrize("alpha", [1.0, 2.0, 10.0])
    @pytest.mark.parametrize("gamma", [1.0, 2.0, 10.0])
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_numpy_vs_torch(self, alpha, gamma, reduction):
        N = 128
        C = 64
        input_np = np.random.rand(N, C)
        target_np = np.random.randint(0, C, size=(N,))
        input = torch.from_numpy(input_np)
        target = torch.from_numpy(target_np)
        assert np.allclose(
            self.loss_numpy(
                input=input_np,
                target=target_np,
                reduction=reduction,
                alpha=alpha,
                gamma=gamma,
            ),
            self_adjusting_dice_loss(
                input=input,
                target=target,
                alpha=alpha,
                gamma=gamma,
                reduction=reduction,
            ),
        )
