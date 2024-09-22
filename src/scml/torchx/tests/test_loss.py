import pytest
import torch
import torch.nn.functional as F

from scml.torchx import focal_loss_for_multiclass_classification


class TestFocalLossForMultiClassClassification:
    def _generate_diverse_input_target_pair(self, shape=(5, 3), **kwargs):
        def logit(p):
            return torch.log(p / (1 - p))

        def generate_tensor_with_range_type(shape, range_type, **kwargs):
            if range_type != "random_binary":
                low, high = {
                    "small": (0.0, 0.2),
                    "big": (0.8, 1.0),
                    "zeros": (0.0, 0.0),
                    "ones": (1.0, 1.0),
                    "random": (0.0, 1.0),
                }[range_type]
                return torch.testing.make_tensor(shape, low=low, high=high, **kwargs)
            return torch.randint(0, 2, shape, **kwargs)

        # This function will return inputs and targets with shape: (shape[0]*9, shape[1])
        inputs = []
        targets = []
        for input_range_type, target_range_type in [
            ("small", "zeros"),
            ("small", "ones"),
            ("small", "random_binary"),
            ("big", "zeros"),
            ("big", "ones"),
            ("big", "random_binary"),
            ("random", "zeros"),
            ("random", "ones"),
            ("random", "random_binary"),
        ]:
            inputs.append(
                logit(
                    generate_tensor_with_range_type(shape, input_range_type, **kwargs)
                )
            )
            targets.append(
                generate_tensor_with_range_type(shape, target_range_type, **kwargs)
            )

        return torch.cat(inputs), torch.cat(targets)

    @pytest.mark.parametrize("alpha", [-1.0, 0.0, 0.58, 1.0])
    @pytest.mark.parametrize("gamma", [0, 2])
    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_correct_ratio(self, alpha, gamma, device, dtype, seed):
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        # For testing the ratio with manual calculation, we require the reduction to be "none"
        reduction = "none"
        torch.random.manual_seed(seed)
        inputs, targets = self._generate_diverse_input_target_pair(
            dtype=dtype, device=device
        )
        focal_loss = focal_loss_for_multiclass_classification(
            input=inputs, target=targets, gamma=gamma, reduction=reduction
        )
        # focal_loss = loss_fn(inputs, targets)
        # focal_loss = ops.sigmoid_focal_loss(
        #    inputs, targets, gamma=gamma, alpha=alpha, reduction=reduction
        # )
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction=reduction
        )
        assert torch.all(
            focal_loss <= ce_loss
        ), "focal loss must be less or equal to cross entropy loss with same input"
        loss_ratio = (focal_loss / ce_loss).squeeze()
        prob = torch.sigmoid(inputs)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        correct_ratio = (1.0 - p_t) ** gamma
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            correct_ratio = correct_ratio * alpha_t
        tol = 1e-3 if dtype is torch.half else 1e-5
        torch.testing.assert_close(correct_ratio, loss_ratio, atol=tol, rtol=tol)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    @pytest.mark.parametrize("seed", [2, 3])
    def test_equal_ce_loss(self, reduction, device, dtype, seed):
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        # focal loss should be equal ce_loss if alpha=-1 and gamma=0
        # alpha = -1
        gamma = 0
        torch.random.manual_seed(seed)
        inputs, targets = self._generate_diverse_input_target_pair(
            dtype=dtype, device=device
        )
        inputs_fl = inputs.clone().requires_grad_()
        targets_fl = targets.clone()
        inputs_ce = inputs.clone().requires_grad_()
        targets_ce = targets.clone()
        focal_loss = focal_loss_for_multiclass_classification(
            input=inputs_fl, target=targets_fl, gamma=gamma, reduction=reduction
        )
        # focal_loss = loss_fn(inputs_fl, targets_fl)
        # focal_loss = ops.sigmoid_focal_loss(
        #    inputs_fl, targets_fl, gamma=gamma, alpha=alpha, reduction=reduction
        # )
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs_ce, targets_ce, reduction=reduction
        )
        torch.testing.assert_close(focal_loss, ce_loss)
        focal_loss.backward()
        ce_loss.backward()
        torch.testing.assert_close(inputs_fl.grad, inputs_ce.grad)

    # Raise ValueError for anonymous reduction mode
    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_reduction_mode(self, device, dtype, reduction="xyz"):
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        torch.random.manual_seed(0)
        inputs, targets = self._generate_diverse_input_target_pair(
            device=device, dtype=dtype
        )
        with pytest.raises(ValueError, match="Invalid"):
            focal_loss_for_multiclass_classification(
                input=inputs, target=targets, reduction=reduction
            )
            # ops.sigmoid_focal_loss(inputs, targets, 0.25, 2, reduction)
