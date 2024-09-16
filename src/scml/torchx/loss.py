from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FocalLossForMultiClassClassification"]


class FocalLossForMultiClassClassification(nn.Module):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    For binary classification, please use https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C` (where C is the number of classes) and floating point dtype.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.

    References:
        - https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8

    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2,
        reduction: str = "mean",
    ):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

        Note that this case is equivalent to applying :class:`~torch.nn.LogSoftmax`
        on an input, followed by :class:`~torch.nn.NLLLoss`.
        """
        log_p = F.log_softmax(input, dim=-1)
        p = torch.exp(log_p)
        return F.nll_loss(
            input=((1 - p) ** self.gamma) * log_p,
            target=target,
            weight=self.weight,
            reduction=self.reduction,
        )
