from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "focal_loss_for_multiclass_classification",
    "uncertainty_weighted_loss",
    "self_adjusting_dice_loss",
]


def uncertainty_weighted_loss(
    losses: Sequence[torch.Tensor], log_variances: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Based on Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (Kendall 2018).
    Log variance represents the uncertainty. The higher the uncertainty, the smaller the weight.
    To prevent the model from simply suppressing all weights to zero, add the uncertainty to final loss.

    https://github.com/yaringal/multi-task-learning-example
    """
    if len(losses) == 0:
        raise ValueError("losses must not be empty")
    if len(losses) != len(log_variances):
        raise ValueError("Length of losses must equal log_variances")
    sm = torch.zeros((1,), dtype=torch.float32, device=log_variances[0].device)
    for i in range(len(losses)):
        # square to prevent negative sum
        lv = torch.pow(log_variances[i], 2)
        precision = torch.exp(-lv)
        sm += precision * losses[i] + lv
    return sm


def focal_loss_for_multiclass_classification(
    input: Tensor,
    target: Tensor,
    gamma: float = 2.0,
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    r"""Compute the focal loss for multi-class classification.
    For binary classification, please use https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html

    Based on the paper: Lin, T. (2017). Focal Loss for Dense Object Detection. arXiv preprint arXiv:1708.02002.

    Args:
        input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss. `input` is expected to be log-probabilities.
        target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
            or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
            K-dimensional loss.
        gamma (int, optional): When an example is misclassified and p_t is small,
            the modulating factor is near 1 and the loss is unaffected. As pt → 1,
            the factor goes to 0 and the loss for well-classified examples is down-weighted.
            The focusing parameter γ smoothly adjusts the rate at which easy examples are down-weighted.
            When γ = 0, FL is equivalent to CE, and as γ is increased,
            the effect of the modulating factor is likewise increased
            (we found γ = 2 to work best in our experiments)
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: -100
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Example::

        >>> # input is of size N x C = 3 x 5
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> # each element in target has to have 0 <= value < C
        >>> target = torch.tensor([1, 0, 4])
        >>> output = focal_loss_for_multiclass_classification(F.log_softmax(input, dim=1), target)
        >>> output.backward()
    """
    # Cross entropy loss is equivalent to applying :class:`~torch.nn.LogSoftmax`
    # on an input, followed by :class:`~torch.nn.NLLLoss`.
    # See :class:`~torch.nn.CrossEntropyLoss` for details.
    log_p = F.log_softmax(input, dim=-1)
    # p_t is the model's estimated probability of the ground truth class
    p_t = torch.exp(log_p)
    return F.nll_loss(
        input=((1 - p_t) ** gamma) * log_p,
        target=target,
        weight=weight,
        reduction=reduction,
        ignore_index=ignore_index,
    )


def self_adjusting_dice_loss(
    input: Tensor,
    target: Tensor,
    alpha: float = 2.0,
    gamma: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss.
    Paper: Li, Xiaoya, et al. "Dice loss for data-imbalanced NLP tasks." (ACL 2020)
    Based on https://github.com/fursovia/self-adj-dice/

    Args:
        alpha (float): a factor to push down the weight of easy examples
            The "Focal loss" paper sets alpha to a default value of 2.
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes.
            This allows negative examples to contribute to the training.
            The paper sets gamma to a default value of 1.
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.

    Shape:
        - input: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - target: `(N)` where each value is in [0, C - 1]
    """
    probs = torch.softmax(input, dim=1)
    # Unsqueeze reshapes `target` to (N, 1)
    # Gather gets the probability of the ground truth class
    probs = torch.gather(probs, dim=1, index=target.unsqueeze(1))
    probs_with_factor = ((1 - probs) ** alpha) * probs
    # Dice coefficient
    dsc = (2 * probs_with_factor + gamma) / (probs_with_factor + 1 + gamma)
    loss = 1 - dsc
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
