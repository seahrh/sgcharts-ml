import warnings
from typing import Iterable, Optional, Sequence, Tuple

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch, nn, F = None, None, None  # type: ignore
    warnings.warn("Install torch to use this feature", ImportWarning)

__all__ = [
    "uncertainty_weighted_loss",
    "whitening",
    "noisy_tune",
    "MultiSampleDropout",
    "WeightedLayerPooling",
    "AttentionPooling",
    "max_pooling",
    "mean_pooling",
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


def whitening(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Whitening is a linear transformation that transforms a vector of random variables with a known
    covariance matrix into a new vector whose covariance is an identity matrix, and has been verified effective
    to improve text and image representations.

    Based on Huang, Junjie, et al. "Whiteningbert: An easy unsupervised sentence embedding approach."
    arXiv preprint arXiv:2104.01767 (2021). https://github.com/Jun-jie-Huang/WhiteningBERT

    :param embeddings: 2D tensor
    :return: whitened embeddings
    """
    mu = torch.mean(embeddings, dim=0, keepdim=True)
    cov = torch.mm((embeddings - mu).t(), embeddings - mu)
    u, s, vt = torch.svd(cov)
    W = torch.mm(u, torch.diag(1 / torch.sqrt(s)))
    res = torch.mm(embeddings - mu, W)
    return res


def noisy_tune(
    model: nn.Module, noise_intensity: float, prefixes: Optional[Iterable[str]] = None
) -> None:
    """NoisyTune: A Little Noise Can Help You Finetune Pretrained Language Models Better (ACL 2022)
    https://aclanthology.org/2022.acl-short.76.pdf
    """
    if noise_intensity < 0:
        raise ValueError("noise_intensity must be non-negative number")
    sd = model.state_dict()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        apply_noise: bool = False
        if prefixes is None:
            apply_noise = True
        else:
            for prefix in prefixes:
                if name.startswith(prefix):
                    apply_noise = True
        if apply_noise:
            sd[name] += (
                (torch.rand(param.size()) - 0.5) * noise_intensity * torch.std(param)
            )
    model.load_state_dict(sd)


class MultiSampleDropout(nn.Module):
    def __init__(
        self,
        classifier: nn.Module,
        size: int,
        start_prob: float,
        increment: float = 0,
        dropout_cls=nn.Dropout,
    ):
        if size < 1:
            raise ValueError("number of dropouts (size) must be a positive integer")
        super().__init__()
        self.dropouts = nn.ModuleList(
            [dropout_cls(start_prob + (increment * i)) for i in range(size)]
        )
        self.classifier = classifier

    def forward(self, x):
        """
        Call the module
        Args:
            x (`torch.tensor`): The input tensor to apply dropout
        """
        m = len(self.dropouts)
        res = self.classifier(self.dropouts[0](x)) / float(m)
        for i in range(1, m):
            res += self.classifier(self.dropouts[i](x)) / float(m)
        return res


class WeightedLayerPooling(nn.Module):
    """
    Token embeddings are weighted mean of their different hidden layer representations
    Based on
    - https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/WeightedLayerPooling.py
    - https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently/notebook
    """

    def __init__(
        self,
        num_hidden_layers: int,
        layer_start: int = 1,
        layer_weights: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        if layer_weights is not None:
            self.layer_weights: nn.Parameter = layer_weights
        else:
            self.layer_weights = nn.Parameter(
                torch.tensor(
                    [1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float
                )
            )

    def forward(self, hidden_states: Tuple[torch.Tensor]) -> torch.Tensor:
        # stack gives (L, B, S, H)
        all_layer_embedding = torch.stack(hidden_states)
        all_layer_embedding = all_layer_embedding[self.layer_start :, :, :, :]
        # unsqueeze gives (L, 1, 1, 1). Expand gives (L, B, S, H)
        weight_factor = (
            self.layer_weights.unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(all_layer_embedding.size())
        )
        # Weighted sum reduces (L, B, S, H) to (B, S, H)
        res: torch.Tensor = (weight_factor * all_layer_embedding).sum(
            dim=0
        ) / self.layer_weights.sum()
        return res


class AttentionPooling(nn.Module):
    """
    Attention operation can learn the contribution of each  h_i_CLS.
    We can use a dot-product attention module to dynamically combine all intermediates.
    Input Shape: (#Layers, Batch Size, Sequence Length, Hidden Size)
    Output Shape: (Batch Size, Final Embedding Size)
    Based on https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
    """

    def __init__(self, hidden_size: int, out_size: int):
        super().__init__()
        # q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, hidden_size))
        self.q = nn.Parameter(
            nn.init.normal_(torch.empty(size=(1, hidden_size)))
        ).float()
        # w_ht = np.random.normal(loc=0.0, scale=0.1, size=(hidden_size, out_size))
        self.w_h = nn.Parameter(
            nn.init.normal_(torch.empty(size=(hidden_size, out_size)))
        ).float()

    def _attention(self, h: torch.Tensor) -> torch.Tensor:
        # Learn layer weights named "Queries" q (1, H)
        # Transpose last 2 dimensions of h from (B, L, H) to (B, H, L)
        # Matrix multiply: (1, H).(B, H, L) = (B, 1, L)
        # Squeeze from (B, 1, L) to (B, L)
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        # Softmax on last dimension, (B, L)
        v = F.softmax(v, -1)
        # Matrix multiply: (B, 1, L).(B, L, H) = (B, 1, H)
        # Transpose last 2 dimensions from (B, 1, H) to (B, H, 1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        # Final projection to get output size
        # TODO check if removing final projection will hurt performance
        # Matrix multiply: (F, H).(B, H, 1) = (B, F, 1)
        # Squeeze from (B, F, 1) to (B, F)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v

    def forward(self, hidden_states: Tuple[torch.Tensor]) -> torch.Tensor:
        # all_hidden_states shape (L, B, S, H)
        n_layers = len(hidden_states)
        # take only the CLS hidden state so shape becomes (L, B, H)
        # start from layer index 1 (ignore input embedding layer at index 0)
        h = torch.stack(
            [hidden_states[i][:, 0, :].squeeze(1) for i in range(1, n_layers)],
            dim=0,
        )
        # reshape from (L, B, H) to (B, L, H)
        size = h.size()
        h = h.view(-1, size[0], size[2])
        return self._attention(h)


def max_pooling(inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Max Pooling
    Based on https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently/notebook
    """
    # attention_mask (torch.FloatTensor of shape (batch_size, sequence_length)
    mask = attention_mask.unsqueeze(-1).expand(inputs.size()).float()
    res: torch.Tensor = torch.max(inputs * mask, dim=1).values
    return res


def mean_pooling(inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling
    Based on https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently/notebook
    """
    # Expand Attention Mask from [batch_size, seq_len] to [batch_size, seq_len, hidden_size].
    mask = attention_mask.unsqueeze(-1).expand(inputs.size()).float()
    # Sum Embeddings along seq length axis so now we have [batch_size, hidden_size].
    sum_embeddings = torch.sum(inputs * mask, 1)
    # Sum Mask along seq length axis, so we can ignore padding tokens.
    sum_mask = mask.sum(1)
    # prevent divide-by-zero
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    # Result has shape (batch size, hidden size)
    return sum_embeddings / sum_mask
