from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "WeightedLayerPooling",
    "AttentionPooling",
    "max_pooling",
    "mean_pooling",
]


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
