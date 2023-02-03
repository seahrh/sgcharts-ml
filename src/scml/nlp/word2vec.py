from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

import scml

__all__ = ["SkipGramWord2Vec"]
log = scml.get_logger(__name__)


class SkipGramWord2Vec(nn.Module):
    """
    PyTorch implementation of the word2vec (skip-gram model)

    Based on https://github.com/n0obcoder/Skip-Gram-Model-PyTorch
    """

    def __init__(
        self,
        lr: float,
        vocab_size: int,
        embedding_size: int,
        noise_dist: Optional[Sequence[float]] = None,
        negative_samples: int = 10,
    ):
        super().__init__()
        self.lr = lr
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        self.noise_dist = (
            torch.ones(self.vocab_size) if noise_dist is None else noise_dist
        )
        # Initialize both embedding tables with uniform distribution
        self.word_embeddings.weight.data.uniform_(-1, 1)

    def forward(self, center_words, outside_words):
        log.debug(
            f"center_word.size={center_words.size()}, outside_word.size={outside_words.size()}"
        )  # bs
        # bs, emb_dim
        em_center = self.word_embeddings(center_words)
        # bs, emb_dim
        em_outside = self.word_embeddings(outside_words)
        log.debug(
            f"em_center.size={em_center.size()}, em_outside.size={em_outside.size()}"
        )
        # dot product: element-wise multiply, followed by sum
        em_dot = torch.mul(em_center, em_outside)  # bs, emb_dim
        em_dot = torch.sum(em_dot, dim=1)  # bs
        log.debug(f"em_dot.size={em_dot.size()}")
        true_pair_loss = F.logsigmoid(em_dot).neg()  # bs
        log.debug(f"true_pair_loss.size={true_pair_loss.size()}")
        loss = true_pair_loss
        if self.negative_samples > 0:
            num_samples = outside_words.size()[0] * self.negative_samples
            neg_input_ids = torch.multinomial(
                self.noise_dist,
                num_samples=num_samples,
                replacement=num_samples > self.vocab_size,
            )
            # bs, num_neg_samples
            # need to set device explicitly here, else error:
            # Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu
            neg_input_ids = neg_input_ids.view(
                outside_words.size()[0], self.negative_samples
            ).to(self.device)
            log.debug(f"neg_input_ids.size={neg_input_ids}")
            # bs, neg_samples, emb_dim
            em_neg = self.word_embeddings(neg_input_ids)
            log.debug(f"em_neg.size={em_neg.size()}")
            # batch matrix multiply
            # (B, K, D) * (B, D, 1) = (B, K, 1)
            # Negated dot product of noise pair
            # Large +dot, large -dot, sigmoid 0, logsigmoid -Inf
            # Large -dot, large +dot, sigmoid 1, logsigmoid zero
            em_dot_neg = torch.bmm(em_neg, em_center.unsqueeze(2)).neg()
            em_dot_neg = em_dot_neg.squeeze(2)
            log.debug(f"em_dot_neg.size={em_dot_neg.size()}")
            noise_pair_loss = F.logsigmoid(em_dot_neg).sum(1).neg()  # bs
            log.debug(f"noise_pair_loss.size={noise_pair_loss.size()}")
            loss += noise_pair_loss
        return loss.mean()
