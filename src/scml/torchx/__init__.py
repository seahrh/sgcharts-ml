import warnings
from configparser import SectionProxy
from typing import Dict, Iterable, List, NamedTuple, Optional, Set

import scml

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch, nn, F = None, None, None  # type: ignore
    warnings.warn("Install torch to use this feature", ImportWarning)

__all__ = [
    "parameter_size",
    "LrSchedulerConf",
    "schedulers",
    "schedulers_by_config",
    "whitening",
    "noisy_tune",
    "MultiSampleDropout",
]


def parameter_size(model: nn.Module) -> int:
    res = 0
    for p in model.parameters():
        if p.requires_grad:
            res += p.numel()
    return res


class LrSchedulerConf(NamedTuple):
    """
    lr_scheduler_config = {
    # REQUIRED: The scheduler instance
    "scheduler": lr_scheduler,
    # The unit of the scheduler's step size, could also be 'step'.
    # 'epoch' updates the scheduler on epoch end whereas 'step'
    # updates it after a optimizer update.
    "interval": "epoch",
    # How many epochs/steps should pass between calls to
    # `scheduler.step()`. 1 corresponds to updating the learning
    # rate after every epoch/step.
    "frequency": 1,
    # Metric to to monitor for schedulers like `ReduceLROnPlateau`
    "monitor": "val_loss",
    # If set to `True`, will enforce that the value specified 'monitor'
    # is available when the scheduler is updated, thus stopping
    # training if not found. If set to `False`, it will only produce a warning
    "strict": True,
    # If using the `LearningRateMonitor` callback to monitor the
    # learning rate progress, this keyword can be used to specify
    # a custom logged name
    "name": None,
    }

    See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html?highlight=configure_optimizers#configure-optimizers
    """

    scheduler: object
    interval: str = "epoch"
    frequency: int = 1
    monitor: str = "val_loss"
    strict: bool = True
    name: Optional[str] = None


def schedulers(optimizer, params: Iterable[Dict[str, str]]) -> List[LrSchedulerConf]:
    res: List[LrSchedulerConf] = []
    for ps in params:
        qn = ps["qualified_name"]
        if qn == "torch.optim.lr_scheduler.ReduceLROnPlateau":
            res.append(
                LrSchedulerConf(
                    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer=optimizer,
                        min_lr=float(ps["min_lr"]),
                        patience=int(ps["patience"]),
                        factor=float(ps["factor"]),
                        verbose=scml.getboolean(ps["verbose"]),
                    ),
                    name=qn,
                )
            )
            continue
        if qn == "torch.optim.lr_scheduler.CosineAnnealingLR":
            res.append(
                LrSchedulerConf(
                    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=optimizer,
                        T_max=int(ps["T_max"]),
                    ),
                    name=qn,
                )
            )
            continue
        if qn == "torch.optim.swa_utils.SWALR":
            res.append(
                LrSchedulerConf(
                    scheduler=torch.optim.swa_utils.SWALR(
                        optimizer=optimizer,
                        swa_lr=float(ps["swa_lr"]),
                        anneal_epochs=int(ps["anneal_epochs"]),
                        anneal_strategy=ps["anneal_strategy"],  # type: ignore[arg-type]
                    ),
                    name=qn,
                )
            )
            continue
        raise ValueError(f"Unsupported scheduler: {qn}")
    return res


def schedulers_by_config(
    optimizer, sections: Iterable[SectionProxy], booleans: Optional[Set[str]] = None
) -> List[LrSchedulerConf]:
    params: List[Dict] = []
    if booleans is None:
        booleans = {"verbose"}
    for section in sections:
        d = dict(section)
        for k in d.keys():
            if k in booleans:
                d[k] = "1" if section.getboolean(k) else "0"
        params.append(d)
    return schedulers(optimizer=optimizer, params=params)


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
    # The parameters() only gives the module parameters i.e. weights and biases.
    # On the other hand, state_dict returns a dictionary containing a whole state of the module.
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


from .loss import *

__all__ += loss.__all__  # type: ignore  # module name is not defined

from .pooling import *

__all__ += pooling.__all__  # type: ignore  # module name is not defined
