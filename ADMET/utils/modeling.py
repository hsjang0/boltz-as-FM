import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList()

        last = input_dim
        for _ in range(layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "lin": nn.Linear(last, hidden),
                        "bn": nn.BatchNorm1d(hidden),
                        "act": nn.ReLU(),
                        "drop": nn.Dropout(dropout),
                    }
                )
            )
            last = hidden + input_dim
        self.final = nn.Linear(last, 1)

    def _apply_bn(self, bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) == 1:
            return F.batch_norm(x, bn.running_mean, bn.running_var, bn.weight, bn.bias, False, 0.0, bn.eps)
        return bn(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_bn(self.input_norm, x)
        raw = self.input_drop(x)
        h = raw
        for block in self.blocks:
            h = block["drop"](block["act"](self._apply_bn(block["bn"], block["lin"](h))))
            h = torch.cat([h, raw], dim=1)
        return self.final(h).squeeze(-1)


def make_scheduler(optimizer, total_epochs: int, warmup: int = 5) -> LambdaLR:
    def lr_fn(epoch: int) -> float:
        if epoch < warmup:
            return epoch / max(1, warmup)
        return (1 + math.cos(math.pi * (epoch - warmup) / max(1, (total_epochs - warmup)))) / 2

    return LambdaLR(optimizer, lr_lambda=lr_fn)
