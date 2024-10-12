import functools
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from ..base import BaseModel, BaseSystem, social_cost_each_l1, TensorFrame


class FlexibleNet(BaseModel):
    def __init__(self, n: int, k: int, d: int):
        super().__init__()


    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        # consider multi-dimensions
        # consider flexibility
        indices = torch.argsort(peaks, dim=-1)

        # TODO: think of a novel architecture
