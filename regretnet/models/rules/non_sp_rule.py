import itertools

import torch

from ..base import BaseModel, BaseSystem, social_cost_each_l1


class NonSPRule(BaseModel):
    """ This Rule selects k agents' peaks as the best locations by brute-force searching.

    Only support one dimensional case now.

    """
    def __init__(self, n: int, k: int):
        super(NonSPRule, self).__init__()
        self.combinations = list(itertools.combinations(range(n), k))

    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        best_cost = None
        best_facilities = None
        for c in self.combinations:
            cost = torch.mean(social_cost_each_l1(peaks, peaks[..., c]), dim=-1)
            if best_cost is None:
                best_cost = cost                    # (*)
                best_facilities = peaks[..., c]     # (*, d, k)
            else:
                mask = torch.lt(cost, best_cost)
                best_facilities[mask] = peaks[..., c][mask]
                best_cost[mask] = cost[mask]
        return best_facilities


class NonSPRuleSystem(BaseSystem):
    def __init__(
            self,
            n: int,
            k: int,
    ):
        super(NonSPRuleSystem, self).__init__(
            NonSPRule(n, k)
        )
        self.save_hyperparameters()
