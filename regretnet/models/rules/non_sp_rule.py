import itertools

import torch

from ..base import BaseModel, BaseSystem, social_cost_each_l1


class NonSPRule(BaseModel):
    """ This Rule selects k agents' peaks as the best locations by brute-force searching.

    Only support one dimensional case now.

    """
    def __init__(self, n: int, k: int, agent_weights: torch.Tensor, objective: str):
        super(NonSPRule, self).__init__()
        self.combinations = list(itertools.combinations(range(n), k))
        self.agent_weights = agent_weights
        self.objective = objective

    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        best_cost = None
        best_facilities = None
        for c in self.combinations:
            # # unweighted social cost
            # cost = torch.mean(social_cost_each_l1(peaks, peaks[..., c]), dim=-1)  # (batch_size,)
            if self.objective == 'max':  # NOTE: max social cost only makes sense for unweighted case
                cost = torch.max(social_cost_each_l1(peaks, peaks[..., c]), dim=-1)[0]  # (batch_size,)  # NOTE: just for completion, does NOT really work for max cost
            else:
                cost = social_cost_each_l1(peaks, peaks[..., c]) @ self.agent_weights/self.agent_weights.sum()  # (batch_size,)
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
            agent_weights: torch.Tensor,
            objective: str,
    ):
        super(NonSPRuleSystem, self).__init__(
            NonSPRule(n, k, agent_weights, objective),
            agent_weights,
            objective,
        )
        self.save_hyperparameters()
