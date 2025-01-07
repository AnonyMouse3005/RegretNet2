import functools
import itertools
import sys
if sys.version_info >= (3, 8):
    from math import comb
else:
    import scipy.special
    comb = functools.partial(scipy.special.comb, exact=True)


import torch

from ..base import BaseModel, BaseSystem, social_cost_each_l1


class DictatorRule(BaseModel):
    """ This Rule chooses the peaks of k agents as the locations by brute-force training.

    Need C_n^k memory, unfeasible when n is too large. The same happens when trading off time for memory.

    So when combinations > max_combinations, use random sampling to sample combinations.

    """
    def __init__(self, n: int, k: int, agent_weights: torch.Tensor, objective: str, max_training_steps: int = 1, max_combinations: int = 10000):
        super(DictatorRule, self).__init__()
        assert n >= k, 'the number of agents `n` must be greater than or equal to the number of facilities `k`'

        if comb(n, k) > max_combinations:
            combinations = []
            for _ in range(max_combinations):
                combinations.append(torch.randperm(n)[:k])
            combinations = torch.stack(combinations, dim=0)
        else:
            combinations = torch.tensor(list(itertools.combinations(range(n), k)))

        self.agent_weights = agent_weights
        self.objective = objective

        self.register_buffer('combinations', combinations)
        self.register_buffer('costs', torch.ones(combinations.size(0)))
        self.register_buffer('current_training_steps', torch.tensor(0))
        self.register_buffer('max_training_steps', torch.tensor(max_training_steps))
        self.register_buffer('best_combination', self.combinations[0, :])

    def train_once(self, peaks: torch.Tensor):
        for i, c in enumerate(self.combinations):
            # # unweighted social cost
            # costs = torch.mean(social_cost_each_l1(peaks, peaks[..., c]))
            if self.objective == 'max':  # NOTE: max social cost only makes sense for unweighted case
                costs = torch.mean(torch.max(social_cost_each_l1(peaks, peaks[..., c]), dim=1)[0])
            else:
                costs = torch.mean(social_cost_each_l1(peaks, peaks[..., c]) @ self.agent_weights/self.agent_weights.sum())
            self.costs[i] = self.current_training_steps / (self.current_training_steps + 1) * self.costs[i] + \
                            1 / (self.current_training_steps + 1) * costs
        self.current_training_steps += 1
        self.best_combination = self.combinations[torch.argmin(self.costs), :]

    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        if self.training and self.current_training_steps < self.max_training_steps:
            self.train_once(peaks)
        return peaks[..., self.best_combination]


class DictatorRuleSystem(BaseSystem):
    def __init__(
            self,
            n: int,
            k: int,
            agent_weights: torch.Tensor,
            objective: str,
            max_training_steps: int = 1,
            max_combinations: int = 10000,
    ):
        super(DictatorRuleSystem, self).__init__(
            DictatorRule(n, k, agent_weights, objective, max_training_steps, max_combinations),
            agent_weights,
            objective,
        )
        self.save_hyperparameters()
