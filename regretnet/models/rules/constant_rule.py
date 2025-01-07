import functools
import sys
if sys.version_info >= (3, 8):
    from math import comb
else:
    import scipy.special
    comb = functools.partial(scipy.special.comb, exact=True)

import torch

from ..base import BaseSystem, BaseModel, social_cost_each_l1


class ConstantRule(BaseModel):
    """ This Rule chooses a set of constant values by brute-force training.

    """
    def __init__(
            self, k: int, agent_weights: torch.Tensor, objective: str, d: int = 1,
            divisions: int = 10, max_training_steps: int = 1, max_combinations: int = 10000
    ):
        super().__init__()
        f = torch.arange(0, 1 + 1e-5, 1/(divisions-1))
        if comb(divisions ** d, k) > max_combinations:
            facilities = []
            for _ in range(max_combinations):
                facilities.append(torch.reshape(f[torch.randint(high=divisions, size=(d*k, ))], shape=(d, k)))
            facilities = torch.stack(facilities, dim=0)
        else:
            f = torch.combinations(f, r=d, with_replacement=True)
            combinations = torch.combinations(torch.arange(f.size(0)), r=k, with_replacement=False)
            facilities = []
            for c in combinations:
                facilities.append(f[c, :])
            facilities = torch.stack(facilities, dim=0).permute(0, 2, 1)

        self.agent_weights = agent_weights
        self.objective = objective

        self.register_buffer('facilities', facilities)
        self.register_buffer('costs', torch.ones(facilities.size(0)))
        self.register_buffer('current_training_steps', torch.tensor(0))
        self.register_buffer('max_training_steps', torch.tensor(max_training_steps))
        self.register_buffer('best_facilities', self.facilities[0, :, :])

    def train_once(self, peaks: torch.Tensor):
        for i, f in enumerate(self.facilities):
            # # unweighted social cost
            # costs = torch.mean(social_cost_each_l1(peaks, f))
            if self.objective == 'max':  # NOTE: max social cost only makes sense for unweighted case
                costs = torch.mean(torch.max(social_cost_each_l1(peaks, f), dim=1)[0])
            else:
                costs = torch.mean(social_cost_each_l1(peaks, f) @ self.agent_weights/self.agent_weights.sum())
            self.costs[i] = self.current_training_steps / (self.current_training_steps + 1) * self.costs[i] + \
                            1 / (self.current_training_steps + 1) * costs
        self.current_training_steps += 1
        self.best_facilities = self.facilities[torch.argmin(self.costs), :, :]

    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        if self.training and self.current_training_steps < self.max_training_steps:
            self.train_once(peaks)
        return self.best_facilities


class ConstantRuleSystem(BaseSystem):
    def __init__(
            self,
            k: int,
            agent_weights: torch.Tensor,
            objective: str,
            d: int = 1,
            divisions: int = 10,
            max_training_steps: int = 1,
            max_combinations: int = 10000,
    ):
        super(ConstantRuleSystem, self).__init__(
            ConstantRule(k, agent_weights, objective, d, divisions, max_training_steps, max_combinations),
            agent_weights,
            objective
        )
        self.save_hyperparameters()
