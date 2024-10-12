import torch

from ..base import BaseSystem, social_cost_each_l1
from .dictator_rule import DictatorRule


class PercentileRule(DictatorRule):
    """ This Rule choose the best percentiles of the peaks as the locations by brute-force training.

    """
    def __init__(self, n: int, k: int, max_training_steps: int = 1, max_combinations: int = 10000):
        super(PercentileRule, self).__init__(n, k, max_training_steps, max_combinations)

    def train_once(self, peaks: torch.Tensor):
        sorted_peaks, _ = torch.sort(peaks, dim=-1)
        for i, c in enumerate(self.combinations):
            costs = torch.mean(social_cost_each_l1(peaks, sorted_peaks[..., c]))
            self.costs[i] = self.current_training_steps / (self.current_training_steps + 1) * self.costs[i] + \
                            1 / (self.current_training_steps + 1) * costs
        self.current_training_steps += 1
        self.best_combination = self.combinations[torch.argmin(self.costs), :]

    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        sorted_peaks, _ = torch.sort(peaks, dim=-1)
        if self.training and self.current_training_steps < self.max_training_steps:
            self.train_once(peaks)
        return sorted_peaks[..., self.best_combination]


class PercentileRuleSystem(BaseSystem):
    def __init__(
            self,
            n: int,
            k: int,
            max_training_steps: int = 1,
            max_combinations: int = 10000,
    ):
        super(PercentileRuleSystem, self).__init__(
            PercentileRule(n, k, max_training_steps, max_combinations)
        )
        self.save_hyperparameters()
