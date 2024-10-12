import functools
from typing import Union, List, Tuple, Optional

import torch
from torch import nn

from ..base import BaseModel, BaseSystem, social_cost_each_l1, TensorFrame


class RegretNet(BaseModel):
    def __init__(
            self,
            n: int,
            k: int,
            hidden_layer_channels: Union[List[int], Tuple[int, ...]] = (40, 40, 40, 40),
    ):
        """An implementation of RegretNet.

        :param n: number of agents
        :param k: number of facilities
        :param hidden_layer_channels: the size of each hidden layer
        """
        super(RegretNet, self).__init__()

        # create linear and relu layers
        layer_channels = [n, *hidden_layer_channels, k]
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(layer_channels[i], layer_channels[i+1], bias=True),
                nn.ReLU() if i < len(layer_channels)-2 else nn.Identity(),
            )
            for i in range(len(layer_channels)-1)
        ])

        # parameter initialization
        for linear, _ in self.layers:
            nn.init.normal_(linear.weight, 0, 0.1)
            nn.init.constant_(linear.bias, 0.1)

    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        z = self.layers(peaks)
        return z


class RegretNetSystem(BaseSystem):
    def __init__(
            self,
            n: int,
            k: int,
            hidden_layer_channels: Union[List[int], Tuple[int, ...]] = (40, 40, 40, 40),
            lr: float = 0.005,
            gamma: float = 0.99,
    ):
        super(RegretNetSystem, self).__init__(
            RegretNet(n, k, hidden_layer_channels),
            functools.partial(torch.optim.Adam, lr=lr),
            functools.partial(torch.optim.lr_scheduler.StepLR, step_size=100, gamma=gamma)
        )
        self.n = n
        self.register_buffer('lambda_t', torch.tensor(0))
        self.register_buffer('rho', torch.tensor(5))
        self.register_buffer('max_regret_all_times', torch.tensor(0))
        self.save_hyperparameters()

    def get_agents_costs_and_max_regrets(self, batch: TensorFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # calculate costs for true peaks
        facilities = self.model(batch.peaks)  # (batch_size, d, k)
        costs = social_cost_each_l1(batch.peaks, facilities)  # (batch_size, n)

        # calculate costs for misreported peaks
        if batch.misreport_peaks is None:
            raise ValueError(f'misreports must be provided when training {self.__class__.__name__}! '
                             f'please set train_num_misreports and test_num_misreports in dataset')
        misreport_facilities = self.model(batch.misreport_peaks)  # (batch_size, n, num_misreports, d, k)
        misreport_costs = social_cost_each_l1(batch.peaks.unsqueeze(1).unsqueeze(1), misreport_facilities)  # (batch_size, n, num_misreports, n)
        misreport_costs = torch.diagonal(misreport_costs, dim1=1, dim2=3)  # (batch_size, num_misreports, n)

        # calculate max regret
        max_regrets = torch.max(torch.relu(costs.unsqueeze(1) - misreport_costs), dim=1)[0]  # (batch_size, n)
        return facilities, costs, max_regrets

    def train_once(self, batch: TensorFrame) -> Optional[torch.Tensor]:
        facilities, costs, max_regrets = self.get_agents_costs_and_max_regrets(batch)

        max_regret = torch.mean(torch.max(max_regrets, dim=1)[0])
        cost = torch.mean(costs)

        self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'], on_step=True, prog_bar=True)

        self.max_regret_all_times = torch.maximum(self.max_regret_all_times, max_regret).detach()
        if self.trainer.global_step % 50 == 49:
            with torch.no_grad():
                self.lambda_t = self.lambda_t + self.rho * self.max_regret_all_times
            self.log('lambda_t', self.lambda_t, prog_bar=True)
            self.log('rho', self.rho, prog_bar=True)

        # self.print(cost, self.lambda_t * max_regret, self.rho * (max_regret ** 2))
        return cost + self.lambda_t * max_regret + self.rho * (max_regret ** 2)
        # return cost

    def infer_once(self, batch: TensorFrame) -> torch.Tensor:
        facilities, costs, max_regrets = self.get_agents_costs_and_max_regrets(batch)

        max_regret = torch.mean(torch.max(max_regrets, dim=1)[0])

        min_social_cost, max_social_cost = torch.aminmax(costs, dim=-1)
        std_social_cost, mean_social_cost = torch.std_mean(costs, dim=-1, unbiased=False)
        self.log('mean_social_cost', torch.mean(mean_social_cost), prog_bar=True)
        self.log('std_social_cost', torch.mean(std_social_cost))
        self.log('min_social_cost', torch.mean(min_social_cost))
        self.log('max_social_cost', torch.mean(max_social_cost))
        self.log('max_regret', max_regret, prog_bar=True)
        self.log('metric', torch.mean(mean_social_cost) if max_regret.item() < 0.003 else float('inf'))
        return facilities

    def on_train_epoch_start(self) -> None:
        torch.zero_(self.max_regret_all_times)
