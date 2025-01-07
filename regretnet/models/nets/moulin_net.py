import functools
from typing import Optional

import torch
from torch import nn

from ..base import BaseModel, BaseSystem, social_cost_each_l1, TensorFrame


class MoulinNet(BaseModel):
    def __init__(
            self,
            n: int,
            k: int,
            L: int,
            J: int,
    ):
        """An implementation of MoulinNet.

        :param n: number of agents
        :param k: number of facilities
        :param L: controls the width of last layer (min layer)
        :param J: controls the width of first layer (max layer)
        """
        super(MoulinNet, self).__init__()
        # use a normal linear layer here, with a penalty term in loss function to ensure (almost) monotonicity.
        # same as the paper's code.
        self.linear = nn.Sequential(
            nn.Linear(n, k*L*J, bias=True),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.linear[0].weight, 0, 0.1)

        self.n = n
        self.k = k
        self.L = L
        self.J = J

        ones = torch.ones(n, n, dtype=torch.float)
        self.register_buffer('nu_perm', 2*torch.tril(ones) - ones, persistent=False)

    def forward_h(self, nu: torch.Tensor) -> torch.Tensor:
        """ Compute a_s given nu (the binary encoding of set S)

        :param nu: (*, n, n)
        :return: (*, n, k)
        """
        nu_shape = nu.size()
        res = self.linear(nu).view(*nu_shape[:-2], self.n, self.k, self.L, self.J)
        res = torch.max(res, dim=-1)[0]
        res = torch.min(res, dim=-1)[0]
        return res

    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tao, s_pi = torch.sort(peaks, dim=-1)
            s_pi = torch.argsort(s_pi, dim=-1)
            tao = tao.unsqueeze(-1)
            s_pi_shape = s_pi.size()
            nu = self.nu_perm[:, s_pi.flatten()].reshape(self.n, *s_pi_shape[:-1], self.n)
            nu = nu.permute(*range(1, len(s_pi_shape)), 0, len(s_pi_shape))

        a_s = self.forward_h(nu)
        z = torch.min(torch.maximum(a_s, tao), dim=-2)[0]
        return z


class MoulinNetSystem(BaseSystem):
    def __init__(
            self,
            n: int,
            k: int,
            agent_weights: torch.Tensor,
            objective: str,
            L: int = 3,
            J: int = 3,
            lr: float = 0.1,
    ):
        super(MoulinNetSystem, self).__init__(
            MoulinNet(n, k, L, J),
            agent_weights,
            objective,
            functools.partial(torch.optim.Adam, lr=lr)
        )
        self.agent_weights = agent_weights
        self.objective = objective
        self.save_hyperparameters()

    def train_once(self, batch: TensorFrame) -> Optional[torch.Tensor]:
        facilities = self.model(batch.peaks)
        self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'], on_step=True, prog_bar=True)
        # # unweighted social cost
        # loss = torch.mean(social_cost_each_l1(batch.peaks, facilities))
        if self.objective == 'max':  # NOTE: max social cost only makes sense for unweighted case
            loss = torch.mean(torch.max(social_cost_each_l1(batch.peaks, facilities), dim=1)[0])
        else:
            loss = torch.mean(social_cost_each_l1(batch.peaks, facilities) @ self.agent_weights/self.agent_weights.sum())
        regular = torch.mean(torch.relu(self.model.linear[0].weight))
        return loss + regular
