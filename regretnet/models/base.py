from typing import Callable, Iterable, Optional, Any, Union, List

import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn

from regretnet.data_modules.base import TensorFrame


def social_cost_each_l1(peaks: torch.Tensor, facilities: torch.Tensor):
    """ Calculate the social cost for each agent with l1 distance.

    :param peaks: locations of agents in shape (batch_size, d, n)
    :param facilities: locations of facilities in shape (batch_size, d, k)
    :return: the social cost for each agent in shape (batch_size, n)
    """
    costs, _ = torch.min(torch.sum(torch.abs(peaks.unsqueeze(-1) - facilities.unsqueeze(-2)), dim=-3), dim=-1)
    # broadcasting: (batch_size, d, n, 1) - (batch_size, d, 1, k) -> (batch_size, d, n, k) i.e., social cost of agent i wrt facility k
    # sum(..., dim=-3) -> (batch_size, n, k) i.e., unchanged for RegretNet
    return costs


class BaseModel(nn.Module):
    """ Base Model that calculates locations of facilities given agents' peaks.

    """
    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        """

        :param peaks: the (possibly misreported) peaks of agents with shape (*, d, n)
        :return: the location of facilities with shape (*, d, k)
        """
        raise NotImplementedError()


class BaseSystem(pl.LightningModule):
    def __init__(
            self,
            model: BaseModel,
            optimizer_builder: Optional[Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]] = None,
            scheduler_builder: Optional[Callable[[torch.optim.Optimizer], Any]] = None,
    ):
        """Base System that wraps models to interact with train, val and test actions.

        :param model: the mechanism model (a neural network or a rule)
        :param optimizer_builder: the optimizer builder accepts `params` to build an optimizer
         (usually constructed by `functools.partial`). Ignored when model is a Rule.
        :param scheduler_builder: the scheduler builder accepts `optimizer` to build a scheduler
         (usually constructed by `functools.partial`). Ignored when model is a Rule.
        """
        super(BaseSystem, self).__init__()
        self.model = model
        self.optimizer_builder = optimizer_builder if len(list(self.model.parameters())) > 0 else None
        self.scheduler_builder = scheduler_builder if self.optimizer_builder is not None else None

    def train_once(self, batch: TensorFrame) -> Optional[torch.Tensor]:
        """Train the model with a batch of samples for one time.

        :param batch: a batch of samples
        :return: a loss value if the model needs to be trained by back propagation, None otherwise.
        """
        facilities = self.model(batch.peaks)
        if self.optimizer_builder is not None:
            self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'], on_step=True, prog_bar=True)
            loss = torch.mean(social_cost_each_l1(batch.peaks, facilities))
            return loss
        else:
            return None

    def infer_once(self, batch: TensorFrame) -> torch.Tensor:
        """Test the model with a batch of samples for one time.

        :param batch: a batch of samples
        :return: the result facility locations
        """
        with torch.no_grad():
            facilities = self.model(batch.peaks)
            social_cost = social_cost_each_l1(batch.peaks, facilities)
            min_social_cost, max_social_cost = torch.aminmax(social_cost, dim=-1)
            std_social_cost, mean_social_cost = torch.std_mean(social_cost, dim=-1, unbiased=False)
            self.log('mean_social_cost', torch.mean(mean_social_cost), prog_bar=True)
            self.log('std_social_cost', torch.mean(std_social_cost))
            self.log('min_social_cost', torch.mean(min_social_cost))
            self.log('max_social_cost', torch.mean(max_social_cost))
            self.log('metric', torch.mean(mean_social_cost))
        return facilities

    def predict_once(self, peaks: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Used in inference. Predict the facilities' locations given the peaks of agents.

        :param peaks: The peaks of agents. Can be a np.ndarray with shape (sample_num, d, n) or list of agents locations
        [l1, l2, l3 ... ln]. The list will be interpreted as (1, 1, n) np.ndarray.
        :return: The locations of facilities with shape (sample_num, d, k)
        """
        if isinstance(peaks, list):
            peaks = [[peaks]]
        peaks = torch.tensor(peaks, device=self.device)
        with torch.no_grad():
            facilities = self.model(peaks)
        return facilities.cpu().numpy()

    def training_step(self, batch: TensorFrame, batch_idx):
        return self.train_once(batch)

    def validation_step(self, batch: TensorFrame, batch_idx):
        _ = self.infer_once(batch)
        return None

    def test_step(self, batch: TensorFrame, batch_idx):
        return self.infer_once(batch)

    def configure_optimizers(self):
        if self.optimizer_builder is not None:
            optimizer = self.optimizer_builder(self.parameters())
            if self.scheduler_builder is not None:
                scheduler = self.scheduler_builder(optimizer)
                return [optimizer], [scheduler]
            return optimizer
        return None
