import functools

import numpy as np
from scipy.stats import truncnorm, norm

from .base import BaseDataset, BaseDataModule, NDArrayFrame


class GaussianDataset(BaseDataset):
    """

    """
    def __init__(
            self, n: int, d: int = 1, mu: float = 0.5, sigma: float = 0.16, range_cut: bool = True,
            sample_num: int = 1000, num_misreports: int = 0, dynamic: bool = True
    ):
        self.n = n
        self.d = d
        self.sample_num = sample_num
        self.num_misreports = num_misreports
        self.dynamic = dynamic

        if range_cut:
            self.distribution = truncnorm((-mu)/sigma, (1-mu)/sigma, loc=mu, scale=sigma)
        else:
            self.distribution = norm(loc=mu, scale=sigma)

        if dynamic:
            self.peaks = self.distribution.rvs((sample_num, d, n)).astype(np.float32)
            if num_misreports > 0:
                self.misreports = self.distribution.rvs((sample_num, num_misreports, d, n)).astype(np.float32)
            else:
                self.misreports = None

    def static_sample(self, item: int) -> NDArrayFrame:
        return NDArrayFrame(
            self.peaks[item],
            None if self.num_misreports == 0 is None else self.misreports[item],
        )

    def dynamic_sample(self) -> NDArrayFrame:
        return NDArrayFrame(
            self.distribution.rvs((self.d, self.n)).astype(np.float32),
            None if self.num_misreports == 0 else self.distribution.rvs((self.num_misreports, self.d, self.n)).astype(
                np.float32),
        )

    def __len__(self) -> int:
        return self.sample_num

    def __getitem__(self, item: int) -> NDArrayFrame:
        if self.dynamic:
            return self.dynamic_sample()
        else:
            return self.static_sample(item)


class GaussianDataModule(BaseDataModule):
    def __init__(
            self,
            n: int,
            d: int = 1,
            mu: float = 0.5,
            sigma: float = 0.16,
            range_cut: bool = True,
            train_num_misreports: int = 5,
            test_num_misreports: int = 50,
            batch_size: int = 1000,
            num_workers: int = 0,
            train_sample_num: int = 1000,
            val_sample_num: int = 2000,
            test_sample_num: int = 2000,
            train_dynamic: bool = True,
    ):
        super(GaussianDataModule, self).__init__(
            functools.partial(
                GaussianDataset, n=n, d=d, mu=mu, sigma=sigma, range_cut=range_cut, sample_num=train_sample_num,
                num_misreports=train_num_misreports, dynamic=train_dynamic,
            ),
            functools.partial(
                GaussianDataset, n=n, d=d, mu=mu, sigma=sigma, range_cut=range_cut, sample_num=val_sample_num,
                num_misreports=test_num_misreports
            ),
            functools.partial(
                GaussianDataset, n=n, d=d, mu=mu, sigma=sigma, range_cut=range_cut, sample_num=test_sample_num,
                num_misreports=test_num_misreports
            ),
            batch_size,
            num_workers
        )
        self.save_hyperparameters()
