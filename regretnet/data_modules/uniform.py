import functools

import numpy as np

from .base import BaseDataset, BaseDataModule, NDArrayFrame


class UniformDataset(BaseDataset):
    """ Uniformly sample agent peaks in [0,1] for any number of dimensions.

    """
    def __init__(self, n: int, d: int = 1, sample_num: int = 1000, num_misreports: int = 0, dynamic: bool = True):
        self.n = n
        self.d = d
        self.sample_num = sample_num
        self.num_misreports = num_misreports
        self.dynamic = dynamic

        if self.dynamic is False:
            self.peaks = np.random.rand(sample_num, d, n).astype(np.float32)
            if num_misreports > 0:
                self.misreports = np.random.rand(sample_num, num_misreports, d, n).astype(np.float32)
            else:
                self.misreports = None

    def static_sample(self, item: int) -> NDArrayFrame:
        return NDArrayFrame(
            self.peaks[item],
            None if self.num_misreports == 0 else self.misreports[item],
        )

    def dynamic_sample(self) -> NDArrayFrame:
        return NDArrayFrame(
            np.random.rand(self.d, self.n).astype(np.float32),
            None if self.num_misreports == 0 else np.random.rand(self.num_misreports, self.d, self.n).astype(
                np.float32),
        )

    def __len__(self) -> int:
        return self.sample_num

    def __getitem__(self, item: int) -> NDArrayFrame:
        if self.dynamic:
            return self.dynamic_sample()
        else:
            return self.static_sample(item)


class UniformDataModule(BaseDataModule):
    def __init__(
            self,
            n: int,
            d: int = 1,
            train_num_misreports: int = 20,  # NOTE: 20 is equivalent to M' = 5 as workaround, albeit slower
            test_num_misreports: int = 50,
            batch_size: int = 500,  # NOTE: for training in particular; use all samples for validating and testing
            num_workers: int = 0,
            train_sample_num: int = 25000,
            val_sample_num: int = 2000,
            test_sample_num: int = 2000,
            train_dynamic: bool = True
    ):
        super(UniformDataModule, self).__init__(
            functools.partial(
                UniformDataset, n=n, d=d, sample_num=train_sample_num,
                num_misreports=train_num_misreports, dynamic=train_dynamic,
            ),
            functools.partial(
                UniformDataset, n=n, d=d, sample_num=val_sample_num, num_misreports=test_num_misreports
            ),
            functools.partial(
                UniformDataset, n=n, d=d, sample_num=test_sample_num, num_misreports=test_num_misreports
            ),
            batch_size,
            num_workers,
            on_the_fly=True
        )
        self.save_hyperparameters()
