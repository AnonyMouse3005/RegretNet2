import functools

import numpy as np

from .base import BaseDataset, BaseDataModule, NDArrayFrame


class CustomDataset(BaseDataset):
    """ Uniformly sample agent peaks in [0,1] for any number of dimensions.

    """
    def __init__(self, data: dict):
        self.d = 1
        peaks = data['peaks'].astype(np.float32)  # (sample_num, n)
        self.sample_num, self.n = peaks.shape
        self.peaks = np.expand_dims(peaks, axis=1)  # (sample_num, d=1, n)
        misreports = data['misreports'].astype(np.float32)  # (sample_num*num_misreports, n)
        self.num_misreports = misreports.shape[0] // self.sample_num
        self.misreports = misreports.reshape((self.sample_num, self.num_misreports, self.d, self.n))  # (sample_num, num_misreports, d, n)

    def __len__(self) -> int:
        return self.sample_num

    def __getitem__(self, item: int) -> NDArrayFrame:
        return NDArrayFrame(
            self.peaks[item],  # (d, n)
            self.misreports[item],  # (num_misreports, d, n)
        )


class CustomDataModule(BaseDataModule):
    def __init__(
            self,
            train_data,
            test_data,
            n: int,
            d: int = 1,
            batch_size: int = 50,
            num_workers: int = 0,
    ):
        super(CustomDataModule, self).__init__(
            functools.partial(
                CustomDataset, train_data,
            ),
            functools.partial(
                CustomDataset, train_data,  # NOTE: no validation step
            ),
            functools.partial(
                CustomDataset, test_data
            ),
            batch_size,
            num_workers
        )
        self.save_hyperparameters()
