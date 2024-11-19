from typing import Optional, Mapping, Iterator, Callable, List
from dataclasses import dataclass, fields
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, default_collate


@dataclass
class NDArrayFrame(Mapping[str, np.ndarray]):
    """A data structure representing one sample in cpu

    """

    _peaks: np.ndarray
    _misreports: Optional[np.ndarray] = None

    @property
    def peaks(self):
        """The true peaks of each agent with shape (d, n)"""
        return self._peaks

    @property
    def misreports(self):
        """The misreport peaks of each agent with shape (num_misreports, d, n)"""
        return self._misreports

    def __getitem__(self, __k: str) -> Optional[np.ndarray]:
        return getattr(self, __k)

    def __len__(self) -> int:
        return len([f for f in fields(self) if getattr(self, f.name) is not None])

    def __iter__(self) -> Iterator[str]:
        for f in fields(self):
            if getattr(self, f.name) is not None:
                yield f.name


@dataclass
class TensorFrame:
    """A data structure representing a batch of samples in gpu

    """

    _peaks: torch.Tensor
    _misreports: Optional[torch.Tensor] = None

    @property
    def peaks(self):
        """The true peaks of each agent with shape (batch_size, d, n)"""
        return self._peaks

    @property
    def misreports(self):
        """The misreport peaks of each agent with shape (batch_size, num_misreports, d, n)"""
        return self._misreports

    @property
    def misreport_peaks(self) -> Optional[torch.Tensor]:
        """Merge misreports into true peaks for each agent for `num_misreports` times
         with shape (batch_size, n, num_misreports, d, n)

         """
        if self.misreports is None:
            return None
        else:
            batch_size, num_misreports, d, n = self.misreports.size()
            misreport_peaks = self.peaks.unsqueeze(1).unsqueeze(2).repeat(1, n, num_misreports, 1, 1)
            for i in range(n):
                misreport_peaks[:, i, :, :, i] = self.misreports[:, :, :, i]  # for each misreport of i, there are n additional profiles (since we want to keep peaks from other agents intact)
            return misreport_peaks  # (batch_size, n, num_misreports, d, n)

    def to(self, device: torch.device):
        return TensorFrame(
            _peaks=self.peaks.to(device),
            _misreports=None if self.misreports is None else self.misreports.to(device)
        )

    @classmethod
    def from_ndarray_frames(cls, frames: List[NDArrayFrame]):
        return cls(**default_collate(list(dict(f) for f in frames)))


class BaseDataset(Dataset):
    """ Base Dataset for agent peak sampling.

    """
    def __len__(self) -> int:
        """

        :return: the length of the dataset
        """
        raise NotImplementedError()

    def __getitem__(self, item: int) -> NDArrayFrame:
        """

        :param item: the index
        :return: the sample at index
        """
        raise NotImplementedError()


class BaseDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_set_factory: Optional[Callable[[], Dataset]] = None,
            val_set_factory: Optional[Callable[[], Dataset]] = None,
            test_set_factory: Optional[Callable[[], Dataset]] = None,
            batch_size: int = 1000,
            num_workers: int = 0,
            on_the_fly: bool = False,  # if data (randomly) generated on the fly within factory
    ):
        """Base DataModule that wraps datasets to interact with train, val and test actions.

        :param train_set_factory: the training dataset builder function
        :param val_set_factory: the validation dataset builder function
        :param test_set_factory: the test dataset builder function
        :param batch_size: number of samples for each pass
        :param num_workers: number of worker threads
        """
        super(BaseDataModule, self).__init__()
        self.train_set_factory = train_set_factory
        self.val_set_factory = val_set_factory
        self.test_set_factory = test_set_factory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.on_the_fly = on_the_fly

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' and self.train_set_factory is not None:
            self.train_set = self.train_set_factory()
        if stage in ['fit', 'validate'] and self.val_set_factory is not None:
            if self.on_the_fly:
                self.val_set = deepcopy(self.train_set)  # no validation step (i.e., use entire train set for validation)
            else:
                self.val_set = self.val_set_factory()
        if stage in ['test', 'predict'] and self.test_set_factory is not None:
            self.test_set = self.test_set_factory()

    def train_dataloader(self) -> Optional[DataLoader]:
        if self.train_set is not None:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=TensorFrame.from_ndarray_frames,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=False if self.num_workers == 0 else True,
            )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_set is not None:
            return DataLoader(
                self.val_set,
                batch_size=self.val_set.sample_num,
                shuffle=False,
                collate_fn=TensorFrame.from_ndarray_frames,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=False if self.num_workers == 0 else True,
            )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_set is not None:
            return DataLoader(
                self.test_set,
                batch_size=self.test_set.sample_num,
                shuffle=False,
                collate_fn=TensorFrame.from_ndarray_frames,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=False if self.num_workers == 0 else True,
            )

    def predict_dataloader(self) -> Optional[DataLoader]:
        return self.test_dataloader()
