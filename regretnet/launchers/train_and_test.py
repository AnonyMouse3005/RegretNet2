import datetime
import random
import time
from typing import Dict, Optional, Tuple

import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerStatus

from ..data_modules import BaseDataModule
from ..models import BaseSystem
from ..configs import output_train_log_dir, output_ckpt_dir
from ..utils import Record


def run_train(
        data_module: BaseDataModule,
        model: BaseSystem,
        tag: Optional[str] = None,
        max_time: Optional[int] = None,
        max_epochs: int = 1000,
        device_id: Optional[int] = None,
        verbose: bool = True,
) -> Tuple[BaseSystem, Record]:
    """Train the model using data_module.

    :param data_module: DataModules from `regretnet.data_modules`
    :param model: Systems from `regretnet.models`
    :param tag: The string identifier of the experiment
    :param max_time: the time limit of the training process
    :param max_epochs: the epoch limit of the training process
    :param device_id: the gpu id to run
    :param verbose: output logs
    :return: The trained model and metric record
    """
    if max_time is not None:
        max_time = datetime.timedelta(seconds=max_time)
    if tag is None:
        filename = 'no_tag-{epoch}-{mean_social_cost:.5f}-' + f'{random.randint(0, 9999999999)}'
    else:
        filename = tag

    trainer = pl.Trainer(
        default_root_dir=str(output_train_log_dir),
        accelerator='gpu',
        devices=[device_id],
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='metric',
                mode='min',
                dirpath=str(output_ckpt_dir),
                filename=filename
            ),
        ],
        check_val_every_n_epoch=max(max_epochs // 50, 1),
        max_epochs=max_epochs,
        max_time=max_time,
        enable_checkpointing=True,
        enable_model_summary=verbose,
        enable_progress_bar=verbose,
        logger=verbose,
    )

    # fit
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()
    r = Record(model=model.__class__.__name__, n=data_module.hparams.n, k=model.hparams.k, d=data_module.hparams.d, phase='train')

    try:
        trainer.fit(model, datamodule=data_module)
        r.duration = time.time() - start_time

        if trainer.state.status == TrainerStatus.INTERRUPTED:
            # keyboard exit
            raise RuntimeError('keyboard interrupt')
        elif max_time is not None and datetime.timedelta(seconds=r.duration) > max_time:
            # time limit exceeded
            raise RuntimeError('time limit exceeded')

        r.mean_social_cost = trainer.logged_metrics['mean_social_cost'].item()
        r.std_social_cost = trainer.logged_metrics['std_social_cost'].item()
        r.min_social_cost = trainer.logged_metrics['min_social_cost'].item()
        r.max_social_cost = trainer.logged_metrics['max_social_cost'].item()
        r.max_regret = trainer.logged_metrics['max_regret'].item() if 'max_regret' in trainer.logged_metrics else float(
            'nan')
        r.max_memory = torch.cuda.max_memory_allocated() / (2 ** 20) / data_module.batch_size
        r.metric = trainer.logged_metrics['metric'].item()

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            # out of cuda memory
            r.max_memory = float('inf')
        elif 'time limit exceeded' in str(e).lower():
            # time limit exceeded
            r.max_memory = torch.cuda.max_memory_allocated() / (2 ** 20) / data_module.batch_size
            r.duration = float('inf')
        else:
            raise
    except PermissionError as e:
        r.max_memory = torch.cuda.max_memory_allocated() / (2 ** 20) / data_module.batch_size
        r.duration = float('inf')
    except MemoryError as e:
        # out of cpu memory
        r.max_memory = float('inf')

    model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    if tag is not None:
        r.write_csv(tag)
    return model, r


def run_test(
        data_module: BaseDataModule,
        model: BaseSystem,
        tag: Optional[str] = None,
        device_id: Optional[int] = None,
        verbose: bool = True,
) -> Record:
    """Test the model with data_module.

    :param data_module: DataModules from `regretnet.data_modules`
    :param model: Systems from `regretnet.models`
    :param tag: The string identifier of the experiment
    :param device_id: the gpu id to run
    :param verbose: output logs
    :return:
    """
    trainer = pl.Trainer(
        default_root_dir=str(output_train_log_dir),
        accelerator='gpu',
        devices=[device_id],
        enable_checkpointing=False,
        enable_model_summary=verbose,
        enable_progress_bar=verbose,
        logger=False,
    )

    # test
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()
    r = Record(model=model.__class__.__name__, n=data_module.hparams.n, k=model.hparams.k, d=data_module.hparams.d, phase='test')

    try:
        trainer.test(model, datamodule=data_module)
        r.duration = time.time() - start_time

        if trainer.state.status == TrainerStatus.INTERRUPTED:
            # keyboard exit
            raise RuntimeError('keyboard interrupt')

        r.mean_social_cost = trainer.logged_metrics['mean_social_cost'].item()
        r.std_social_cost = trainer.logged_metrics['std_social_cost'].item()
        r.min_social_cost = trainer.logged_metrics['min_social_cost'].item()
        r.max_social_cost = trainer.logged_metrics['max_social_cost'].item()
        r.max_regret = trainer.logged_metrics['max_regret'].item() if 'max_regret' in trainer.logged_metrics else float(
            'nan')
        r.max_memory = torch.cuda.max_memory_allocated() / (2 ** 20) / data_module.batch_size
        r.metric = trainer.logged_metrics['metric'].item()

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            # out of cuda memory
            r.max_memory = float('inf')
        elif 'time limit exceeded' in str(e).lower():
            # time limit exceeded
            r.max_memory = torch.cuda.max_memory_allocated() / (2 ** 20) / data_module.batch_size
            r.duration = float('inf')
        else:
            raise
    except PermissionError as e:
        r.max_memory = torch.cuda.max_memory_allocated() / (2 ** 20) / data_module.batch_size
        r.duration = float('inf')
    except MemoryError as e:
        # out of cpu memory
        r.max_memory = float('inf')

    if tag is not None:
        r.write_csv(tag)
    return r
