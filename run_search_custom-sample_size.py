import contextlib
import functools
import logging
import os
import sys
from typing import Type, List, Tuple
import pickle

import torch
import numpy as np
import pytorch_lightning as pl
from torch import multiprocessing as mp

from regretnet.data_modules import BaseDataModule, CustomDataModule
from regretnet.models.rules import PercentileRuleSystem, NonSPRuleSystem, ConstantRuleSystem, DictatorRuleSystem
from regretnet.models.nets import MoulinNetSystem, RegretNetSystem
from regretnet.launchers import run_train, run_test
from regretnet.configs import output_result_dir, output_train_log_dir
from regretnet.utils import Record, lightning_logger


def model_builders(model_classes: List[Type[pl.LightningModule]], n: int, k: int, d: int, agent_weights: torch.Tensor):
    builders = [
        functools.partial(NonSPRuleSystem, n=n, k=k),
        functools.partial(PercentileRuleSystem, n=n, k=k, max_training_steps=1),
        functools.partial(DictatorRuleSystem, n=n, k=k, max_training_steps=1),
        functools.partial(ConstantRuleSystem, k=k, d=d, divisions=n*2, max_training_steps=1),
        functools.partial(MoulinNetSystem, n=n, k=k),
        functools.partial(RegretNetSystem, n=n, k=k, agent_weights=agent_weights, lr=0.005, gamma=0.99, hidden_layer_channels=[40, 40, 40, 40]),
    ]
    return list(b for b in builders if b.func in model_classes)


def train_and_test_once(model, data_module, num_epoch, device_id) -> Tuple[Record, Record]:
    ''' NOTE: comment these out because not sure what they actually do'''
    # trainer = pl.Trainer(
    #     default_root_dir=str(output_train_log_dir),
    #     accelerator='gpu',
    #     devices=[device_id],
    #     # auto_scale_batch_size='power',  # NOTE: deprecated in 2.4
    #     enable_checkpointing=False,
    #     logger=False,
    # )
    # tuner = pl.tuner.Tuner(trainer)
    # tuner.scale_batch_size(model, mode="power")
    # trainer.tune(model, datamodule=data_module)
    # del trainer
    model, train_record = run_train(data_module, model, max_epochs=num_epoch, device_id=device_id, verbose=False)
    test_record = run_test(data_module, model, device_id=device_id, verbose=False)
    return train_record, test_record


def worker_init(device_id_queue: mp.Queue):
    device_id = device_id_queue.get()
    logger = mp.get_logger().getChild(mp.current_process().name)
    mp.current_process().device_id = device_id
    mp.current_process().logger = logger
    os.environ['PL_DISABLE_FORK'] = '1'
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt='%(name)s:%(levelname)-5s:%(message)s'))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(f'Initialization complete with device_id={device_id}.')


def worker_func(model_builder, data_module_builder, num_epoch, seed) -> Tuple[Record, Record]:
    # device_id = mp.current_process().device_id
    # logger = mp.current_process().logger
    lightning_logger.setLevel(logging.ERROR + 10)
    # with contextlib.redirect_stdout(open(os.devnull, 'w')):  # NOTE: not sure why redirecting stderr/stdout to devnull
    #     with contextlib.redirect_stderr(open(os.devnull, 'w')):
    pl.seed_everything(10)
    data_module = data_module_builder()
    pl.seed_everything(seed)
    model = model_builder()
    train_record, test_record = train_and_test_once(model, data_module, num_epoch, 0)
    # logger.info(f'Task complete. Result: {str(test_record)}.')
    return train_record, test_record


def main():
    """Load custom data from file."""
    with open('data/all_data_test.pkl', 'rb') as f:
        all_test_data = pickle.load(f)

    """Change search parameters here."""
    # experiment settings
    weighted = True
    data_distributions = [
        'uniform',
        # 'normal',
        # 'beta1', 'beta2'
        ]  # ['uniform', 'normal', 'beta1', 'beta2']
    train_sizes = [
        2000,
        5000, 10000, 20000, 50000
        ]
    devices = [0]

    # select range of parameters
    n_range = [
        5,
        # 9,10
        ]
    k_range = [
        # 1,
        # 2,3,
        4]
    d_range = [1]

    # select model_classes and specify corresponding num_trials and num_epochs
    model_classes = [
        # NonSPRuleSystem, PercentileRuleSystem, MoulinNetSystem,
        RegretNetSystem]
    num_trials = [
        # 1, 1, 4,
        10]
    num_epochs = [
        # 1, 3, 100,
        1000]

    """Change search parameters above."""

    for train_size in train_sizes:
        with open(f'data/all_data_train_R{train_size:06d}.pkl', 'rb') as f:
            all_train_data = pickle.load(f)

        for data_distribution in data_distributions:
            tag = f'search_R{train_size:06d}_{data_distribution}_weighted' if weighted else f'search_R{train_size:06d}_{data_distribution}_unweighted'
            print(f'searching model={list(c.__name__ for c in model_classes)} from n={n_range}, k={k_range}, d={d_range}, {data_distribution} distribution')
            print(f'results will be stored in {str(output_result_dir.resolve())}')

            # mp.set_start_method('spawn')
            # device_id_queue = mp.Queue()
            # for d in devices:
            #     device_id_queue.put(d)

            # with mp.Pool(processes=len(devices), initializer=worker_init, initargs=(device_id_queue, )) as p:
            results = []
            for n in n_range:
                dataset = (data_distribution, n)
                print(f'running on {dataset}')
                if weighted:
                    if n == 5:
                        agent_weights = torch.tensor([5.,1.,1.,1.,1.], device=0)
                    elif n == 9:
                        agent_weights = torch.tensor([5.,5.,1.,1.,1.,1.,1.,1.,1.], device=0)
                    elif n == 10:
                        agent_weights = torch.tensor([5.,5.,1.,1.,1.,1.,1.,1.,1.,1.], device=0)
                    else:
                        agent_weights = torch.ones(n, device=0)
                else:
                    agent_weights = torch.ones(n, device=0)

                train_data, test_data = all_train_data[dataset], all_test_data[dataset]
                for d in d_range:
                    data_module_builder = functools.partial(
                        CustomDataModule, train_data, test_data, n=n, d=d
                    )
                    for k in k_range:
                        for model_builder, num_trial, num_epoch in zip(
                                model_builders(model_classes, n, k, d, agent_weights),
                                num_trials,
                                num_epochs
                        ):
                            # results.append(p.starmap_async(worker_func, list(
                            #     [model_builder, data_module_builder, num_epoch, seed]
                            #     for seed in np.random.randint(low=0, high=10000, size=num_trial)
                            # )))
                            results.append([worker_func(model_builder, data_module_builder, num_epoch, seed)
                                for seed in np.random.randint(low=0, high=10000, size=num_trial)]
                            )

            for r in results:
                # while not r.ready():
                #     r.wait(1)
                # record_list: List[Tuple[Record, Record]] = r.get()
                train_record_sc, test_record_sc = min(r, key=lambda r: r[1].metric)  # only save the best trial in terms of "metric" i.e., mean_social_cost
                train_record_rgt, test_record_rgt = min(r, key=lambda r: r[1].max_regret)  # only save the best trial in terms of max_regret
                print(f'Best record (sc): {test_record_sc}')
                print(f'Best record (rgt): {test_record_rgt}')
                train_record_sc.write_csv(tag+'_sc')
                test_record_sc.write_csv(tag+'_sc')
                train_record_rgt.write_csv(tag+'_rgt')
                test_record_rgt.write_csv(tag+'_rgt')


if __name__ == '__main__':
    main()
