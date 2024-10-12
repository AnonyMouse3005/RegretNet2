import time
import os

import pytorch_lightning as pl

from regretnet.data_modules import UniformDataModule, BaseDataModule
from regretnet.models.rules import PercentileRuleSystem, NonSPRuleSystem, ConstantRuleSystem, DictatorRuleSystem
from regretnet.models.nets import MoulinNetSystem, RegretNetSystem
from regretnet.launchers import run_train, run_test
from regretnet.utils import set_gpu_memory_limit, has_ckpt, load_ckpt


def main():
    """Change parameters here."""
    # experiment settings
    device = 0  # only support one device because of auto batch_size tuning.

    # parameters
    n = 5
    k = 2
    d = 1
    seed = 40

    # dataset and model
    data_module = UniformDataModule(n=n, d=d)
    model = RegretNetSystem(n=n, k=k)
    num_epoch = 1000

    """Change parameters above."""

    tag = f'n{n}_k{k}_d{d}_seed{seed}_{model.__class__.__name__}'

    pl.seed_everything(seed)
    set_gpu_memory_limit(1024)

    # load from tag checkpoint or train from start
    if has_ckpt(tag):
        print(f'load from {tag} checkpoint')
        model = load_ckpt(model, tag)
    else:
        print('train from start')
        model, _ = run_train(data_module, model, tag, max_epochs=num_epoch, device_id=device)

    time.sleep(1)
    while True:
        sample = input(f'input a list of {n} agent locations or quit:')
        if sample.lower() == 'quit':
            break
        try:
            sample = list(float(s) for s in sample.split(' '))
            assert max(sample) <= 1, 'the agent peaks should not be greater than 1'
            assert min(sample) >= 0, 'the agent peaks should not be less than 0'
            assert len(sample) == n, f'there should be {n} agents'
        except ValueError as e:
            print('cannot parse input, should be a list of float split by space')
        except AssertionError as e:
            print(e)
            continue

        print(f'{model.__class__.__name__}:')
        print(model.predict_once(sample))


if __name__ == '__main__':
    main()
