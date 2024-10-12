from typing import Union, Type

from ..models import BaseSystem
from ..configs import output_ckpt_dir


def load_ckpt(model: Union[BaseSystem, Type[BaseSystem]], tag: str):
    p = output_ckpt_dir.joinpath(f'{tag}.ckpt')
    if p.exists():
        return model.load_from_checkpoint(str(p))
    else:
        raise FileNotFoundError(f'checkpoint for {tag} not found!')


def has_ckpt(tag: str):
    p = output_ckpt_dir.joinpath(f'{tag}.ckpt')
    return p.exists()
