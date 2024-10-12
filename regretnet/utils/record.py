import sys
from dataclasses import dataclass, fields

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from ..configs import output_result_dir


@dataclass()
class Record:
    model: str
    n: int
    k: int
    d: int

    phase: Literal['train', 'test']

    mean_social_cost: float = float('inf')
    max_regret: float = float('inf')

    std_social_cost: float = float('inf')
    min_social_cost: float = float('inf')
    max_social_cost: float = float('inf')

    duration: float = float('nan')
    max_memory: float = float('nan')

    metric: float = float('inf')

    def write_csv(self, tag: str):
        p = output_result_dir.joinpath(f'{tag}.csv')
        if not p.exists():
            with open(p, 'w') as f:
                titles = ','.join(i.name for i in fields(self)) + '\n'
                f.write(titles)
                record = ','.join(str(getattr(self, i.name)) for i in fields(self)) + '\n'
                f.write(record)
        else:
            with open(p, 'a') as f:
                record = ','.join(str(getattr(self, i.name)) for i in fields(self)) + '\n'
                f.write(record)

    def __str__(self):
        return f'mSC={self.mean_social_cost} with model={self.model}, n={self.n}, k={self.k}, d={self.d}'

    def __le__(self, other: 'Record'):
        return self.metric <= other.metric
