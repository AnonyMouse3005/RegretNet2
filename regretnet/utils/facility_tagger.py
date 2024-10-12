import platform
from dataclasses import dataclass

if platform.python_version_tuple() > ('3', '6'):
    from typing import Literal, Union, List
else:
    from typing import Union
    from typing_extensions import Literal

import numpy as np


@dataclass
class FacilityTag:
    mechanism: str = Literal['Percentile', 'Constant']
    value: Union[float, int] = 0


def tag_facilities(peaks: np.ndarray, facilities: np.ndarray) -> List[List[FacilityTag]]:
    tags = []
    for p, f in zip(peaks, facilities):
        current_tags = []
        sorted_p = sorted(p)
        sorted_f = sorted(f)
        
