from typing import Optional

import torch


def set_gpu_memory_limit(limit_mb: int, device: Optional[torch.device] = None):
    """Set gpu memory limit per process

    :param limit_mb: memory limit in MB
    :param device: gpu device
    :return:
    """
    free, total = torch.cuda.mem_get_info(device=device)
    total_mb = total / 1048576
    fraction = limit_mb / total_mb
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
