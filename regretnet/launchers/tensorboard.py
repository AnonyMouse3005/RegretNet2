import subprocess
from typing import Optional

from ..configs import output_train_log_dir


def run_tensorboard(tag: Optional[str] = None):
    """Run tensorboard server. This function blocks.

    :param tag: the tag of the experiment to inspect. If tag is None then all experiments will be inspected.
    :return:
    """
    subprocess.run(
        ['tensorboard', '--logdir', '.', '--bind_all'],
        shell=True,
        cwd=str(output_train_log_dir) if tag is None else str(output_train_log_dir.joinpath(tag))
    )
