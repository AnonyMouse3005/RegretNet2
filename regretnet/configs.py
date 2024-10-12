from pathlib import Path

s = set(locals().keys())

root_dir: Path = Path(__file__).parents[1]
output_dir: Path = root_dir.joinpath('outputs')
output_ckpt_dir: Path = output_dir.joinpath('checkpoints')
output_result_dir: Path = output_dir.joinpath('results')
output_train_log_dir: Path = output_dir.joinpath('train_logs')

s = set(locals().keys()) - s
for name in s:
    if name[-3:] == 'dir' and isinstance(locals()[name], Path):
        locals()[name].mkdir(exist_ok=True, parents=True)
