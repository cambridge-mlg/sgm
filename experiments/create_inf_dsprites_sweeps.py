from itertools import product
from pathlib import Path

import wandb
import yaml

from experiments.utils import format_thousand

ENTITY = "invariance-learners"
PROJECT = "iclr2024experiments"
MAX_NUM_RUNS = 32
SWEEP_TYPE = "bayes"  # "rand"
SWEEP_CONFIG = f"inf_{SWEEP_TYPE}_hyper_sweep.yaml"


parent_path = Path(__file__).parent
sweep_path = parent_path / SWEEP_CONFIG

job_folder = parent_path / f"jobs_inf_{SWEEP_TYPE}_dsprites_sweep"
job_folder.mkdir(exist_ok=True)

with sweep_path.open() as file:
    sweep_config = yaml.safe_load(file)

sweep_name = f"inf_{SWEEP_TYPE}_dsprites_sweep"
print(sweep_name)
sweep_config["name"] = sweep_name
sweep_config["command"][2] = f"--config=experiments/configs/inf_dsprites.py"

sweep_config["run_cap"] = MAX_NUM_RUNS

sweep_id = wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)

job_file = job_folder / f"{sweep_name}%01:00:00.txt"
if job_file.exists():
    job_file.unlink()
with job_file.open("w") as job_file:
    for _ in range(MAX_NUM_RUNS):
        job_file.write(f"wandb agent --count 1 {ENTITY}/{PROJECT}/{sweep_id}\n")
    pass
