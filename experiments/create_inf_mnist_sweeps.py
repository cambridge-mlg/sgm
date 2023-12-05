from itertools import product
from pathlib import Path

import wandb
import yaml

from experiments.utils import format_thousand

ENTITY = "invariance-learners"
PROJECT = "icml2024"
MAX_NUM_RUNS = 144
ANGLES = [0]
NUM_TRNS = [
    # 25_000,
    # 37_500,
    50_000,
]
SWEEP_TYPE = "grid"  # "rand" or "bayes"
SWEEP_CONFIG = f"inf_{SWEEP_TYPE}_hyper_sweep.yaml"


parent_path = Path(__file__).parent
sweep_path = parent_path / SWEEP_CONFIG

job_folder = parent_path / f"jobs_inf_{SWEEP_TYPE}_mnist_sweep_seed1"
job_folder.mkdir(exist_ok=True)

for angle, num_trn in product(ANGLES, NUM_TRNS):
    with sweep_path.open() as file:
        sweep_config = yaml.safe_load(file)

    sweep_name = f"inf_{SWEEP_TYPE}_mnist_sweep_{angle:03}_{format_thousand(num_trn)}_1"
    print(sweep_name)
    sweep_config["name"] = sweep_name
    sweep_config["command"][
        2
    ] = f"--config=experiments/configs/inf_mnist.py:{angle},{num_trn}"

    sweep_config["run_cap"] = MAX_NUM_RUNS

    sweep_id = wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)

    job_file = job_folder / f"{sweep_name}%01:30:00.txt"
    if job_file.exists():
        job_file.unlink()
    with job_file.open("w") as job_file:
        for _ in range(MAX_NUM_RUNS):
            job_file.write(f"wandb agent --count 1 {ENTITY}/{PROJECT}/{sweep_id}\n")
        pass
