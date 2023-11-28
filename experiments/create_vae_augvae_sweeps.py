from itertools import product
from pathlib import Path

import yaml

import wandb
from experiments.utils import format_thousand

ENTITY = "invariance-learners"
PROJECT = "iclr2024experiments"
MAX_NUM_RUNS = 32
SWEEP_CONFIG = "vae_angles_hyper_sweep.yaml"
ANGLES = [0, 15, 90, 180]
NUM_TRNS = [50_000]
MODEL_NAMES = [
    # "augvae",
    "vae",
]


parent_path = Path(__file__).parent
sweep_path = parent_path / SWEEP_CONFIG

for model_name in MODEL_NAMES:
    job_folder = parent_path / f"jobs_{model_name}_sweep"
    job_folder.mkdir(exist_ok=True)

    for angle, num_trn in product(ANGLES, NUM_TRNS):
        with sweep_path.open() as file:
            sweep_config = yaml.safe_load(file)

        sweep_name = f"{model_name}_sweep_{angle:03}_{format_thousand(num_trn)}_v2"
        print(sweep_name)
        sweep_config["name"] = sweep_name

        if model_name == "augvae":
            sweep_config["command"][
                2
            ] = f"--vae_config=experiments/configs/vae_mnist.py:{angle},{num_trn}"
            sweep_config["command"].insert(
                3, f"--pgm_config=experiments/configs/pgm_mnist.py:{angle},{num_trn}"
            )
            sweep_config["program"] = "experiments/train_augvae.py"
        else:
            sweep_config["command"][
                2
            ] = f"--config=experiments/configs/vae_mnist.py:{angle},{num_trn}"

        sweep_config["run_cap"] = MAX_NUM_RUNS

        sweep_id = wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)

        time = "00:30:00" if model_name == "vae" else "01:00:00"
        job_file = job_folder / f"{sweep_name}%{time}.txt"
        if job_file.exists():
            job_file.unlink()
        with job_file.open("w") as job_file:
            for _ in range(MAX_NUM_RUNS):
                job_file.write(f"wandb agent --count 1 {ENTITY}/{PROJECT}/{sweep_id}\n")
            pass
