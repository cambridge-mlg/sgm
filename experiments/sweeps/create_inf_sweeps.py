from itertools import product
from pathlib import Path

import wandb
import yaml

from experiments.utils import format_thousand

ENTITY = "invariance-learners"
PROJECT = "icml2024"
MAX_NUM_RUNS = 144
ANGLES = [
    0,
    # 90,
    # 180,
    # None,
]
NUM_TRNS = [
    12_500,
    # 25_000,
    # 37_500,
    # 50_000,
    # None,
]
SEEDS = [
    # 0,
    1,
    2,
]
DATASETS = [
    "MNIST",
    # "aug_dsprites",
    # "aug_dspritesv2",
]
SWEEP_TYPE = "grid"  # "grid" or "rand" or "bayes"
SWEEP_CONFIG = f"inf_{SWEEP_TYPE}_hyper_sweep.yaml"

fmt_name = lambda dataset_name: dataset_name.split("_")[-1].lower()

parent_path = Path(__file__).parent
sweep_path = parent_path / SWEEP_CONFIG

job_folder = parent_path.parent / "jobs" / f"inf_{SWEEP_TYPE}_sweep_12k5_23"
job_folder.mkdir(exist_ok=True)

for dataset, angle, num_trn, seed in product(DATASETS, ANGLES, NUM_TRNS, SEEDS):
    if dataset == "MNIST" and (num_trn is None or angle is None):
        continue

    if (dataset == "aug_dsprites" or dataset == "aug_dspritesv2") and not (
        num_trn is None and angle is None
    ):
        continue

    with sweep_path.open() as file:
        sweep_config = yaml.safe_load(file)

    sweep_name = f"inf_{SWEEP_TYPE}_{fmt_name(dataset)}_sweep"
    if num_trn is not None and angle is not None:
        sweep_name += f"_{angle:03}_{format_thousand(num_trn)}"
    sweep_name += f"_{seed}"
    print(sweep_name)
    sweep_config["name"] = sweep_name

    if dataset == "MNIST":
        sweep_config["command"][
            2
        ] = f"--config=experiments/configs/inf_{fmt_name(dataset)}.py:{angle},{num_trn}"
    else:
        sweep_config["command"][
            2
        ] = f"--config=experiments/configs/inf_{fmt_name(dataset)}.py"

    sweep_config["command"].append(f"--config.seed={seed}")

    sweep_config["run_cap"] = MAX_NUM_RUNS

    sweep_id = wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)

    job_file = job_folder / f"{sweep_name}%01:30:00.txt"
    if job_file.exists():
        job_file.unlink()
    with job_file.open("w") as job_file:
        for _ in range(MAX_NUM_RUNS):
            job_file.write(f"wandb agent --count 1 {ENTITY}/{PROJECT}/{sweep_id}\n")
        pass
