from itertools import product
from pathlib import Path

import wandb
import yaml

from experiments.utils import format_thousand

ENTITY = "invariance-learners"
PROJECT = "icml2024"
MAX_NUM_RUNS = 216  # 288
ANGLES = [
    # 0,
    # 90,
    # 180,
    None,
]
NUM_TRNS = [
    # 3_500,
    # 7_000,
    # 25_000,
    # 37_500,
    # 50_000,
    # None,
    262_144,
    65_536,
    16_384,
]
SEEDS = [
    0,
    # 1,
    # 2,
]
DATASETS = [
    # "MNIST",
    # "aug_dsprites",
    # "aug_dspritesv2",
    # "galaxy_mnist",
    "patch_camelyon",
]
SWEEP_TYPE = "grid"  # "grid" or "rand" or "bayes"
SWEEP_CONFIG = f"gen_{SWEEP_TYPE}_hyper_sweep_camelyon.yaml"

fmt_name = {
    "MNIST": "mnist",
    "aug_dsprites": "dsprites",
    "aug_dspritesv2": "dspritesv2",
    "galaxy_mnist": "galaxy",
    "patch_camelyon": "camelyon",
}

parent_path = Path(__file__).parent
sweep_path = parent_path / SWEEP_CONFIG

job_folder = parent_path.parent / "jobs" / f"gen_{SWEEP_TYPE}_sweep_camelyon"
job_folder.mkdir(exist_ok=True)

for dataset, angle, num_trn, seed in product(DATASETS, ANGLES, NUM_TRNS, SEEDS):
    if dataset == "MNIST" and (num_trn is None or angle is None):
        continue

    if (dataset == "aug_dsprites" or dataset == "aug_dspritesv2") and not (
        num_trn is None and angle is None
    ):
        continue

    if (dataset == "galaxy_mnist") and (num_trn is None or angle is not None):
        continue

    if (dataset == "patch_camelyon") and (num_trn is None or angle is not None):
        continue

    with sweep_path.open() as file:
        sweep_config = yaml.safe_load(file)

    sweep_name = f"gen_{SWEEP_TYPE}_{fmt_name[dataset]}_sweep"
    if dataset == "MNIST":
        sweep_name += f"_{angle:03}_{format_thousand(num_trn)}"
    if dataset == "galaxy_mnist" or dataset == "patch_camelyon":
        sweep_name += f"_{format_thousand(num_trn)}"
    sweep_name += f"_{seed}"
    print(sweep_name)
    sweep_config["name"] = sweep_name

    sweep_command_gen = f"--gen_config=experiments/configs/gen_{fmt_name[dataset]}.py"
    if dataset == "MNIST":
        sweep_command_gen += f":{angle},{num_trn}"
    if dataset == "galaxy_mnist" or dataset == "patch_camelyon":
        sweep_command_gen += f":{num_trn}"
    sweep_config["command"][2] = sweep_command_gen

    sweep_config_inf = f"--inf_config=experiments/configs/inf_best.py:{dataset},{seed}"
    if dataset == "MNIST":
        sweep_config_inf += f",0,{num_trn}"
    if dataset == "galaxy_mnist" or dataset == "patch_camelyon":
        sweep_config_inf += f",{num_trn}"
    sweep_config["command"][3] = sweep_config_inf

    sweep_config["command"].append(f"--gen_config.seed={seed}")

    inf_ckpt = f"/home/jua23/rds/hpc-work/learning-invariances-models/inf_best_ckpt_{dataset}_{seed}"
    if dataset == "MNIST":
        inf_ckpt += f"_0_{num_trn}"
    if dataset == "galaxy_mnist" or dataset == "patch_camelyon":
        inf_ckpt += f"_{num_trn}"
    sweep_config["command"].append(f"--inf_config.checkpoint={inf_ckpt}")

    sweep_config["run_cap"] = MAX_NUM_RUNS

    sweep_id = wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)

    job_file = job_folder / f"{sweep_name}%03:00:00.txt"
    if job_file.exists():
        job_file.unlink()
    with job_file.open("w") as job_file:
        for _ in range(MAX_NUM_RUNS):
            job_file.write(f"wandb agent --count 1 {ENTITY}/{PROJECT}/{sweep_id}\n")
        pass
