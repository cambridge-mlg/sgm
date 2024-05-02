from itertools import product
from pathlib import Path

import wandb
import yaml

from experiments.utils import format_thousand

ENTITY = "invariance-learners"
PROJECT = "icml2024"
CHECKPOINT_DIR = "/home/jua23/rds/hpc-work/learning-invariances-models"
MAX_NUM_RUNS = 36
ANGLES = [
    # 0,
    # 15,
    # 90,
    # 180,
    None
]
NUM_TRNS = [
    # 50_000,
    # 37_500,
    # 25_000,
    # 12_500,
    7_000,
    3_500,
]
MODEL_NAMES = [
    "augvae",
    # "invvae",
    "vae",
    # "vae_wsda",
]
SEEDS = [
    0,
    1,
    2,
]
DATASETS = [
    # "MNIST",
    "galaxy_mnist",
]
SWEEP_TYPE = "grid"  # "grid" or "rand" or "bayes"
SWEEP_CONFIG = f"vae_angles_{SWEEP_TYPE}_hyper_sweep.yaml"

fmt_name = {
    "MNIST": "mnist",
    "galaxy_mnist": "galaxy",
}

parent_path = Path(__file__).parent
sweep_path = parent_path / SWEEP_CONFIG

for model_name in MODEL_NAMES:
    job_folder = parent_path.parent / "jobs" / f"{model_name}_{SWEEP_TYPE}_sweep"
    job_folder.mkdir(exist_ok=True)

    for dataset, angle, num_trn, seed in product(DATASETS, ANGLES, NUM_TRNS, SEEDS):
        if dataset == "MNIST" and (num_trn is None or angle is None):
            continue

        if (dataset == "galaxy_mnist") and (num_trn is None or angle is not None):
            continue

        with sweep_path.open() as file:
            sweep_config = yaml.safe_load(file)

        sweep_name = f"{model_name}_{fmt_name[dataset]}_sweep"
        if dataset == "MNIST":
            sweep_name += f"_{angle:03}"
        sweep_name += f"_{format_thousand(num_trn)}"
        sweep_name += f"_{seed}"
        print(sweep_name)
        sweep_config["name"] = sweep_name
        sweep_config["program"] = f"experiments/train/train_{model_name}.py"

        match dataset:
            case "MNIST":
                params = (dataset, seed, angle, num_trn)
            case "galaxy_mnist":
                params = (dataset, seed, num_trn)
        params = [str(p) for p in params]

        if model_name in ["augvae", "invvae"]:
            sweep_config["command"][
                2
            ] = f"--vae_config=experiments/configs/vae_{fmt_name[dataset]}.py:{','.join(params[2:])}"
            sweep_config["command"].insert(
                3,
                f"--inf_config=experiments/configs/inf_best.py:{','.join(params)}",
            )
            sweep_config["command"].insert(
                3,
                f"--gen_config=experiments/configs/gen_best.py:{','.join(params)}",
            )
            sweep_config["command"].append(
                f"--inf_config.checkpoint={CHECKPOINT_DIR}/inf_best_ckpt_{'_'.join(params)}"
            )
            sweep_config["command"].append(
                f"--gen_config.checkpoint={CHECKPOINT_DIR}/gen_best_ckpt_{'_'.join(params)}"
            )
            sweep_config["command"].append(f"--vae_config.seed={seed}")
            sweep_config["command"].append(f"--vae_config.test_split=test")
        else:
            sweep_config["command"][
                2
            ] = f"--config=experiments/configs/vae_{fmt_name[dataset]}.py:{','.join(params[2:])}"
            sweep_config["command"].append(f"--config.seed={seed}")
            sweep_config["command"].append(f"--config.test_split=test")

        # sweep_config["run_cap"] = MAX_NUM_RUNS

        sweep_id = wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)

        time = "01:30:00"
        job_file = job_folder / f"{sweep_name}%{time}.txt"
        if job_file.exists():
            job_file.unlink()
        with job_file.open("w") as job_file:
            for _ in range(MAX_NUM_RUNS):
                job_file.write(f"wandb agent --count 1 {ENTITY}/{PROJECT}/{sweep_id}\n")
            pass
