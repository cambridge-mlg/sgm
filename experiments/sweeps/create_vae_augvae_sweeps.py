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
    0,
    15,
    90,
    180,
]
NUM_TRNS = [
    50_000,
    37_500,
    25_000,
    12_500,
]
MODEL_NAMES = [
    # "augvae",
    # "invvae",
    # "vae",
    "vae_wsda"
]
SEEDS = [
    # 0,
    1,
    2,
]
SWEEP_TYPE = "grid"  # "grid" or "rand" or "bayes"
SWEEP_CONFIG = f"vae_angles_{SWEEP_TYPE}_hyper_sweep.yaml"

parent_path = Path(__file__).parent
sweep_path = parent_path / SWEEP_CONFIG

for model_name in MODEL_NAMES:
    job_folder = parent_path.parent / "jobs" / f"{model_name}_{SWEEP_TYPE}_sweep12"
    job_folder.mkdir(exist_ok=True)

    for angle, num_trn, seed in product(ANGLES, NUM_TRNS, SEEDS):
        # match (model_name, angle, num_trn, seed):
        #     case ("vae" | "augvae", 0 | 180, 50_000 | 25_000, 0):
        #         continue

        with sweep_path.open() as file:
            sweep_config = yaml.safe_load(file)

        sweep_name = f"{model_name}_sweep_{angle:03}_{format_thousand(num_trn)}_{seed}"
        print(sweep_name)
        sweep_config["name"] = sweep_name
        sweep_config["program"] = f"experiments/train/train_{model_name}.py"

        if model_name in ["augvae", "invvae"]:
            sweep_config["command"][
                2
            ] = f"--vae_config=experiments/configs/vae_mnist.py:{angle},{num_trn}"
            sweep_config["command"].insert(
                3,
                f"--inf_config=experiments/configs/inf_best.py:MNIST,{seed},{angle},{num_trn}",
            )
            sweep_config["command"].insert(
                3,
                f"--gen_config=experiments/configs/gen_best.py:MNIST,{seed},{angle},{num_trn}",
            )
            sweep_config["command"].append(
                f"--inf_config.checkpoint={CHECKPOINT_DIR}/inf_best_ckpt_MNIST_{seed}_{angle}_{num_trn}"
            )
            sweep_config["command"].append(
                f"--gen_config.checkpoint={CHECKPOINT_DIR}/gen_best_ckpt_MNIST_{seed}_{angle}_{num_trn}"
            )
            sweep_config["command"].append(f"--vae_config.seed={seed}")
            sweep_config["command"].append(f"--vae_config.test_split=test")
        else:
            sweep_config["command"][
                2
            ] = f"--config=experiments/configs/vae_mnist.py:{angle},{num_trn}"
            sweep_config["command"].append(f"--config.seed={seed}")
            sweep_config["command"].append(f"--config.test_split=test")

        # sweep_config["run_cap"] = MAX_NUM_RUNS

        sweep_id = wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)

        time = "01:00:00"
        job_file = job_folder / f"{sweep_name}%{time}.txt"
        if job_file.exists():
            job_file.unlink()
        with job_file.open("w") as job_file:
            for _ in range(MAX_NUM_RUNS):
                job_file.write(f"wandb agent --count 1 {ENTITY}/{PROJECT}/{sweep_id}\n")
            pass
