from itertools import product
from pathlib import Path

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
    # "galaxy_mnist",
]

fmt_model_name = {
    "vae": "VAE",
    "augvae": "AugVAE",
    "invvae": "InvVAE",
}

parent_path = Path(__file__).parent

for model_name in MODEL_NAMES:
    job_folder = parent_path.parent / "jobs" / f"{model_name}_best_sweep"
    job_folder.mkdir(exist_ok=True)

    job_file = job_folder / f"{model_name}_best_sweep%01:30:00.txt"
    if job_file.exists():
        job_file.unlink()

    with job_file.open("w") as jf:
        for dataset, angle, num_trn, seed in product(DATASETS, ANGLES, NUM_TRNS, SEEDS):
            if dataset == "MNIST" and (num_trn is None or angle is None):
                continue

            if (dataset == "galaxy_mnist") and (num_trn is None or angle is not None):
                continue

            if (dataset == "patch_camelyon") and (num_trn is None or angle is not None):
                continue

            match dataset:
                case "MNIST":
                    params = (dataset, seed, angle, num_trn)
                case "galaxy_mnist" | "patch_camelyon":
                    params = (dataset, seed, num_trn)
            params = [str(p) for p in params]

            if model_name in ["augvae", "invvae"]:
                extra_args = (
                    f"--vae_config=experiments/configs/vae_best.py:{fmt_model_name[model_name]},{','.join(params)} "
                    "--vae_config.test_split=test "
                    f"--gen_config=experiments/configs/gen_best.py:{','.join(params)} "
                    f"--gen_config.checkpoint={CHECKPOINT_DIR}/gen_best_ckpt_{'_'.join(params)} "
                    f"--inf_config=experiments/configs/inf_best.py:{','.join(params)} "
                    f"--inf_config.checkpoint={CHECKPOINT_DIR}/inf_best_ckpt_{'_'.join(params)} "
                )
            else:
                extra_args = (
                    f"--config=experiments/configs/vae_best.py:{fmt_model_name[model_name]},{','.join(params)} "
                    "--config.test_split=test "
                )

            sweep_command = (
                "python "
                f"experiments/train/train_{model_name}.py "
                f"{extra_args}"
                f"--wandb_project={PROJECT} "
                f"--wandb_entity={ENTITY} "
                f"--wandb_name={model_name}_best_{'_'.join(params)} "
                f"--wandb_tags={model_name}_best "
                "--rerun "
            )

            jf.write(sweep_command + "\n")
