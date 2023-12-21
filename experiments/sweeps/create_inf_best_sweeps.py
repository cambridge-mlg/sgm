from itertools import product
from pathlib import Path

ENTITY = "invariance-learners"
PROJECT = "icml2024"
CHECKPOINT_DIR = "/home/jua23/rds/hpc-work/learning-invariances-models"
ANGLES = [
    0,
    15,
    90,
    180,
    None,
]
NUM_TRNS = [
    25_000,
    37_500,
    50_000,
    None,
]
SEEDS = [
    0,
    1,
    2,
]
DATASETS = [
    "MNIST",
    "aug_dsprites",
]


parent_path = Path(__file__).parent

job_folder = parent_path.parent / "jobs" / f"inf_best_sweep"
job_folder.mkdir(exist_ok=True, parents=True)

job_file = job_folder / f"inf_best_sweep%01:30:00.txt"
if job_file.exists():
    job_file.unlink()

with job_file.open("w") as jf:
    for dataset, angle, num_trn, seed in product(DATASETS, ANGLES, NUM_TRNS, SEEDS):
        if dataset == "MNIST" and (num_trn is None or angle is None):
            continue

        if dataset == "aug_dsprites" and not (num_trn is None and angle is not None):
            continue

        params = (
            (dataset, seed, angle, num_trn) if dataset == "MNIST" else (dataset, seed)
        )
        params = [str(p) for p in params]

        sweep_command = (
            "python "
            "experiments/train/train_inference_model.py "
            f"--config=experiments/configs/inf_best.py:{','.join(params)} "
            f"--config.checkpoint={CHECKPOINT_DIR}/inf_best_ckpt_{'_'.join(params)} "
            f"--wandb_project={PROJECT} "
            f"--wandb_entity={ENTITY} "
            f"--wandb_name=inf_best_{'_'.join(params)} "
            f"--wandb_tags=inf_best "
            "--rerun "
        )

        jf.write(sweep_command + "\n")
