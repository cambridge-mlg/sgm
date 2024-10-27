from itertools import product
from pathlib import Path

ENTITY = "invariance-learners"
PROJECT = "icml2024"
CHECKPOINT_DIR = "/home/jua23/rds/hpc-work/learning-invariances-models"
ANGLES = [
    # 0,
    # 15,
    # 90,
    # 180,
    None,
]
NUM_TRNS = [
    # 3_500,
    # 7_000,
    # 12_500,
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


parent_path = Path(__file__).parent

job_folder = parent_path.parent / "jobs" / f"gen_best_sweep_camelyon"
job_folder.mkdir(exist_ok=True, parents=True)

job_file = job_folder / f"gen_best_sweep%03:00:00.txt"
if job_file.exists():
    job_file.unlink()

with job_file.open("w") as jf:
    for dataset, angle, num_trn, seed in product(DATASETS, ANGLES, NUM_TRNS, SEEDS):
        if dataset == "MNIST" and (num_trn is None or angle is None):
            continue

        if (dataset == "aug_dsprites" or dataset == "aug_dspritesv2") and not (
            (num_trn is None) and (angle is None)
        ):
            continue

        if dataset == "galaxy_mnist" and (angle is not None):
            continue

        if dataset == "patch_camelyon" and (angle is not None):
            continue

        match dataset:
            case "MNIST":
                params = (dataset, seed, angle, num_trn)
            case "galaxy_mnist" | "patch_camelyon":
                params = (dataset, seed, num_trn)
            case _:
                params = (dataset, seed)
        params = [str(p) for p in params]

        sweep_command = (
            "python "
            "experiments/train/train_generative_model.py "
            f"--gen_config=experiments/configs/gen_best.py:{','.join(params)} "
            f"--gen_config.checkpoint={CHECKPOINT_DIR}/gen_best_ckpt_{'_'.join(params)} "
            f"--inf_config=experiments/configs/inf_best.py:{','.join(params)} "
            f"--inf_config.checkpoint={CHECKPOINT_DIR}/inf_best_ckpt_{'_'.join(params)} "
            f"--wandb_project={PROJECT} "
            f"--wandb_entity={ENTITY} "
            f"--wandb_name=gen_best_{'_'.join(params)} "
            f"--wandb_tags=gen_best "
            "--rerun "
        )

        jf.write(sweep_command + "\n")
