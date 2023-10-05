from itertools import product
from pathlib import Path

################################### Configs #####################################

EXPERIMENT_NAME = "vae_sweep"
JOBS_FOLDER = f"jobs_{EXPERIMENT_NAME}"
DELETE_PREV_FOLDER = True
# Set ^ to False if adding new jobs to the experiment while it is still running.
# Deleting a jobs folder while it is running will cause it to fail.
SCRIPT = "train_vae.py"
CONFIG_NAMES = [
    "vae_mnist",
]
ANGLES = [0, 45, 90, 180, 360]  # [0, 1, 5, 15, 30, 45, 90, 135, 180, 270, 360]
RANDOM_SEEDS = [0]
NUM_TRNS = [5_000, 15_000, 50_000]
INIT_LRS = [3e-4, 1e-4]
STEPS = [1_000, 3_000, 10_000]
FLAGS_TO_ADD = ["--wandb_tags=vae,mnist,angle_sweep,num_trn_sweep"]

################################################################################

times = {
    "vae_mnist": "01:00:00",
}

jobsfolder = Path(f"./{JOBS_FOLDER}")
if jobsfolder.exists() and DELETE_PREV_FOLDER:
    for jobfile in jobsfolder.glob("*"):
        jobfile.unlink()
    jobsfolder.rmdir()
jobsfolder.mkdir(exist_ok=True, parents=True)

for config_name in CONFIG_NAMES:
    jobsfile = jobsfolder / f"{EXPERIMENT_NAME}-{config_name}%{times[config_name]}.txt"
    if jobsfile.exists():
        jobsfile.unlink()
    jobsfile.touch()

    with open(jobsfile, "w") as f:
        for angle, random_seed, num_trn, init_lr, steps, flags_to_add in product(
            ANGLES, RANDOM_SEEDS, NUM_TRNS, INIT_LRS, STEPS, FLAGS_TO_ADD
        ):
            line = (
                f"{SCRIPT} "
                f"--config configs/{config_name}.py:{angle},{num_trn} "
                f"--config.seed {random_seed} "
                f"--config.init_lr {init_lr} "
                f"--config.steps {steps} "
                f"--config.warmup_steps {steps // 10}"
                f" {flags_to_add}"
            )
            f.write(line + "\n")
