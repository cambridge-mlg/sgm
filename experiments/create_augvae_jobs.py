from pathlib import Path
from hashlib import md5
from itertools import product

################################### Configs #####################################

EXPERIMENT_NAME = "augvae_mnist_angle_sweep"
JOBS_FOLDER = f"jobs_{EXPERIMENT_NAME}"
DELETE_PREV_FOLDER = True
SCRIPT = "train_ssilvae.py"
# RESULTS_FOLDER = f'~/rds/rds-t2-cs133-hh9aMiOkJqI/jua23/blt/icml_results/small_scale/{EXPERIMENT_NAME}'

ANGLES = [0, 1, 5, 15, 30, 45, 90, 135, 180, 270, 360]
NUM_TRNS = [50_000] # [50_000, 25_000, 12_500, 6_250, 3_125, 1_562]
TOTAL_STEPS = [7501] # [15_001, 7501, 3751]
RANDOM_SEEDS = [0]
FLAGS_TO_ADD = ["--wandb_tags=mnist,angle_sweep"]

################################################################################

jobsfolder = Path(f"./{JOBS_FOLDER}")
if jobsfolder.exists() and DELETE_PREV_FOLDER:
    for jobfile in jobsfolder.glob("*"):
        jobfile.unlink()
    jobsfolder.rmdir()
jobsfolder.mkdir(exist_ok=True, parents=True)

TIME = "00:45:00"

jobsfile = jobsfolder / f"{EXPERIMENT_NAME}-%{TIME}.txt"
if jobsfile.exists():
    jobsfile.unlink()
jobsfile.touch()

with open(jobsfile, "w") as f:
    for angle, num_trn, total_steps, random_seed in product(
        ANGLES, NUM_TRNS, TOTAL_STEPS, RANDOM_SEEDS
    ):
        for flags_to_add in FLAGS_TO_ADD:
            # unique_hash = md5("".join(sorted(flags_to_add)).strip().encode()).hexdigest()
            # if Path(f'{RESULTS_FOLDER}/{model_name}/{weight_decay}/{random_seed}/{num_epochs}/{loss_hessian_model}/{unique_hash}/0_done.txt').expanduser().exists():
            #     continue

            line = (
                f"{SCRIPT} "
                f"--angle {angle} "
                f"--num_trn {num_trn} "
                f"--total_steps {total_steps} "
                f"--seed {random_seed} "
                f" {flags_to_add}"
            )
            f.write(line + "\n")
