from hashlib import md5
from itertools import product
from pathlib import Path

################################### Configs #####################################

EXPERIMENT_NAME = "test_pgm_sweep"
JOBS_FOLDER = f"jobs_{EXPERIMENT_NAME}"
DELETE_PREV_FOLDER = True
# Set ^ to False if adding new jobs to the experiment while it is still running.
# Deleting a jobs folder while it is running will cause it to fail.
SCRIPT = "train.py"
# RESULTS_FOLDER = f'~/rds/rds-t2-cs133-hh9aMiOkJqI/jua23/blt/icml_results/small_scale/{EXPERIMENT_NAME}'
CONFIG_NAMES = [
    "pgm_mnist",
]
ANGLES = [0, 1, 5, 15, 30, 45, 90, 135, 180, 270, 360]
RANDOM_SEEDS = [0, 1, 2]
FLAGS_TO_ADD = ["--wandb_tags=pgm,mnist,angle_sweep"]

################################################################################

times = {
    "pgm_sweep": "00:30:00",
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
        for angle, random_seed, flags_to_add in product(
            ANGLES, RANDOM_SEEDS, FLAGS_TO_ADD
        ):
            # unique_hash = md5("".join(sorted(flags_to_add)).strip().encode()).hexdigest()

            line = (
                f"{SCRIPT} "
                f"--config configs/{config_name}.py:{angle} "
                f"--config.seed {random_seed} "
                # f'--config.results_folder {RESULTS_FOLDER}/{unique_hash} '
                f" {flags_to_add}"
            )
            f.write(line + "\n")
