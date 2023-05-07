from pathlib import Path
from hashlib import md5
from itertools import product

################################### Configs #####################################

EXPERIMENT_NAME = "vae_angle_num_trn_sweep"
JOBS_FOLDER = f"jobs"
DELETE_PREV_FOLDER = True
SCRIPT = "train.py"
# RESULTS_FOLDER = f'~/rds/rds-t2-cs133-hh9aMiOkJqI/jua23/blt/icml_results/small_scale/{EXPERIMENT_NAME}'
CONFIG_NAMES = [
    "vae_mnist",
]
ANGLES = [0, 15, 30, 45, 60, 75, 90, 180]
NUM_TRNS = [10000, 20000, 30000, 40000, 50000]
RANDOM_SEEDS = [
    0,
    1,
    2,
]
FLAGS_TO_ADD = ["--wandb_tags=angle_sweep,num_trn_sweep"]

# NOTE: if you add configs you probably want to specify the results_folder more down below

################################################################################

times = {
    "vae_mnist": "00:30:00",
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
        for angle, num_trn, random_seed in product(ANGLES, NUM_TRNS, RANDOM_SEEDS):
            for flags_to_add in FLAGS_TO_ADD:
                unique_hash = md5("".join(sorted(flags_to_add)).strip().encode()).hexdigest()
                # if Path(f'{RESULTS_FOLDER}/{model_name}/{weight_decay}/{random_seed}/{num_epochs}/{loss_hessian_model}/{unique_hash}/0_done.txt').expanduser().exists():
                #     continue

                line = (
                    f"{SCRIPT} "
                    f"--config configs/{config_name}.py:{angle},{num_trn} "
                    f"--config.seed {random_seed} "
                    # f'--results_folder {RESULTS_FOLDER}/{model_name}/{weight_decay}/{random_seed}/{num_epochs}/{loss_hessian_model}/{unique_hash} '
                    f" {flags_to_add}"
                )
                f.write(line + "\n")
