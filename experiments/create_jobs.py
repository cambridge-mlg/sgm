from pathlib import Path
from hashlib import md5
from itertools import product

################################### Configs #####################################

EXPERIMENT_NAME = "vae_mnist_angle_sweep"
JOBS_FOLDER = f"jobs_{EXPERIMENT_NAME}"
DELETE_PREV_FOLDER = True
SCRIPT = "train.py"
# RESULTS_FOLDER = f'~/rds/rds-t2-cs133-hh9aMiOkJqI/jua23/blt/icml_results/small_scale/{EXPERIMENT_NAME}'
CONFIG_NAMES = [
    "vae_mnist",
]
ANGLES = [0, 1, 5, 15, 30, 45, 90, 135, 180, 270, 360]
NUM_TRNS = [50_000, 25_000, 12_500, 6_250, 3_125, 1_562]
TOTAL_STEPS = [15_001, 7501, 3751]
GAMMA_MULTS = [1]
SIZE_MULTS = [0.5, 1, 2]
RANDOM_SEEDS = [0]
FLAGS_TO_ADD = ["--wandb_tags=vae,mnist,angle_sweep,baseline"]

################################################################################

times = {
    "ssilvae_mnist": "01:00:00",
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
        for angle, num_trn, total_steps, γ_mult, random_seed, size_mult in product(
            ANGLES, NUM_TRNS, TOTAL_STEPS, GAMMA_MULTS, RANDOM_SEEDS, SIZE_MULTS
        ):
            for flags_to_add in FLAGS_TO_ADD:
                # unique_hash = md5("".join(sorted(flags_to_add)).strip().encode()).hexdigest()
                # if Path(f'{RESULTS_FOLDER}/{model_name}/{weight_decay}/{random_seed}/{num_epochs}/{loss_hessian_model}/{unique_hash}/0_done.txt').expanduser().exists():
                #     continue

                if random_seed == 0 and size_mult == 1 and total_steps == 15_001 and num_trn in [50_000, 12_500, 3125]:
                    continue

                conv_dims = tuple([int(s * size_mult) for s in [64, 128, 256]])
                dense_dims = tuple([int(s * size_mult) for s in [256]])
                latent_dim = int(16 * size_mult)

                line = (
                    f"{SCRIPT} "
                    f"--config configs/{config_name}.py:{angle},{num_trn},{total_steps},{γ_mult} "
                    f"--config.seed {random_seed} "
                    "--config.batch_size_eval 64 "
                    f"--config.model.Z_given_X.conv_dims '{conv_dims}' "
                    f"--config.model.Z_given_X.dense_dims '{dense_dims}' "
                    f"--config.model.X_given_Z.conv_dims '{tuple(reversed(conv_dims))}' "
                    f"--config.model.X_given_Z.dense_dims '{tuple(reversed(dense_dims))}' "
                    f"--config.model.latent_dim {latent_dim} "
                    # f'--results_folder {RESULTS_FOLDER}/{model_name}/{weight_decay}/{random_seed}/{num_epochs}/{loss_hessian_model}/{unique_hash} '
                    f" {flags_to_add}"
                )
                f.write(line + "\n")
