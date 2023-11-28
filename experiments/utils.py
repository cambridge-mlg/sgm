import wandb
from absl import flags, logging
from ml_collections import config_dict, config_flags

FLAGS = flags.FLAGS


def duplicated_run(config):
    fake_run = wandb.init(
        mode="disabled",
        config=config.to_dict(),
    )
    # ^ we create this fake run to get the config dict in the same format as the existing runs in wandb

    logging.info("Checking if config already exists in wandb.")
    runs = wandb.Api().runs(f"{FLAGS.wandb_entity}/{FLAGS.wandb_project}")
    finished_runs = [run for run in runs if run.state == "finished"]
    frozen_config = config_dict.FrozenConfigDict(fake_run.config)
    logging.info(f"checking {len(finished_runs)} runs")

    for run in finished_runs:
        run_config = config_dict.FrozenConfigDict(run.config)
        if frozen_config == run_config:
            logging.info(
                f"Found matching config in run with id {run.id} and name {run.name}."
                "Skipping training. Use --rerun to rerun the experiment. Config was: \n"
                f"{frozen_config}"
            )
            return True

    return False


def format_thousand(num):
    thousands = num // 1000
    remainder = num % 1000 // 100
    return f"{thousands}k{remainder if remainder != 0 else ''}"
