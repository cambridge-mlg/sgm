import os

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import ciclo
import flax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import wandb
from absl import app, flags, logging
from clu import deterministic_data, parameter_overview
from jax.config import config as jax_config
from ml_collections import config_dict, config_flags

from src.models.utils import reset_metrics
from src.models.vae import (
    VAE,
    create_vae_state,
    make_vae_plotting_fns,
    make_vae_train_and_eval,
)
from src.utils.input import get_data
from src.utils.training import custom_wandb_logger

flax.config.update("flax_use_orbax_checkpointing", True)
logging.set_verbosity(logging.INFO)
plt.rcParams["savefig.facecolor"] = "white"

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config")
flags.mark_flag_as_required("config")
flags.DEFINE_enum(
    "wandb_mode", "online", ["online", "offline", "disabled"], "Mode for wandb.run."
)
flags.DEFINE_list("wandb_tags", [], "Tags for wandb.run.")
flags.DEFINE_string("wandb_notes", "", "Notes for wandb.run.")
flags.DEFINE_string("wandb_project", "aistats2024", "Project for wandb.run.")
flags.DEFINE_string("wandb_entity", "invariance-learners", "Entity for wandb.run.")
flags.DEFINE_string("wandb_name", None, "Name for wandb.run.")
flags.DEFINE_bool(
    "rerun", False, "Rerun the experiment even if the config already appears in wandb."
)


def main(_):
    config = FLAGS.config

    if not FLAGS.rerun:
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
                return 0

    with wandb.init(
        mode=FLAGS.wandb_mode,
        tags=FLAGS.wandb_tags,
        notes=FLAGS.wandb_notes,
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_entity,
        name=FLAGS.wandb_name,
        config=config.to_dict(),
        settings=wandb.Settings(code_dir="../"),
    ) as run:
        rng = random.PRNGKey(config.seed)
        data_rng, init_rng, state_rng = random.split(rng, 3)

        train_ds, val_ds, _ = get_data(config, data_rng)

        model = VAE(**config.model.to_dict())

        variables = model.init(
            {"params": init_rng, "sample": init_rng},
            jnp.empty((28, 28, 1)),
            train=False,
        )

        parameter_overview.log_parameter_overview(variables)

        params = flax.core.freeze(variables["params"])
        del variables

        state = create_vae_state(params, state_rng, config)

        train_step, eval_step = make_vae_train_and_eval(model, config)
        x = next(deterministic_data.start_input_pipeline(val_ds))["image"][0]
        reconstruction_plot, sampling_plot = make_vae_plotting_fns(config, model, x)

        _, _, _ = ciclo.train_loop(
            state,
            deterministic_data.start_input_pipeline(train_ds),
            {
                ciclo.on_train_step: [train_step],
                ciclo.every(int(config.steps * config.plot_freq)): [
                    sampling_plot,
                    reconstruction_plot,
                ],
                ciclo.on_reset_step: reset_metrics,
                ciclo.on_test_step: [
                    eval_step,
                ],
                ciclo.every(1): custom_wandb_logger(run=run),
            },
            test_dataset=lambda: deterministic_data.start_input_pipeline(val_ds),
            epoch_duration=int(config.steps * config.eval_freq),
            callbacks=[
                ciclo.keras_bar(total=config.steps),
            ],
            stop=config.steps + 1,
        )


if __name__ == "__main__":
    jax_config.config_with_absl()
    app.run(main)
