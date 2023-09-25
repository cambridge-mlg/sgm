import os

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax.config import config

jnp.log(jnp.exp(1) - 1.0)
# TODO: figure out why we get CUDA failures if this ^ isn't here.

import ciclo
import flax
import matplotlib.pyplot as plt
from absl import app, flags, logging
from clu import deterministic_data, parameter_overview
from ml_collections import config_flags

import wandb
from src.models.proto_gen_model import (
    PrototypicalGenerativeModel,
    create_pgm_state,
    make_pgm_train_and_eval,
)
from src.models.utils import reset_metrics
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
flags.DEFINE_string("wandb_project", "iclr2024experiments", "Project for wandb.run.")
flags.DEFINE_string("wandb_entity", "invariance-learners", "Entity for wandb.run.")
flags.DEFINE_string("wandb_name", None, "Name for wandb.run.")


def main(_):
    with wandb.init(
        mode=FLAGS.wandb_mode,
        tags=FLAGS.wandb_tags,
        notes=FLAGS.wandb_notes,
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_entity,
        name=FLAGS.config.name,
        settings=wandb.Settings(code_dir="../"),
    ) as run:
        rng = random.PRNGKey(FLAGS.config.seed)
        data_rng, init_rng, state_rng = random.split(rng, 3)

        train_ds, val_ds, _ = get_data(FLAGS.config, data_rng)

        model = PrototypicalGenerativeModel(**FLAGS.config.model.to_dict())

        variables = model.init(
            {"params": init_rng, "sample": init_rng},
            jnp.empty((28, 28, 1)),
            train=False,
        )

        parameter_overview.log_parameter_overview(variables)

        params = flax.core.freeze(variables["params"])
        del variables

        state = create_pgm_state(params, state_rng, FLAGS.config)

        train_step, eval_step = make_pgm_train_and_eval(FLAGS.config, model)

        total_steps = FLAGS.config.gen_steps + FLAGS.config.inf_steps
        _, _, _ = ciclo.train_loop(
            state,
            deterministic_data.start_input_pipeline(train_ds),
            {
                ciclo.on_train_step: [train_step],
                ciclo.on_reset_step: reset_metrics,
                ciclo.on_test_step: eval_step,
                ciclo.every(1): custom_wandb_logger(run=run),
            },
            test_dataset=lambda: deterministic_data.start_input_pipeline(val_ds),
            epoch_duration=int(total_steps * FLAGS.config.eval_freq),
            callbacks=[
                ciclo.keras_bar(total=total_steps),
            ],
            stop=total_steps + 1,
        )


if __name__ == "__main__":
    config.config_with_absl()
    app.run(main)
