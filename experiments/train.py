import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".45"

from absl import app
from absl import flags
from absl import logging
import jax.random as random
from jax.config import config
from ml_collections import config_flags

from src.utils.training import setup_model, get_dataset_splits, train_loop


logging.set_verbosity(logging.INFO)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config")
flags.mark_flag_as_required("config")
flags.DEFINE_enum("wandb_mode", "online", ["online", "offline", "disabled"], "Mode for wandb.run.")
flags.DEFINE_list("wandb_tags", [], "Tags for wandb.run.")
flags.DEFINE_string("wandb_notes", "", "Notes for wandb.run.")
flags.DEFINE_string("wandb_project", "neurips2023experiments", "Project for wandb.run.")
flags.DEFINE_string("wandb_entity", "invariance-learners", "Entity for wandb.run.")


def main(argv):
    # Set the seed
    rng = random.PRNGKey(FLAGS.config.seed)
    data_rng, model_rng = random.split(rng)

    # Get the data
    train_ds, val_ds, _ = get_dataset_splits(
        FLAGS.config, data_rng, FLAGS.config.batch_size, FLAGS.config.batch_size
    )

    # Get the model
    model, state = setup_model(FLAGS.config, model_rng, train_ds)

    # Train the model
    train_loop(
        FLAGS.config,
        model,
        state,
        train_ds,
        val_ds,
        wandb_kwargs={
            "mode": FLAGS.wandb_mode,
            "tags": FLAGS.wandb_tags,
            "notes": FLAGS.wandb_notes,
            "project": FLAGS.wandb_project,
            "entity": FLAGS.wandb_entity,
        },
    )


if __name__ == "__main__":
    config.config_with_absl()
    app.run(main)
