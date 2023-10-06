import os


# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"


import ciclo
import flax
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from absl import app, flags, logging
from clu import deterministic_data, parameter_overview
from jax.config import config
from ml_collections import config_flags
from orbax.checkpoint import (
    Checkpointer,
    CheckpointManager,
    JsonCheckpointHandler,
    PyTreeCheckpointHandler,
)

import wandb
from src.models.proto_gen_model_separated import (
    TransformationInferenceNet,
    create_canonicalizer_state,
    make_canonicalizer_train_and_eval,
)
from src.models.utils import reset_metrics
from src.transformations import transform_image
from src.utils.datasets.augmented_dsprites import (
    plot_prototypes_by_shape,
    visualise_latent_distribution_from_config,
)
from src.utils.input import get_data
from src.utils.logging import get_and_make_datebased_output_directory
from src.utils.proto_plots import (
    construct_plot_augmented_data_samples_canonicalizations,
    construct_plot_data_samples_canonicalizations,
    construct_plot_training_augmented_samples,
    plot_proto_model_training_metrics,
    plot_training_samples,
)
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
flags.DEFINE_bool("wandb_save_code", default=True, help="Save code to wandb.")


def main(_):
    with wandb.init(
        mode=FLAGS.wandb_mode,
        tags=FLAGS.wandb_tags,
        notes=FLAGS.wandb_notes,
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_entity,
        save_code=FLAGS.wandb_save_code,
        group="train-canonicalizer",
    ) as run:
        config = FLAGS.config
        # --- Make directories for saving ckeckpoints/logs ---
        output_dir = config.get(
            "output_dir",
            get_and_make_datebased_output_directory(),
        )
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Log the config:
        run.config.update(config.to_dict())
        rng = random.PRNGKey(config.seed)
        data_rng, init_rng, state_rng = random.split(rng, 3)

        if config.dataset == "aug_dsprites":
            fig, _ = visualise_latent_distribution_from_config(config.aug_dsprites)
            run.log({"dsprites_augment_distribution": wandb.Image(fig)})
            plt.close(fig)

        # --- Data ---
        logging.info("Constructing the dataset")
        train_ds, val_ds, _ = get_data(config, data_rng)
        logging.info("Finished constructing the dataset")
        # --- Network setup ---
        canon_model = TransformationInferenceNet(**config.model.inference.to_dict())

        canon_init_rng, init_rng = random.split(init_rng)

        logging.info("Initialise the model")
        variables = canon_model.init(
            {"params": canon_init_rng, "sample": canon_init_rng},
            jnp.empty((64, 64, 1))
            if "dsprites" in config.dataset
            else jnp.empty((28, 28, 1)),
            train=False,
        )

        parameter_overview.log_parameter_overview(variables)

        canon_params = flax.core.freeze(variables["params"])

        canon_state_rng, state_rng = random.split(state_rng)
        canon_state = create_canonicalizer_state(canon_params, canon_state_rng, config)

        train_step_canon, eval_step_canon = make_canonicalizer_train_and_eval(
            config, canon_model
        )

        # --- Logging visualisations ---: # TODO move this to src/
        @jax.jit
        def get_prototype(x, rng, params):
            p_η = canon_model.apply({"params": params}, x, train=False)
            η = p_η.sample(seed=rng)
            xhat = transform_image(x, -η, order=config.interpolation_order)
            return xhat

        plot_data_samples_canonicalizations = (
            construct_plot_data_samples_canonicalizations(
                get_prototype_fn=get_prototype
            )
        )

        def plot_and_log_data_samples_canonicalizations(state, batch):
            fig = plot_data_samples_canonicalizations(state, batch)
            wandb.log({"canonicalizations": wandb.Image(fig)}, step=state.step)
            plt.close(fig)

        plot_augmented_data_samples_canonicalizations = (
            construct_plot_augmented_data_samples_canonicalizations(
                get_prototype_fn=get_prototype, config=config
            )
        )

        def plot_and_log_data_augmented_samples_canonicalizations(state, batch):
            fig = plot_augmented_data_samples_canonicalizations(state, batch)
            wandb.log(
                {"augmented_canonicalizations": wandb.Image(fig)}, step=state.step
            )
            plt.close(fig)

        plot_training_augmented_samples = construct_plot_training_augmented_samples(
            config=config
        )

        def plot_and_log_training_augmented_samples(state, batch):
            fig = plot_training_augmented_samples(state, batch)
            wandb.log({"training_samples_augmented": wandb.Image(fig)}, step=state.step)
            plt.close(fig)

        # --- Training ---
        total_steps = config.inf_steps
        canon_final_state, history, _ = ciclo.train_loop(
            canon_state,
            deterministic_data.start_input_pipeline(train_ds),
            {
                ciclo.on_train_step: [train_step_canon],
                ciclo.on_reset_step: reset_metrics,
                ciclo.on_test_step: eval_step_canon,
                ciclo.every(1): custom_wandb_logger(run=run),
                ciclo.every(int(total_steps * config.eval_freq)): [
                    plot_and_log_data_samples_canonicalizations,
                    plot_and_log_data_augmented_samples_canonicalizations,
                    plot_and_log_training_augmented_samples,
                    plot_training_samples,
                ],
            },
            test_dataset=lambda: deterministic_data.start_input_pipeline(val_ds),
            epoch_duration=int(total_steps * config.eval_freq),
            callbacks=[
                ciclo.keras_bar(total=total_steps),
                # ciclo.early_stopping("loss_test", patience=total_steps // 10, min_delta=1e-4, mode="min", restore_best_weights=True),
                # ciclo.checkpoint("checkpoint", monitor="loss_test", mode="min", overwrite=True),
            ],
            stop=total_steps + 1,
        )
        # --- Save the model ---
        handlers = {
            "state": Checkpointer(PyTreeCheckpointHandler()),
            "config": Checkpointer(JsonCheckpointHandler()),
        }
        manager = CheckpointManager(checkpoint_dir, checkpointers=handlers)
        manager.save(
            canon_final_state.step,
            {"state": canon_final_state, "config": config.to_dict()},
        )

        # --- Log reconstructions for eval: ---
        def canon_function(x, rng):
            η = canon_model.apply(
                {"params": canon_final_state.params}, x, train=False
            ).sample(seed=rng)
            return η

        if config.dataset == "aug_dsprites":
            it = deterministic_data.start_input_pipeline(val_ds)
            batch = next(it)

            fig = plot_prototypes_by_shape(canon_function=canon_function, batch=batch)
            run.log({"dsprites_by_elem_final_prototypes": wandb.Image(fig)})

        # --- Log final metrics as an image: ---
        fig = plot_proto_model_training_metrics(history)
        run.log({"run_metrics_summary": wandb.Image(fig)})
        fig.savefig(output_dir / "run_metrics_summary.pdf", bbox_inches="tight")


if __name__ == "__main__":
    config.config_with_absl()
    app.run(main)
