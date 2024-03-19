import ciclo
import flax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import wandb
from absl import app, flags, logging
from clu import deterministic_data
from jax.config import config as jax_config
from ml_collections import config_dict, config_flags

from experiments.utils import duplicated_run
from src.models.utils import reset_metrics
from src.models.vae_wsda import (
    VAE_WSDA,
    create_vae_wsda_state,
    make_vae_wsda_plotting_fns,
    make_vae_wsda_train_and_eval,
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
    config.model_name = "VAE_WSDA"

    if not FLAGS.rerun:
        if duplicated_run(config):
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
        config = config_dict.ConfigDict(wandb.config)

        with config.ignore_type():
            if isinstance(config.model.conv_dims, str):
                config.model.conv_dims = tuple(
                    int(x) for x in config.model.conv_dims.split(",")
                )
            if isinstance(config.model.dense_dims, str):
                config.model.dense_dims = tuple(
                    int(x) for x in config.model.dense_dims.split(",")
                )

        rng = random.PRNGKey(config.seed)
        data_rng, init_rng = random.split(rng)

        train_ds, val_ds, test_ds = get_data(config, data_rng)
        input_shape = train_ds.element_spec["image"].shape[2:]

        model = VAE_WSDA(vae=config.model.to_dict())

        state = create_vae_wsda_state(model, config, init_rng, input_shape)

        train_step, eval_step = make_vae_wsda_train_and_eval(model, config)
        x = next(deterministic_data.start_input_pipeline(val_ds))["image"][0]
        reconstruction_plot, sampling_plot = make_vae_wsda_plotting_fns(
            config, model, x
        )

        final_state, _, _ = ciclo.train_loop(
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

        # Run test set eval, which should include IWLB
        if test_ds is not None:
            config_test = config.copy_and_resolve_references()
            config_test.run_iwlb = True
            _, test_step = make_vae_wsda_train_and_eval(model, config_test)

            _, test_history, _ = ciclo.test_loop(
                final_state,
                deterministic_data.start_input_pipeline(test_ds),
                {
                    ciclo.on_reset_step: reset_metrics,
                    ciclo.on_test_step: [
                        test_step,
                    ],
                },
            )
            loss, elbo, ll, kld, iwlb, x_mse = test_history.collect(
                "loss", "elbo", "ll", "kld", "iwlb", "x_mse"
            )
            run.summary["test/loss"] = loss[-1]
            run.summary["test/elbo"] = elbo[-1]
            run.summary["test/ll"] = ll[-1]
            run.summary["test/kld"] = kld[-1]
            run.summary["test/iwlb"] = iwlb[-1]
            run.summary["test/x_mse"] = x_mse[-1]


if __name__ == "__main__":
    jax_config.config_with_absl()
    app.run(main)
