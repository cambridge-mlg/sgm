import os
from itertools import product

import ciclo
import flax
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import wandb
from absl import app, flags, logging
from clu import deterministic_data
from jax.config import config as jax_config
from ml_collections import config_dict, config_flags
from scipy.stats import gaussian_kde

from experiments.utils import duplicated_run
from src.models.aug_vae import (
    AUG_VAE,
    create_aug_vae_state,
    make_aug_vae_plotting_fns,
    make_aug_vae_train_and_eval,
)
from src.models.transformation_generative_model import (
    TransformationGenerativeNet,
    create_transformation_generative_state,
    make_transformation_generative_train_and_eval,
)
from src.models.transformation_inference_model import (
    TransformationInferenceNet,
    create_transformation_inference_state,
    make_transformation_inference_train_and_eval,
)
from src.models.utils import reset_metrics
from src.transformations.affine import gen_affine_matrix_no_shear
from src.utils.gen_plots import plot_gen_dists, plot_gen_model_training_metrics
from src.utils.input import get_data
from src.utils.proto_plots import (
    make_get_prototype_fn,
    plot_proto_model_training_metrics,
    plot_protos_and_recons,
)
from src.utils.training import custom_wandb_logger

flax.config.update("flax_use_orbax_checkpointing", True)
logging.set_verbosity(logging.INFO)
plt.rcParams["savefig.facecolor"] = "white"

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("pgm_config")
flags.mark_flag_as_required("pgm_config")
config_flags.DEFINE_config_file("vae_config")
flags.mark_flag_as_required("vae_config")
flags.DEFINE_enum(
    "wandb_mode", "online", ["online", "offline", "disabled"], "Mode for wandb.run."
)
flags.DEFINE_list("wandb_tags", [], "Tags for wandb.run.")
flags.DEFINE_string("wandb_notes", "", "Notes for wandb.run.")
flags.DEFINE_string("wandb_project", "iclr2024experiments", "Project for wandb.run.")
flags.DEFINE_string("wandb_entity", "invariance-learners", "Entity for wandb.run.")
flags.DEFINE_string("wandb_name", None, "Name for wandb.run.")
flags.DEFINE_bool(
    "rerun", False, "Rerun the experiment even if the config already appears in wandb."
)


# TODO: update since we don't use a shared pgm config any more
def main(_):
    pgm_config = FLAGS.pgm_config
    vae_config = FLAGS.vae_config
    vae_config.model_name = "AugVAE"

    if not FLAGS.rerun:
        if duplicated_run(pgm_config) and duplicated_run(vae_config):
            return 0

    with wandb.init(
        mode=FLAGS.wandb_mode,
        tags=FLAGS.wandb_tags,
        notes=FLAGS.wandb_notes,
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_entity,
        name=FLAGS.wandb_name,
        config=vae_config.to_dict(),
        settings=wandb.Settings(code_dir="../"),
    ) as run:
        vae_config = config_dict.ConfigDict(wandb.config)

        with vae_config.ignore_type():
            if isinstance(vae_config.model.conv_dims, str):
                vae_config.model.conv_dims = tuple(
                    int(x) for x in vae_config.model.conv_dims.split(",")
                )
            if isinstance(vae_config.model.dense_dims, str):
                vae_config.model.dense_dims = tuple(
                    int(x) for x in vae_config.model.dense_dims.split(",")
                )

        rng = random.PRNGKey(pgm_config.seed)
        data_rng, init_rng = random.split(rng)

        train_ds, val_ds, _ = get_data(pgm_config, data_rng)
        input_shape = train_ds.element_spec["image"].shape[2:]

        inf_model = TransformationInferenceNet(
            bounds=pgm_config.get("augment_bounds", None),
            offset=pgm_config.get("augment_offset", None),
            **pgm_config.model.inference.to_dict(),
        )

        inf_state = create_transformation_inference_state(
            inf_model, pgm_config, init_rng, input_shape
        )

        train_step, eval_step = make_transformation_inference_train_and_eval(
            inf_model, pgm_config
        )

        total_steps = pgm_config.inf_steps
        inf_final_state, history, _ = ciclo.train_loop(
            inf_state,
            deterministic_data.start_input_pipeline(train_ds),
            {
                ciclo.on_train_step: [train_step],
                ciclo.on_reset_step: reset_metrics,
                ciclo.on_test_step: eval_step,
            },
            test_dataset=lambda: deterministic_data.start_input_pipeline(val_ds),
            epoch_duration=int(total_steps * pgm_config.eval_freq),
            callbacks=[
                ciclo.keras_bar(total=total_steps),
            ],
            stop=total_steps + 1,
        )

        fig = plot_proto_model_training_metrics(history)
        run.summary["proto_training_metrics"] = wandb.Image(fig)
        plt.close(fig)

        val_iter = deterministic_data.start_input_pipeline(val_ds)
        val_batch = next(val_iter)

        get_prototype = make_get_prototype_fn(
            inf_model,
            inf_final_state,
            rng,
            pgm_config.interpolation_order,
            gen_affine_matrix_no_shear,
        )

        for i, (x_, mask) in enumerate(
            product(
                [
                    val_batch["image"][0][14],
                    val_batch["image"][0][12],
                ],
                [
                    jnp.array([0, 0, 1, 0, 0]),
                    jnp.array([1, 1, 0, 0, 0]),
                    jnp.array([1, 1, 1, 1, 1]),
                ],
            )
        ):
            fig = plot_protos_and_recons(
                x_, jnp.array(config.augment_bounds[:5]) * mask, get_prototype
            )
            run.summary[f"inf_plots_{i}"] = wandb.Image(fig)
            plt.close(fig)

        ####### GEN MODEL
        def prototype_function(x, rng):
            η = inf_model.apply(
                {"params": inf_final_state.params}, x, train=False
            ).sample(seed=rng)
            return η

        rng = random.PRNGKey(pgm_config.seed)
        data_rng, init_rng = random.split(rng)

        train_ds, val_ds, _ = get_data(pgm_config, data_rng)
        input_shape = train_ds.element_spec["image"].shape[2:]

        gen_model = TransformationGenerativeNet(
            bounds=pgm_config.get("augment_bounds", None),
            offset=pgm_config.get("augment_offset", None),
            **pgm_config.model.generative.to_dict(),
        )

        gen_state = create_transformation_generative_state(
            gen_model,
            pgm_config,
            init_rng,
            input_shape,
        )

        train_step, eval_step = make_transformation_generative_train_and_eval(
            gen_model, pgm_config, prototype_function=prototype_function
        )

        gen_final_state, history, _ = ciclo.train_loop(
            gen_state,
            deterministic_data.start_input_pipeline(train_ds),
            {
                ciclo.on_train_step: [train_step],
                ciclo.on_reset_step: reset_metrics,
                ciclo.on_test_step: eval_step,
            },
            test_dataset=lambda: deterministic_data.start_input_pipeline(val_ds),
            epoch_duration=int(pgm_config.gen_steps * pgm_config.eval_freq),
            callbacks=[
                ciclo.keras_bar(total=pgm_config.gen_steps),
            ],
            stop=pgm_config.gen_steps + 1,
        )

        fig = plot_gen_model_training_metrics(history)
        run.summary[f"gen_training_metrics"] = wandb.Image(fig)

        plt.close(fig)

        val_iter = deterministic_data.start_input_pipeline(val_ds)
        val_batch = next(val_iter)
        for i, x in enumerate(
            [
                val_batch["image"][0][14],
                val_batch["image"][0][1],
                val_batch["image"][0][4],
                val_batch["image"][0][9],
            ]
        ):
            plot_gen_dists(
                x,
                prototype_function,
                rng,
                gen_model,
                gen_final_state.params,
                pgm_config,
            )
            run.summary[f"gen_plots_{i}"] = wandb.Image(fig)
            plt.close(fig)

        ####### VAE MODEL
        rng = random.PRNGKey(vae_config.seed)
        data_rng, init_rng = random.split(rng, 2)

        train_ds, val_ds, _ = get_data(vae_config, data_rng)
        input_shape = train_ds.element_spec["image"].shape[2:]

        aug_vae_model = AUG_VAE(
            vae=vae_config.model.to_dict(),
            inference=pgm_config.model.inference.to_dict(),
            generative=pgm_config.model.generative.to_dict(),
            interpolation_order=pgm_config.interpolation_order,
            bounds=pgm_config.get("augment_bounds", None),
            offset=pgm_config.get("augment_offset", None),
        )

        aug_vae_state = create_aug_vae_state(
            aug_vae_model,
            vae_config,
            init_rng,
            input_shape,
            inf_final_state,
            gen_final_state,
        )

        train_step, eval_step = make_aug_vae_train_and_eval(aug_vae_model, vae_config)
        x = next(deterministic_data.start_input_pipeline(val_ds))["image"][0]
        reconstruction_plot, sampling_plot = make_aug_vae_plotting_fns(
            vae_config, aug_vae_model, x
        )

        final_aug_vae_state, history, _ = ciclo.train_loop(
            aug_vae_state,
            deterministic_data.start_input_pipeline(train_ds),
            {
                ciclo.on_train_step: [train_step],
                ciclo.every(int(vae_config.steps * vae_config.plot_freq)): [
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
            epoch_duration=int(vae_config.steps * vae_config.eval_freq),
            callbacks=[
                ciclo.keras_bar(total=vae_config.steps),
            ],
            stop=vae_config.steps + 1,
        )


if __name__ == "__main__":
    jax_config.config_with_absl()
    app.run(main)