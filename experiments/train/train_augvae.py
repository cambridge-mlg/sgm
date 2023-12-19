from itertools import product
from pathlib import Path

import ciclo
import flax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags, logging
from clu import deterministic_data
from jax.config import config as jax_config
from ml_collections import config_dict, config_flags

import wandb
from experiments.utils import (
    assert_inf_gen_compatiblity,
    duplicated_run,
    load_checkpoint,
    save_checkpoint,
)
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
config_flags.DEFINE_config_file("inf_config")
flags.mark_flag_as_required("inf_config")
config_flags.DEFINE_config_file("gen_config")
flags.mark_flag_as_required("gen_config")
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


def main(_):
    inf_config = FLAGS.inf_config
    gen_config = FLAGS.gen_config
    vae_config = FLAGS.vae_config
    vae_config.model_name = "AugVAE"

    assert_inf_gen_compatiblity(inf_config, gen_config)

    if not FLAGS.rerun:
        if (
            duplicated_run(inf_config)
            and duplicated_run(gen_config)
            and duplicated_run(vae_config)
        ):
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

        ####### INF MODEL #######
        rng = random.PRNGKey(inf_config.seed)
        data_rng, init_rng = random.split(rng)

        train_ds, val_ds, _ = get_data(inf_config, data_rng)
        input_shape = train_ds.element_spec["image"].shape[2:]

        inf_model = TransformationInferenceNet(
            bounds=inf_config.get("augment_bounds", None),
            offset=inf_config.get("augment_offset", None),
            **inf_config.model.to_dict(),
        )

        inf_state = create_transformation_inference_state(
            inf_model, inf_config, init_rng, input_shape
        )

        # Only do training if there is no inf model checkpoint to load:
        inf_model_checkpoint_path = inf_config.get("checkpoint", "")
        if (
            inf_model_checkpoint_path != ""
            and not Path(inf_model_checkpoint_path).exists()
        ):
            train_step, eval_step = make_transformation_inference_train_and_eval(
                inf_model, inf_config
            )

            inf_final_state, history, _ = ciclo.train_loop(
                inf_state,
                deterministic_data.start_input_pipeline(train_ds),
                {
                    ciclo.on_train_step: [train_step],
                    ciclo.on_reset_step: reset_metrics,
                    ciclo.on_test_step: eval_step,
                },
                test_dataset=lambda: deterministic_data.start_input_pipeline(val_ds),
                epoch_duration=int(inf_config.steps * inf_config.eval_freq),
                callbacks=[
                    ciclo.keras_bar(total=inf_config.steps),
                ],
                stop=inf_config.steps + 1,
            )

            save_checkpoint(inf_model_checkpoint_path, inf_final_state, inf_config)

            fig = plot_proto_model_training_metrics(history)
            run.summary["proto_training_metrics"] = wandb.Image(fig)
            plt.close(fig)
        else:
            inf_final_state, _ = load_checkpoint(
                inf_model_checkpoint_path, inf_state, inf_config
            )

        val_iter = deterministic_data.start_input_pipeline(val_ds)
        val_batch = next(val_iter)

        get_prototype = make_get_prototype_fn(
            inf_model,
            inf_final_state,
            rng,
            inf_config.interpolation_order,
            gen_affine_matrix_no_shear,
        )

        for i, (x_, mask) in enumerate(
            product(
                [
                    val_batch["image"][0][14],
                    val_batch["image"][0][12],
                    # val_batch["image"][0][1],
                    # val_batch["image"][0][4],
                    # val_batch["image"][0][9],
                ],
                [
                    # jnp.array([0, 0, 1, 0, 0]),
                    # jnp.array([1, 1, 0, 0, 0]),
                    jnp.array([1, 1, 1, 1, 1]),
                ],
            )
        ):
            fig = plot_protos_and_recons(
                x_, jnp.array(inf_config.augment_bounds) * mask, get_prototype
            )
            run.summary[f"inf_plots_{i}"] = wandb.Image(fig)
            plt.close(fig)

        ####### GEN MODEL #######
        def prototype_function(x, rng):
            η = inf_model.apply(
                {"params": inf_final_state.params}, x, train=False
            ).sample(seed=rng)
            return η

        rng = random.PRNGKey(gen_config.seed)
        data_rng, init_rng = random.split(rng)

        train_ds, val_ds, _ = get_data(gen_config, data_rng)
        input_shape = train_ds.element_spec["image"].shape[2:]

        gen_model = TransformationGenerativeNet(
            bounds=gen_config.get("augment_bounds", None),
            offset=gen_config.get("augment_offset", None),
            **gen_config.model.to_dict(),
        )

        gen_state = create_transformation_generative_state(
            gen_model,
            gen_config,
            init_rng,
            input_shape,
        )

        # Only do training if there is no gen model checkpoint to load:
        gen_model_checkpoint_path = gen_config.get("checkpoint", "")
        if (
            gen_model_checkpoint_path != ""
            and not Path(gen_model_checkpoint_path).exists()
        ):
            train_step, eval_step = make_transformation_generative_train_and_eval(
                gen_model, gen_config, prototype_function=prototype_function
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
                epoch_duration=int(gen_config.steps * gen_config.eval_freq),
                callbacks=[
                    ciclo.keras_bar(total=gen_config.steps),
                ],
                stop=gen_config.steps + 1,
            )

            save_checkpoint(gen_model_checkpoint_path, gen_final_state, gen_config)

            fig = plot_gen_model_training_metrics(history)
            run.summary[f"gen_training_metrics"] = wandb.Image(fig)
            plt.close(fig)
        else:
            gen_final_state, _ = load_checkpoint(
                gen_model_checkpoint_path, gen_state, gen_config
            )

        val_iter = deterministic_data.start_input_pipeline(val_ds)
        val_batch = next(val_iter)
        for i, x in enumerate(
            [
                val_batch["image"][0][14],
                val_batch["image"][0][12],
                # val_batch["image"][0][1],
                # val_batch["image"][0][4],
                # val_batch["image"][0][9],
            ]
        ):
            fig = plot_gen_dists(
                x,
                prototype_function,
                rng,
                gen_model,
                gen_final_state.params,
                gen_config,
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
            inference=inf_config.model.inference.to_dict(),
            generative=gen_config.model.generative.to_dict(),
            interpolation_order=gen_config.interpolation_order,
            bounds=gen_config.get("augment_bounds", None),
            offset=gen_config.get("augment_offset", None),
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
