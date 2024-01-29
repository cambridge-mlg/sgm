from itertools import product
from pathlib import Path

import ciclo
import flax
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import wandb
from absl import app, flags, logging
from clu import deterministic_data
from jax.config import config as jax_config
from ml_collections import config_dict, config_flags

from experiments.utils import (
    assert_inf_gen_compatiblity,
    duplicated_run,
    load_checkpoint,
    save_checkpoint,
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
from src.transformations import transforms
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

    assert_inf_gen_compatiblity(inf_config, gen_config)

    if not FLAGS.rerun:
        if duplicated_run(inf_config) and duplicated_run(gen_config):
            return 0

    with wandb.init(
        mode=FLAGS.wandb_mode,
        tags=FLAGS.wandb_tags,
        notes=FLAGS.wandb_notes,
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_entity,
        name=FLAGS.wandb_name,
        config=gen_config.to_dict(),
        settings=wandb.Settings(code_dir="../"),
    ) as run:
        gen_config = config_dict.ConfigDict(wandb.config)

        with gen_config.ignore_type():
            if isinstance(gen_config.model.hidden_dims, str):
                gen_config.model.hidden_dims = tuple(
                    int(x) for x in gen_config.model.hidden_dims.split(",")
                )
            if isinstance(gen_config.model.conditioner.hidden_dims, str):
                gen_config.model.conditioner.hidden_dims = tuple(
                    int(x) for x in gen_config.model.conditioner.hidden_dims.split(",")
                )
            if isinstance(gen_config.model.transform, str):
                gen_config.model.transform = getattr(
                    transforms, gen_config.model.transform.split(".")[-1]
                )

        ####### INF MODEL #######
        rng = random.PRNGKey(inf_config.seed)
        data_rng, init_rng = random.split(rng)

        train_ds, val_ds, _ = get_data(inf_config, data_rng)
        input_shape = train_ds.element_spec["image"].shape[2:]

        if gen_config.model.squash_to_bounds:
            inf_config.model.squash_to_bounds = True

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
        # if inf_model_checkpoint_path != "" and gen_config.model.squash_to_bounds:
        #     inf_model_checkpoint_path = inf_model_checkpoint_path + "_squashed"

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
            inf_final_state, inf_config_ = load_checkpoint(
                inf_model_checkpoint_path, inf_state, inf_config
            )
            if (
                gen_config.model.squash_to_bounds
                and not inf_config_.model.squash_to_bounds
            ):
                logging.warning(
                    "\n\n\n\nThe config loaded from the checkpoint does not have squash_to_bounds=True.\n"
                    "This is probably a mistake.\n\n\n\n"
                )

        val_iter = deterministic_data.start_input_pipeline(val_ds)
        val_batch = next(val_iter)

        get_prototype = make_get_prototype_fn(
            inf_model,
            inf_final_state,
            rng,
            inf_model.transform,
            inf_config.get("transform_kwargs", None),
        )

        for i, (x_, mask) in enumerate(
            product(
                [
                    val_batch["image"][0][14],
                    val_batch["image"][0][12],
                    val_batch["image"][0][7],
                    val_batch["image"][0][8],
                    val_batch["image"][0][9],
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
                x_,
                jnp.array(gen_config.augment_bounds[:5]) * mask,
                inf_model.transform,
                get_prototype,
                inf_config.get("transform_kwargs", None),
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
                ciclo.every(1): custom_wandb_logger(run=run),
            },
            test_dataset=lambda: deterministic_data.start_input_pipeline(val_ds),
            epoch_duration=int(gen_config.steps * gen_config.eval_freq),
            callbacks=[
                ciclo.keras_bar(total=gen_config.steps),
            ],
            stop=gen_config.steps + 1,
        )

        # Save the checkpoint if a path is provided:
        gen_model_checkpoint_path = gen_config.get("checkpoint", "")
        if gen_model_checkpoint_path != "":
            save_checkpoint(gen_model_checkpoint_path, gen_final_state, gen_config)

        fig = plot_gen_model_training_metrics(history)
        run.summary[f"gen_training_metrics"] = wandb.Image(fig)
        plt.close(fig)

        val_iter = deterministic_data.start_input_pipeline(val_ds)
        val_batch = next(val_iter)
        for i, x in enumerate(
            [
                val_batch["image"][0][14],
                val_batch["image"][0][12],
                val_batch["image"][0][7],
                val_batch["image"][0][8],
                val_batch["image"][0][9],
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

        transform_image = lambda x, η: gen_model.transform(η).apply(
            x, **(gen_config.get("transform_kwargs", None) or {})
        )
        transformed_xs = jax.vmap(transform_image, in_axes=(None, 0))(
            val_batch["image"][0][12],
            jnp.linspace(
                -jnp.array(gen_config.augment_bounds),
                jnp.array(gen_config.augment_bounds),
                3,
            ),
        )

        for i, xi in enumerate(transformed_xs):
            fig = plot_gen_dists(
                xi,
                prototype_function,
                rng,
                gen_model,
                gen_final_state.params,
                gen_config,
            )
            run.summary[f"gen_rep_plots_{i}"] = wandb.Image(fig)
            plt.close(fig)


if __name__ == "__main__":
    jax_config.config_with_absl()
    app.run(main)
