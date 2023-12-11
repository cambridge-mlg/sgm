from itertools import product
from pathlib import Path

import ciclo
import flax
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint
import wandb
from absl import app, flags, logging
from clu import deterministic_data
from flax.training import orbax_utils
from jax.config import config as jax_config
from ml_collections import config_dict, config_flags

from experiments.utils import duplicated_run
from src.models.transformation_inference_model import (
    TransformationInferenceNet,
    create_transformation_inference_state,
    make_transformation_inference_train_and_eval,
)
from src.models.utils import reset_metrics
from src.transformations.affine import gen_affine_matrix_no_shear
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
flags.DEFINE_bool(
    "rerun", False, "Rerun the experiment even if the config already appears in wandb."
)


def main(_):
    config = FLAGS.config

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
            if isinstance(config.model.hidden_dims, str):
                config.model.hidden_dims = tuple(
                    int(x) for x in config.model.hidden_dims.split(",")
                )

        rng = random.PRNGKey(config.seed)
        data_rng, init_rng = random.split(rng)

        train_ds, val_ds, _ = get_data(config, data_rng)
        input_shape = train_ds.element_spec["image"].shape[2:]

        model = TransformationInferenceNet(
            bounds=config.get("augment_bounds", None),
            offset=config.get("augment_offset", None),
            **config.model.to_dict(),
        )

        state = create_transformation_inference_state(
            model, config, init_rng, input_shape=input_shape
        )

        train_step, eval_step = make_transformation_inference_train_and_eval(
            model, config
        )

        final_state, history, _ = ciclo.train_loop(
            state,
            deterministic_data.start_input_pipeline(train_ds),
            {
                ciclo.on_train_step: [train_step],
                ciclo.on_reset_step: reset_metrics,
                ciclo.on_test_step: eval_step,
                ciclo.every(1): custom_wandb_logger(run=run),
            },
            test_dataset=lambda: deterministic_data.start_input_pipeline(val_ds),
            epoch_duration=int(config.steps * config.eval_freq),
            callbacks=[
                ciclo.keras_bar(total=config.steps),
            ],
            stop=config.steps + 1,
        )

        # Save the checkpoint if a path is provided:
        model_checkpoint_path = config.get("checkpoint", "")
        if model_checkpoint_path != "":
            model_checkpoint_path = Path(model_checkpoint_path)
            if model_checkpoint_path.exists():
                model_checkpoint_path.rmtree()
            model_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            logging.info(f"Saving model checkpoint to {model_checkpoint_path}.")
            ckpt = {"state": final_state, "config": config}
            checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpointer.save(model_checkpoint_path, ckpt, save_args=save_args)

        fig = plot_proto_model_training_metrics(history)
        run.summary["inf_training_metrics"] = wandb.Image(fig)
        plt.close(fig)

        val_iter = deterministic_data.start_input_pipeline(val_ds)
        val_batch = next(val_iter)

        get_prototype = make_get_prototype_fn(
            model,
            final_state,
            rng,
            config.interpolation_order,
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


if __name__ == "__main__":
    jax_config.config_with_absl()
    app.run(main)
