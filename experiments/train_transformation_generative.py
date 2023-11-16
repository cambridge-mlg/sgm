import os
from pathlib import Path

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"


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
from src.models.transformation_generative_model import (
    TransformationGenerativeNet,
    create_transformation_generative_state,
    make_transformation_generative_train_and_eval,
)
from src.models.transformation_inference_model import TransformationInferenceNet

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
from src.utils.transform_generative_plots import (
    plot_generative_histograms,
    plot_transform_gen_model_training_metrics,
)

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

flags.DEFINE_string(
    "prototype_model_dir",
    "",
    "Path to output directory for the transformation inferencem model run to use.",
)


def main_with_wandb(_):
    config = FLAGS.config
    # Update with prototype_model_dir
    config = config.unlock()
    config.prototype_model_dir = FLAGS.prototype_model_dir
    config = config.lock()

    with wandb.init(
        mode=FLAGS.wandb_mode,
        tags=FLAGS.wandb_tags,
        notes=FLAGS.wandb_notes,
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_entity,
        config=config.to_dict(),
        save_code=FLAGS.wandb_save_code,
    ) as run:
        # Reload the config from wandb (in case using a sweep):
        config.update(wandb.config)
        # Run main:
        main(config, run, config.prototype_model_dir)


def main(config, run, prototype_model_dir: str):
    # --- Make directories for saving ckeckpoints/logs ---
    output_dir = config.get(
        "output_dir",
        get_and_make_datebased_output_directory(),
    )
    logging.info(f"Saving to:\n{output_dir}")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # - Set up rng
    rng = random.PRNGKey(config.seed)
    data_rng, init_rng, state_rng = random.split(rng, 3)

    # --- Load the prototype model ---
    handlers = {
        "state": Checkpointer(PyTreeCheckpointHandler()),
        "config": Checkpointer(JsonCheckpointHandler()),
    }
    manager = CheckpointManager(
        Path(prototype_model_dir) / "checkpoints", checkpointers=handlers
    )
    last_prototype_step = manager.latest_step()
    proto_ckpt_restored = manager.restore(last_prototype_step)

    proto_config = proto_ckpt_restored["config"]
    proto_state = proto_ckpt_restored["state"]

    proto_model = TransformationInferenceNet(**proto_config["model"]["inference"])

    def prototype_function(x, rng):
        η = proto_model.apply({"params": proto_state["params"]}, x, train=False).sample(
            seed=rng
        )
        return η

    # - Possibly Visualise the DSPRITES data augmentation distribution
    if config.dataset == "aug_dsprites":
        fig, _ = visualise_latent_distribution_from_config(config.aug_dsprites)
        run.log({"dsprites_augment_distribution": wandb.Image(fig)})
        plt.close(fig)

    # --- Data ---
    logging.info("Constructing the dataset")
    train_ds, val_ds, _ = get_data(config, data_rng)
    logging.info("Finished constructing the dataset")
    # --- Network setup ---
    gen_model = TransformationGenerativeNet(**config.model.generative.to_dict())

    gen_init_rng, init_rng = random.split(init_rng)
    variables = gen_model.init(
        {"params": gen_init_rng, "sample": gen_init_rng},
        jnp.empty((64, 64, 1))
        if "dsprites" in config.dataset
        else jnp.empty((28, 28, 1)), 
        η=jnp.empty((5,)),
        train=False,
    )

    parameter_overview.log_parameter_overview(variables)

    gen_params = flax.core.freeze(variables["params"])

    gen_state = create_transformation_generative_state(gen_params, state_rng, config)

    train_step, eval_step = make_transformation_generative_train_and_eval(
        config, gen_model, prototype_function=prototype_function
    )

    # --- Training ---
    total_steps = config.gen_steps
    gen_final_state, history, _ = ciclo.train_loop(
        gen_state,
        deterministic_data.start_input_pipeline(train_ds),
        {
            ciclo.on_train_step: [train_step],
            ciclo.on_reset_step: reset_metrics,
            ciclo.on_test_step: eval_step,
            ciclo.every(1): custom_wandb_logger(run=run),  # type:ignore
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
        gen_final_state.step,
        {"state": gen_final_state, "config": config.to_dict()},
    )

    # --- Log final metrics as an image: ---
    fig = plot_transform_gen_model_training_metrics(history)
    run.log({"run_metrics_summary": wandb.Image(fig)})
    fig.savefig(output_dir / "run_metrics_summary.pdf", bbox_inches="tight")

    # --- Log generative histograms ---
    nrows = 5
    val_iter = deterministic_data.start_input_pipeline(val_ds)
    val_batch = next(val_iter)
    val_histograms_fig = plt.figure(figsize=(14, 3 * nrows))
    val_histograms_subfigs = val_histograms_fig.subfigures(nrows=nrows)
    for i in range(nrows):
        plot_generative_histograms(
            x=val_batch["image"][0, i],
            rng=rng,
            prototype_function=prototype_function,
            interpolation_order=config.interpolation_order,
            transform_gen_distribution_function=lambda xhat: gen_model.apply(
                {"params": gen_final_state.params}, xhat, train=False
            )[0],
            fig=val_histograms_subfigs[i],
        )
    wandb.log({"val_histograms": wandb.Image(val_histograms_fig)})
    # - Same one, but with a transformed digit:
    trans_histograms_fig = plt.figure(figsize=(14, 3 * nrows))
    trans_histograms_subfigs = trans_histograms_fig.subfigures(nrows=nrows)
    transformed_xs = jax.vmap(transform_image, in_axes=(None, 0))(
        val_batch["image"][0, 0],
        jnp.concatenate(
            (
                jnp.linspace(
                    -jnp.array(config.eval_augment_bounds),
                    jnp.array(config.eval_augment_bounds),
                    nrows,
                ),
            )
        ),
    )
    for i in range(nrows):
        plot_generative_histograms(
            x=transformed_xs[i],
            rng=rng,
            prototype_function=prototype_function,
            interpolation_order=config.interpolation_order,
            transform_gen_distribution_function=lambda xhat: gen_model.apply(
                {"params": gen_final_state.params}, xhat, train=False
            )[0],
            fig=trans_histograms_subfigs[i],
        )
    wandb.log({"trans_histograms": wandb.Image(val_histograms_fig)})


if __name__ == "__main__":
    config.config_with_absl()
    app.run(main_with_wandb)
