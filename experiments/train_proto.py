import functools
import math
import os
import distrax

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"


import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax.config import config

import ciclo
import flax
import jax
import matplotlib.pyplot as plt
import seaborn as sns
from absl import app, flags, logging
from clu import deterministic_data, parameter_overview
from flax.training import checkpoints
from ml_collections import config_flags

import wandb
from src.utils.input import get_data

# from src.models.proto_gen_model import PrototypicalGenerativeModel, create_pgm_state, make_pgm_train_and_eval
from src.models.proto_gen_model_separated import (
    TransformationInferenceNet,
    make_canonicalizer_train_and_eval,
    create_canonicalizer_state,
)
from src.utils.datasets.augmented_dsprites import (
    visualise_latent_distribution_from_config,
)
from src.models.utils import reset_metrics
from src.transformations import transform_image
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
            jnp.empty((64, 64, 1)) if "dsprites" in config.dataset else jnp.empty((28, 28, 1)),
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

        def plot_data_samples_canonicalizations(state, batch):
            n_images = 56

            @jax.jit
            def get_images_and_prototypes(state, batch):
                step_rng = random.fold_in(state.rng, state.step)

                # Group the images by labels:
                images = batch["image"][0, :n_images]
                sort_idxs = jnp.argsort(batch["label"][0, :n_images])
                images_to_plot = images[sort_idxs]

                # Get the canonicalizations:

                prototypes = jax.vmap(get_prototype, in_axes=(0, 0, None))(
                    images_to_plot,
                    random.split(step_rng, len(images_to_plot)),
                    state.params,
                )
                return images_to_plot, prototypes

            images_to_plot, prototypes = get_images_and_prototypes(state, batch)
            images_to_plot, prototypes = map(
                lambda z: np.array(z), (images_to_plot, prototypes)
            )

            vmin = min(prototypes.min(), images_to_plot.min())
            vmax = max(prototypes.max(), images_to_plot.max())
            nrows = math.ceil(math.sqrt(n_images))
            ncols = math.ceil(n_images / nrows) * 2
            imshow_kwargs = dict(cmap="gray", vmin=vmin, vmax=vmax)
            fig, axes = plt.subplots(
                ncols=ncols, nrows=nrows, figsize=(ncols * 1.5, nrows * 1.5)
            )
            for ax in axes.ravel():
                ax.axis("off")
            for j in range(ncols // 2):
                for i in range(nrows):
                    idx = i * (ncols // 2) + j
                    if idx >= n_images:
                        break
                    else:
                        axes[i, 2 * j].imshow(images_to_plot[idx], **imshow_kwargs)
                        axes[i, 2 * j + 1].imshow(prototypes[idx], **imshow_kwargs)
                axes[0, 2 * j].set_title("Original")
                axes[0, 2 * j + 1].set_title("Canonicalized")
            return fig

        def plot_and_log_data_samples_canonicalizations(state, batch):
            fig = plot_data_samples_canonicalizations(state, batch)
            wandb.log({"canonicalizations": wandb.Image(fig)}, step=state.step)
            plt.close(fig)

        @jax.jit
        def get_aug_image(image, img_rng, augment_bounds_mult):
            Η_rand = distrax.Uniform(
                # Separate model bounds and augment bounds
                low=-jnp.array(config.augment_bounds) * augment_bounds_mult
                + jnp.array(config.augment_offset),
                high=jnp.array(config.augment_bounds) * augment_bounds_mult
                + jnp.array(config.augment_offset),
            )
            η_rand = Η_rand.sample(seed=img_rng, sample_shape=())

            x_rand = transform_image(image, η_rand, order=config.interpolation_order)
            return x_rand

        def plot_augmented_data_samples_canonicalizations(state, batch):
            n_images = 7
            n_samples = 8

            @jax.jit
            def get_augmented_images_and_prototypes(state, batch):
                step_rng = random.fold_in(state.rng, state.step)

                # Group the images by labels:
                images = batch["image"][0, :n_images]  # Shape (n_images, 64, 64, 1)
                sort_idxs = jnp.argsort(batch["label"][0, :n_images])
                base_images_to_plot = images[sort_idxs]

                images_to_plot = jax.vmap(
                    lambda rng: jax.vmap(
                        get_aug_image,
                        in_axes=(0, 0, None),
                    )(
                        base_images_to_plot,
                        random.split(rng, n_images),
                        state.augment_bounds_mult,
                    ),
                    in_axes=(0),
                )(
                    random.split(step_rng, n_samples)
                )  # Shape (n_samples, n_images, 64, 64, 1)
                # Get the canonicalizations:
                prototypes = jax.vmap(
                    lambda im, rng, params: jax.vmap(
                        get_prototype,
                        in_axes=(0, 0, None),
                    )(im, random.split(rng, n_images), params),
                    in_axes=(0, 0, None),
                )(
                    images_to_plot,
                    random.split(step_rng, len(images_to_plot)),
                    state.params,
                )
                return images_to_plot, prototypes

            images_to_plot, prototypes = get_augmented_images_and_prototypes(
                state, batch
            )
            images_to_plot, prototypes = map(
                lambda z: np.array(z), (images_to_plot, prototypes)
            )

            vmin = min(prototypes.min(), images_to_plot.min())
            vmax = max(prototypes.max(), images_to_plot.max())
            nrows = n_samples
            ncols = n_images * 2
            imshow_kwargs = dict(cmap="gray", vmin=vmin, vmax=vmax)
            fig = plt.figure(figsize=(ncols * 1.7, nrows * 1.5))
            subfigs = fig.subfigures(nrows=1, ncols=ncols, wspace=0.05)
            for j in range(n_images):
                subfig = subfigs[j]
                subfig.suptitle(f"Image {j}")
                axes = subfig.subplots(nrows=n_samples, ncols=2)
                for i in range(n_samples):
                    axes[i, 0].imshow(images_to_plot[i, j], **imshow_kwargs)
                    axes[i, 1].imshow(prototypes[i, j], **imshow_kwargs)
                axes[0, 0].set_title("Original", fontsize=9)
                axes[0, 1].set_title("Canon.", fontsize=9)
                for ax in axes.ravel():
                    ax.axis("off")
            return fig

        def plot_and_log_data_augmented_samples_canonicalizations(state, batch):
            fig = plot_augmented_data_samples_canonicalizations(state, batch)
            wandb.log(
                {"augmented_canonicalizations": wandb.Image(fig)}, step=state.step
            )
            plt.close(fig)

        def plot_training_augment_samples(state, batch):
            n_images = 20

            @jax.jit
            def get_image_original_and_augment(state, batch):
                step_rng = random.fold_in(state.rng, state.step)

                # Group the images by labels:
                images = batch["image"][0, :n_images]

                images_augmented = jax.vmap(get_aug_image, in_axes=(0, 0, None))(
                    images, random.split(step_rng, n_images), state.augment_bounds_mult
                )
                return images, images_augmented

            images, images_augmented = get_image_original_and_augment(state, batch)
            images, images_augmented = map(
                lambda z: np.array(z), (images, images_augmented)
            )

            vmin = min(images.min(), images_augmented.min())
            vmax = max(images.max(), images_augmented.max())
            nrows = math.ceil(math.sqrt(n_images))
            ncols = math.ceil(n_images / nrows) * 2
            imshow_kwargs = dict(cmap="gray", vmin=vmin, vmax=vmax)
            fig, axes = plt.subplots(
                ncols=ncols, nrows=nrows, figsize=(ncols * 1.5, nrows * 1.5)
            )
            for ax in axes.ravel():
                ax.axis("off")
            for j in range(ncols // 2):
                for i in range(nrows):
                    idx = i * (ncols // 2) + j
                    if idx >= n_images:
                        break
                    else:
                        axes[i, 2 * j].imshow(images[idx], **imshow_kwargs)
                        axes[i, 2 * j + 1].imshow(
                            images_augmented[idx], **imshow_kwargs
                        )
                axes[0, 2 * j].set_title("Original")
                axes[0, 2 * j + 1].set_title("Train. augmented")
            return fig

        def plot_and_log_training_augment_samples(state, batch):
            fig = plot_training_augment_samples(state, batch)
            wandb.log({"training_samples_augmented": wandb.Image(fig)}, step=state.step)
            plt.close(fig)

        def plot_training_samples(state, batch):
            n_images = batch["image"].shape[1]
            images_np = np.array(batch["image"][0])
            nrows = math.ceil(math.sqrt(n_images))
            ncols = math.ceil(n_images / nrows)
            fig, axes = plt.subplots(
                ncols=ncols, nrows=nrows, figsize=(ncols * 1.5, nrows * 1.5)
            )
            for i in range(nrows):
                for j in range(ncols):
                    axes[i, j].axis("off")
                    if i * ncols + j < n_images:
                        axes[i, j].imshow(images_np[i * ncols + j], cmap="gray")
            fig.tight_layout(pad=0.0)
            wandb.log({"training_samples": wandb.Image(fig)}, step=state.step)
            plt.close(fig)

        # --- Training ---

        total_steps = config.inf_steps
        # total_steps = config.inf_steps
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
                    plot_and_log_training_augment_samples,
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

        # --- Log reconstructions for eval: ---
        if config.dataset == "aug_dsprites":

            def get_proto(x):
                p_η = canon_model.apply(
                    {"params": canon_final_state.params}, x, train=False
                )
                η = p_η.sample(seed=rng)
                xhat = transform_image(x, -η, order=3)
                return xhat, η

            it = deterministic_data.start_input_pipeline(val_ds)
            batch = next(it)
            # print(batch["image"].shape)
            square_images = batch["image"][batch["label"] == 0]
            square_prototypes, _ = jax.vmap(get_proto)(square_images)
            ellipse_images = batch["image"][batch["label"] == 1]
            ellipse_prototypes, _ = jax.vmap(get_proto)(ellipse_images)
            heart_images = batch["image"][batch["label"] == 2]
            heart_prototypes, _ = jax.vmap(get_proto)(heart_images)

            vmin = min(
                map(
                    lambda ims: ims.min() if len(ims) else np.inf,
                    [
                        square_images,
                        ellipse_images,
                        heart_images,
                        square_prototypes,
                        ellipse_prototypes,
                        heart_prototypes,
                    ],
                )
            )
            vmax = max(
                map(
                    lambda ims: ims.max() if len(ims) else -np.inf,
                    [
                        square_images,
                        ellipse_images,
                        heart_images,
                        square_prototypes,
                        ellipse_prototypes,
                        heart_prototypes,
                    ],
                )
            )
            ncols = 10
            imshow_kwargs = dict(cmap="gray", vmin=vmin, vmax=vmax)
            fig, axes = plt.subplots(
                ncols=ncols, nrows=6, figsize=(ncols * 1.5, 1.5 * 6)
            )
            for ax in axes.ravel():
                ax.axis("off")
            for i in range(ncols):
                for j, shape_images, shape_prototypes in zip(
                    range(3),
                    [square_images, ellipse_images, heart_images],
                    [square_prototypes, ellipse_prototypes, heart_prototypes],
                ):
                    if i >= len(shape_images):
                        continue
                    axes[2 * j, i].imshow(shape_images[i], **imshow_kwargs)
                    axes[2 * j, i].set_title(
                        f"mse:{((shape_images[i] - shape_prototypes[i])**2).mean():.4f}",
                        fontsize=9,
                    )
                    axes[2 * j + 1, i].imshow(shape_prototypes[i], **imshow_kwargs)
                    axes[2 * j + 1, i].set_title(
                        "$ _{mse\_proto}$"
                        + f":{((shape_prototypes[0] - shape_prototypes[i])**2).mean():.4f}",
                        fontsize=9,
                    )

            axes[0, 0].set_ylabel("Original Square")
            axes[1, 0].set_ylabel("Proto Square")
            axes[2, 0].set_ylabel("Original Ellipse")
            axes[3, 0].set_ylabel("Proto Ellipse")
            axes[4, 0].set_ylabel("Original Heart")
            axes[5, 0].set_ylabel("Proto Heart")

            run.log({"dsprites_by_elem_final_prototypes": wandb.Image(fig)})

        # --- Log final metrics as an image: ---
        with sns.axes_style("whitegrid"):
            # plot the training history
            steps, loss, x_mse, lr_inf, lr_σ = history.collect(
                "steps",
                "loss",
                "x_mse",
                "lr_inf",
                "lr_σ",
            )
            sigma = history.collect("σ")
            augment_bounds_mult = history.collect("augment_bounds_mult")
            steps_test, loss_test, x_mse_test = history.collect(
                "steps", "loss_test", "x_mse_test"
            )

            label_paired_image_mse = history.collect("label_paired_image_mse_test")

            n_plots = 5
            fig, axs = plt.subplots(
                n_plots, 1, figsize=(15, n_plots * 3.0), dpi=300, sharex=True
            )

            axs[0].plot(steps, loss, label=f"train {loss[-1]:.4f}")
            axs[0].plot(steps_test, loss_test, label=f"test  {loss_test[-1]:.4f}")
            axs[0].legend()
            # axs[0].set_yscale("log")
            axs[0].set_title("Loss")

            axs[1].plot(steps, x_mse, label=f"train {x_mse[-1]:.4f}")
            axs[1].plot(steps_test, x_mse_test, label=f"test  {x_mse_test[-1]:.4f}")
            axs[1].legend()
            axs[1].set_title("x_mse")

            axs[2].plot(
                steps_test,
                label_paired_image_mse,
                label=f"test {label_paired_image_mse[-1]:.4f}",
                color="red",
            )
            axs[2].legend()
            axs[2].set_title("label_paired_image_mse")

            axs[3].plot(steps, sigma, color="red")
            axs[3].set_yscale("log")
            axs[3].set_title("σ")

            axs[-1].plot(steps, lr_inf, "--", label=f"inf   {lr_inf[-1]:.4f}")
            axs[-1].plot(steps, lr_σ, "--", label=f"σ    {lr_σ[-1]:.4f}")
            axs[-1].plot(
                steps,
                augment_bounds_mult,
                label=f"augment_bounds_mult {augment_bounds_mult[-1]:.4f}",
            )
            axs[-1].legend(loc="lower right")
            axs[-1].set_yscale("log")

            axs[-1].set_xlim(min(steps), max(steps))
            axs[-1].set_xlabel("Steps")
            run.log({"run_metrics_summary": wandb.Image(fig)})


if __name__ == "__main__":
    config.config_with_absl()
    app.run(main)
