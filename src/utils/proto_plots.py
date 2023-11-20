"""
Plotting functions and utilities for logging training progress of a transformation inference (prototype inference) model.
"""
import math
from typing import Protocol

import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax import random

from src.transformations import transform_image
from src.utils.plotting import rescale_for_imshow


class GetPrototypeFn(Protocol):
    def __call__(
        self,
        image: jnp.ndarray,
        rng: jnp.ndarray,
        params,
    ) -> jnp.ndarray:
        ...


def construct_plot_data_samples_canonicalizations(
    get_prototype_fn: GetPrototypeFn, n_images: int = 56
):
    nrows = math.ceil(math.sqrt(n_images))
    ncols = math.ceil(n_images / nrows) * 2

    @jax.jit
    def get_images_and_prototypes(state, batch):
        step_rng = random.fold_in(state.rng, state.step)

        # Group the images by labels:
        images = batch["image"][0, :n_images]
        sort_idxs = jnp.argsort(batch["label"][0, :n_images])
        images_to_plot = images[sort_idxs]

        # Get the canonicalizations:
        prototypes = jax.vmap(get_prototype_fn, in_axes=(0, 0, None))(
            images_to_plot,
            random.split(step_rng, len(images_to_plot)),
            state.params,
        )
        return images_to_plot, prototypes

    def plot_data_samples_canonicalizations(state, batch):
        images_to_plot, prototypes = get_images_and_prototypes(state, batch)
        images_to_plot, prototypes = map(
            lambda z: np.array(z), (images_to_plot, prototypes)
        )

        vmin = min(prototypes.min(), images_to_plot.min())
        vmax = max(prototypes.max(), images_to_plot.max())
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
            axes[0, 2 * j + 1].set_title("Prototype")
        return fig

    return plot_data_samples_canonicalizations


def get_aug_image_fn(config):
    @jax.jit
    def aug_image(image, img_rng, augment_bounds_mult: float):
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

    return aug_image


def construct_plot_augmented_data_samples_canonicalizations(
    get_prototype_fn: GetPrototypeFn, config, n_images: int = 7, n_samples: int = 8
):
    aug_image = get_aug_image_fn(config)

    @jax.jit
    def get_augmented_images_and_prototypes(state, batch):
        step_rng = random.fold_in(state.rng, state.step)

        # Group the images by labels:
        images = batch["image"][0, :n_images]  # Shape (n_images, H, W, 1)
        sort_idxs = jnp.argsort(batch["label"][0, :n_images])
        base_images_to_plot = images[sort_idxs]

        images_to_plot = jax.vmap(
            lambda rng: jax.vmap(
                aug_image,
                in_axes=(0, 0, None),
            )(
                base_images_to_plot,
                random.split(rng, n_images),
                state.augment_bounds_mult,
            ),
            in_axes=(0),
        )(
            random.split(step_rng, n_samples)
        )  # Shape (n_samples, n_images, H, W, 1)
        # Get the canonicalizations:
        prototypes = jax.vmap(
            lambda im, rng, params: jax.vmap(
                get_prototype_fn,
                in_axes=(0, 0, None),
            )(im, random.split(rng, n_images), params),
            in_axes=(0, 0, None),
        )(
            images_to_plot,
            random.split(step_rng, len(images_to_plot)),
            state.params,
        )
        return images_to_plot, prototypes

    def plot_augmented_data_samples_canonicalizations(state, batch):
        """
        Plot augmented data samples and their canonicalizations. The columns are grouped by the original image,
        and each row within shows an augmented image and the corresponding canonicalization.
        """

        images_to_plot, prototypes = get_augmented_images_and_prototypes(state, batch)
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
            axes[0, 1].set_title("Proto.", fontsize=9)
            for ax in axes.ravel():
                ax.axis("off")
        return fig

    return plot_augmented_data_samples_canonicalizations


def construct_plot_training_augmented_samples(config, n_images: int = 49):
    aug_image = get_aug_image_fn(config)

    nrows = math.ceil(math.sqrt(n_images))
    ncols = math.ceil(n_images / nrows) * 2

    @jax.jit
    def get_image_original_and_augment(state, batch):
        step_rng = random.fold_in(state.rng, state.step)

        # Group the images by labels:
        images = batch["image"][0, :n_images]

        images_augmented = jax.vmap(aug_image, in_axes=(0, 0, None))(
            images, random.split(step_rng, n_images), state.augment_bounds_mult
        )
        return images, images_augmented

    def plot_training_augmented_samples(state, batch):
        images, images_augmented = get_image_original_and_augment(state, batch)
        images, images_augmented = map(
            lambda z: np.array(z), (images, images_augmented)
        )

        vmin = min(images.min(), images_augmented.min())
        vmax = max(images.max(), images_augmented.max())
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
                    axes[i, 2 * j + 1].imshow(images_augmented[idx], **imshow_kwargs)
            axes[0, 2 * j].set_title("Original")
            axes[0, 2 * j + 1].set_title("Train. augmented")
        return fig

    return plot_training_augmented_samples


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
    return fig


def plot_proto_model_training_metrics(history):
    # Plot the training history
    colors = sns.color_palette("husl", 3)
    steps, loss, x_mse, lr_inf, lr_σ = history.collect(
        "steps",
        "loss",
        "x_mse",
        "lr_inf",
        "lr_σ",
    )
    sigma = history.collect("σ")
    augment_bounds_mult = history.collect("augment_bounds_mult")
    blur_sigma = history.collect("blur_sigma")
    steps_test, loss_test, x_mse_test = history.collect(
        "steps", "loss_test", "x_mse_test"
    )

    label_paired_image_mse = history.collect("label_paired_image_mse_test")

    n_plots = 5
    fig, axs = plt.subplots(
        n_plots, 1, figsize=(15, n_plots * 3.0), dpi=100, sharex=True
    )

    axs[0].plot(steps, loss, label=f"train {loss[-1]:.4f}", color=colors[0])
    axs[0].plot(
        steps_test, loss_test, label=f"test  {loss_test[-1]:.4f}", color=colors[1]
    )
    axs[0].legend()
    # axs[0].set_yscale("log")
    axs[0].set_title("Loss")

    axs[1].plot(steps, x_mse, label=f"train {x_mse[-1]:.4f}", color=colors[0])
    axs[1].plot(
        steps_test, x_mse_test, label=f"test  {x_mse_test[-1]:.4f}", color=colors[1]
    )
    axs[1].legend()
    axs[1].set_title("x_mse")

    axs[2].plot(
        steps_test,
        label_paired_image_mse,
        label=f"test {label_paired_image_mse[-1]:.4f}",
        color=colors[0],
    )
    axs[2].legend()
    axs[2].set_title("label_paired_image_mse")

    axs[3].plot(steps, sigma, color=colors[1])
    axs[3].set_yscale("log")
    axs[3].set_title("σ")

    # Schedule axis:
    lr_axis = axs[-1]
    multiplier_axis = lr_axis.twinx()

    (p1,) = lr_axis.plot(
        steps, lr_inf, "-", label=f"inf   {lr_inf[-1]:.4f}", color=colors[0]
    )
    (p2,) = lr_axis.plot(
        steps, lr_σ, "--", label=f"σ    {lr_σ[-1]:.4f}", color=colors[0]
    )
    (p3,) = multiplier_axis.plot(
        steps,
        augment_bounds_mult,
        "-",
        label=f"augment_bounds_mult {augment_bounds_mult[-1]:.4f}",
        color=colors[1],
    )
    (p4,) = multiplier_axis.plot(
        steps,
        blur_sigma,
        "--",
        label=f"blur sigma {blur_sigma[-1]:.4f}",
        color=colors[1],
    )
    lines = [p1, p2, p3, p4]
    lr_axis.legend(lines, [l.get_label() for l in lines])

    lr_axis.set_yscale("log")
    multiplier_axis.set_yscale("linear")

    lr_axis.set_ylabel(f"LR")
    multiplier_axis.set_ylabel("Multipliers")

    lr_axis.yaxis.label.set_color(p1.get_color())
    multiplier_axis.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    lr_axis.tick_params(axis="y", colors=p1.get_color(), **tkw)
    multiplier_axis.tick_params(axis="y", colors=p3.get_color(), **tkw)

    axs[-1].set_xlim(min(steps), max(steps))
    axs[-1].set_xlabel("Steps")

    for ax in axs:
        ax.grid(color=(0.9, 0.9, 0.9))
    return fig


def plot_protos_and_recons(x, bounds, get_prototype):
    transformed_xs = jax.vmap(transform_image, in_axes=(None, 0))(
        x,
        jnp.linspace(
            -bounds,
            bounds,
            13,
        ),
    )

    xhats, _ = jax.vmap(get_prototype)(transformed_xs)

    fig, axs = plt.subplots(2, len(xhats), figsize=(15, 3))

    for ax, x in zip(axs[0], list(transformed_xs)):
        ax.imshow(rescale_for_imshow(x), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax, xhat in zip(axs[1], list(xhats)):
        ax.imshow(rescale_for_imshow(xhat), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])

    return fig
