"""
Plotting functions and utilities for logging training progress of a canonicalizer (prototype inference) model.
"""
from typing import Callable, Protocol
import distrax
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import math


import matplotlib.pyplot as plt

from transformations import transform_image


class GetPrototypeFn(Protocol):
    def __call__(
        self,
        image: jnp.ndarray,
        rng: jnp.ndarray,
        params,
    ) -> jnp.ndarray:
        ...


def construct_plot_data_samples_canonicalizations(get_prototype_fn: GetPrototypeFn, n_images: int = 56):
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
            axes[0, 2 * j + 1].set_title("Canonicalized")
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

def construct_plot_augmented_data_samples_canonicalizations(get_prototype_fn: GetPrototypeFn, config, n_images: int = 7, n_samples: int = 8):
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

    return plot_augmented_data_samples_canonicalizations


def construct_plot_training_augmented_samples(config, n_images: int):
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
                    axes[i, 2 * j + 1].imshow(
                        images_augmented[idx], **imshow_kwargs
                    )
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
