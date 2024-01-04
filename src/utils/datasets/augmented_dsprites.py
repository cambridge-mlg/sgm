import enum
import functools
from enum import StrEnum
from os import PathLike
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Protocol, Union

import jax
import jax.numpy as jnp
import numpy as np
import requests
import scipy.stats
import tensorflow as tf
from jax import random

from src.transformations.affine import (
    gen_affine_matrix_no_shear,
    transform_image_with_affine_matrix,
)


# --- Schema classes for the configuration of the augmented DSprites dataset:
class DistributionType(StrEnum):
    UNIFORM = enum.auto()
    TRUNCATED_NORMAL = enum.auto()
    TRUNCATED_BINORMAL = enum.auto()
    BIUNIFORM = enum.auto()
    DELTA = enum.auto()


class DistributionConfig(NamedTuple):
    type: DistributionType
    kwargs: dict[str, Union[float, int]]


def expected_kwargs_for_distribution_type(
    distribution_type: DistributionType,
) -> set[str]:
    match distribution_type:
        case DistributionType.UNIFORM:
            return {"low", "high"}
        case DistributionType.BIUNIFORM:
            return {"low1", "high1", "low2", "high2"}
        case DistributionType.TRUNCATED_NORMAL:
            return {"loc", "scale", "minval", "maxval"}
        case DistributionType.DELTA:
            return {"value"}
        case _:
            raise NotImplementedError(
                f"Invalid distribution type {distribution_type}."
                f"Expected one of {DistributionType.__members__}"
            )


class ShapeDistributionConfig(Protocol):
    unnormalised_shape_prob: float
    orientation: DistributionConfig
    scale: DistributionConfig
    x_position: DistributionConfig
    y_position: DistributionConfig


class AugDspritesConfig(Protocol):
    """Configuration for the augmented DSprites dataset."""

    square_distribution: ShapeDistributionConfig
    ellipse_distribution: ShapeDistributionConfig
    heart_distribution: ShapeDistributionConfig


class DspritesLatent(NamedTuple):
    value_orientation: float
    value_scale: float
    value_x_position: float
    value_y_position: float
    label_shape: int


ArrayLike = Union[np.ndarray, jnp.ndarray]


class DspritesExamples(NamedTuple):
    image: ArrayLike
    label_shape: ArrayLike
    value_scale: ArrayLike
    value_orientation: ArrayLike
    value_x_position: ArrayLike
    value_y_position: ArrayLike


@functools.cache
def construct_dsprites(
    root: PathLike,
    download: bool = True,
) -> DspritesExamples:
    """
    DSprites dataset from https://github.com/deepmind/dsprites-dataset with 3 classes (shapes).

    Contains 737280 images.

    Images (inputs) are of shape [64, 64, 1]

    """
    filepath = Path(root) / "dsprites.npz"
    # filepath_hda = Path(root) / "dsprites.hdf5"
    if download and not filepath.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
        download_dsprites(filepath)
    # Load dataset from file
    dataset_zip = np.load(filepath, allow_pickle=True, encoding="latin1")

    imgs = dataset_zip["imgs"]
    shapes = dataset_zip["latents_classes"][:, 1]
    latents = dataset_zip["latents_values"][:, 2:]
    scales, orientations, x_positions, y_positions = latents.T
    return DspritesExamples(
        image=imgs,
        label_shape=shapes,
        value_scale=scales,
        value_orientation=orientations,
        value_x_position=x_positions,
        value_y_position=y_positions,
    )


def download_dsprites(filepath: PathLike):
    filepath = Path(filepath)
    # Download the dataset from https://github.com/deepmind/dsprites-dataset
    if filepath.exists():
        raise FileExistsError(f"File {filepath} already exists.")
    url = "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"
    r = requests.get(url, allow_redirects=True)
    # Open or create file:
    with open(filepath, "wb") as f:
        f.write(r.content)


def construct_augmented_dsprites(
    aug_dsprites_config: AugDspritesConfig,
    sampler_rng: random.PRNGKeyArray,
) -> tf.data.Dataset:
    dataset_examples = construct_dsprites(root="data/dsprites", download=True)

    # --- Get the log-density of each example's latent configuration under config:
    @jax.jit
    def latent_log_prob_for_config(
        value_orientation: float,
        value_scale: float,
        value_x_position: float,
        value_y_position: float,
        label_shape: int,
    ) -> float:
        return latent_log_prob(
            DspritesLatent(
                value_orientation=value_orientation,
                value_scale=value_scale,
                value_x_position=value_x_position,
                value_y_position=value_y_position,
                label_shape=label_shape,
            ),
            config=aug_dsprites_config,
        )

    log_probs = jax.vmap(latent_log_prob_for_config, in_axes=(0, 0, 0, 0, 0))(
        dataset_examples.value_orientation,
        dataset_examples.value_scale,
        dataset_examples.value_x_position,
        dataset_examples.value_y_position,
        dataset_examples.label_shape,
    )
    # Convert to numpy and float64 for numerical stability:
    log_probs = np.array(log_probs, dtype=np.float64)

    # These are log-densities, so they will not sum to 1. Need to normalise.
    # However, we can't just divide by the sum, because we want to preserve the marginal
    # probability of each shape. Hence, we need to find the indices for a given shape, and
    # then normalise the log-densities for that shape.
    def normalise_log_probs_for_mask(
        log_p: np.ndarray, mask: np.ndarray, total_prob: float
    ) -> np.ndarray:
        """Normalise the probs to sum up to total_prob"""
        if total_prob == 0.0:
            return np.where(mask, -np.infty, log_p)
        else:
            return np.where(
                mask,
                log_p
                - scipy.special.logsumexp(np.where(mask, log_p, -np.infty))
                + np.log(total_prob),
                log_p,
            )

    # Normalise for squares (0), ellipses (1), and hearts (2):
    shape_probs = (
        shape_probs := np.array(
            [
                aug_dsprites_config.square_distribution.unnormalised_shape_prob,
                aug_dsprites_config.ellipse_distribution.unnormalised_shape_prob,
                aug_dsprites_config.heart_distribution.unnormalised_shape_prob,
            ]
        )
    ) / shape_probs.sum()

    log_probs = normalise_log_probs_for_mask(
        log_probs, dataset_examples.label_shape == 0, total_prob=shape_probs[0]
    )
    log_probs = normalise_log_probs_for_mask(
        log_probs, dataset_examples.label_shape == 1, total_prob=shape_probs[1]
    )
    log_probs = normalise_log_probs_for_mask(
        log_probs, dataset_examples.label_shape == 2, total_prob=shape_probs[2]
    )

    # Normalise once more to get the final probabilities (and for numerical stability)
    norm_log_probs = log_probs - scipy.special.logsumexp(log_probs)
    probs = np.exp(norm_log_probs)

    def example_idx_sampler(
        rng: random.PRNGKeyArray, num_vectorized_idx_samples: int = 50000
    ) -> Iterable[int]:
        """
        Sample an example index from the dataset with probability proportional to `probs`.
        `num_batched_samples` helps with speed by vectorising the otherwise expenseive
        `jax.random.choice` function.
        """
        rng = np.random.default_rng(np.array(rng))
        num_examples = len(probs)
        while True:
            yield rng.choice(num_examples, size=num_vectorized_idx_samples, p=probs)

    num_vectorized_idx_samples = aug_dsprites_config.get(
        "num_vectorized_idx_samples", 100000
    )
    index_dataset = tf.data.Dataset.from_generator(
        example_idx_sampler,
        output_signature=tf.TensorSpec(
            shape=(num_vectorized_idx_samples,), dtype=tf.int64
        ),
        args=(sampler_rng, num_vectorized_idx_samples),
    )
    # Unbatch vectorised samples of indices
    index_dataset = index_dataset.unbatch()

    dataset_examples_tf = DspritesExamples(
        image=tf.convert_to_tensor(dataset_examples.image[..., None]),
        label_shape=tf.convert_to_tensor(dataset_examples.label_shape),
        value_scale=tf.convert_to_tensor(dataset_examples.value_scale),
        value_orientation=tf.convert_to_tensor(dataset_examples.value_orientation),
        value_x_position=tf.convert_to_tensor(dataset_examples.value_x_position),
        value_y_position=tf.convert_to_tensor(dataset_examples.value_y_position),
    )

    @tf.function
    def get_dataset_example(idx):
        return {
            "image": dataset_examples_tf.image[idx],
            "label_shape": dataset_examples_tf.label_shape[idx],
            "value_scale": dataset_examples_tf.value_scale[idx],
            "value_orientation": dataset_examples_tf.value_orientation[idx],
            "value_x_position": dataset_examples_tf.value_x_position[idx],
            "value_y_position": dataset_examples_tf.value_y_position[idx],
        }

    return index_dataset.map(get_dataset_example, num_parallel_calls=tf.data.AUTOTUNE)


def latent_log_prob(
    latent: DspritesLatent,
    config: AugDspritesConfig,
) -> float:
    """
    Evaluate the log-prob/density of the latent configuration under the given config. Written to be jittable and
    vmappable by jax.

    Not always a log-prob, not always a density, but a mix. Depends on the distribution.
    As long as we normalise the log-probs on a discrete grid at the end (and don't do mixtures of discrete and
    continuous distributions), it's fine.
    """
    config_per_shape: dict[int, ShapeDistributionConfig] = {
        0: config.square_distribution,
        1: config.ellipse_distribution,
        2: config.heart_distribution,
    }

    class LatentLogProbFuncPerShape(NamedTuple):
        orientation: ScalarLogDensity
        scale: ScalarLogDensity
        x_position: ScalarLogDensity
        y_position: ScalarLogDensity

    log_probs_funcs_per_shape = {
        shape: LatentLogProbFuncPerShape(
            orientation=get_log_prob_func(shape_config.orientation),
            scale=get_log_prob_func(shape_config.scale),
            x_position=get_log_prob_func(shape_config.x_position),
            y_position=get_log_prob_func(shape_config.y_position),
        )
        for shape, shape_config in config_per_shape.items()
    }

    def per_latent_log_prob_funcs_to_joint_log_prob(
        log_probs_funcs: LatentLogProbFuncPerShape,
    ):
        """
        Get the joint log-probability/density of a latent configuration from the log-probability.

        Cautionary tale: do not use dictionary/list comprehensions with lambda functions
        (e.g. `{shape: lambda x: func_for_shape[shape](x) for shape in shapes}`), but
        construct functions explicitely. That's because the lambda function will be evaluated
        at the end of the loop, and will always use the last value of the loop variable.
        """

        def joint_log_prob(l: DspritesLatent) -> float:
            return (
                log_probs_funcs.orientation(l.value_orientation)
                + log_probs_funcs.scale(l.value_scale)
                + log_probs_funcs.x_position(l.value_x_position)
                + log_probs_funcs.y_position(l.value_y_position)
            )

        return joint_log_prob

    return jax.lax.switch(
        latent.label_shape,
        [
            per_latent_log_prob_funcs_to_joint_log_prob(
                log_probs_funcs_per_shape[shape]
            )
            for shape in range(3)
        ],
        latent,
    )


ScalarLogDensity = Callable[[float], float]


def truncated_gaussian_log_prob(
    loc: float,
    scale: float,
    minval: float,
    maxval: float,
) -> ScalarLogDensity:
    return lambda x: jax.scipy.stats.truncnorm.logpdf(
        x=x,
        a=(minval - loc) / scale,
        b=(maxval - loc) / scale,
        loc=loc,
        scale=scale,
    )


def mixture_log_prob(
    prob_a: float,
    component_a: ScalarLogDensity,
    component_b: ScalarLogDensity,
) -> ScalarLogDensity:
    return lambda x: jnp.logaddexp(
        jnp.log(prob_a) + component_a(x),
        jnp.log(1 - prob_a) + component_b(x),
    )


def convert_str_to_distribution_config(
    distribution_conf: DistributionConfig | str,
) -> DistributionConfig:
    match distribution_conf:
        case DistributionConfig(_, _):
            return distribution_conf
        case str():
            # expecting a string of the form "distribution_type(kwargs1=val1, ....)"
            # e.g. "uniform(low=0.0, high=1.0)"
            assert distribution_conf.endswith(
                ")"
            ), "Expecting a string of the form 'distribution_type(kwargs1=val1, ....)'"
            distribution_type = distribution_conf.split("(")[0]
            kwargs = [
                kwarg_val_pair.split("=")
                for kwarg_val_pair in distribution_conf.strip()
                .replace(" ", "")
                .split("(")[1][:-1]
                .split(",")
            ]
            return DistributionConfig(
                type=distribution_type,
                kwargs={kwarg: float(val) for kwarg, val in kwargs},
            )
        case _:
            raise NotImplemented(
                f"Invalid distribution configuration {distribution_conf}"
            )


def get_log_prob_func(
    distribution_conf: DistributionConfig | str,
) -> ScalarLogDensity:
    distribution_conf = convert_str_to_distribution_config(distribution_conf)
    kwargs = distribution_conf.kwargs
    assert set(kwargs.keys()) == expected_kwargs_for_distribution_type(
        distribution_conf.type
    ), (
        f"Expected kwargs {expected_kwargs_for_distribution_type(distribution_conf.type)} "
        f"for distribution type {distribution_conf.type}, got {set(kwargs.keys())}"
    )
    match distribution_conf.type:
        case DistributionType.UNIFORM:
            return lambda x: jax.scipy.stats.uniform.logpdf(
                x=x, loc=kwargs["low"], scale=kwargs["high"] - kwargs["low"]
            )
        case DistributionType.BIUNIFORM:
            component_a = lambda x: jax.scipy.stats.uniform.logpdf(
                x=x, loc=kwargs["low1"], scale=kwargs["high1"] - kwargs["low1"]
            )
            component_b = lambda x: jax.scipy.stats.uniform.logpdf(
                x=x, loc=kwargs["low2"], scale=kwargs["high2"] - kwargs["low2"]
            )
            return lambda x: mixture_log_prob(
                prob_a=0.5,
                component_a=component_a,
                component_b=component_b,
            )(x)
        case DistributionType.TRUNCATED_NORMAL:
            return lambda x: truncated_gaussian_log_prob(
                loc=kwargs["loc"],
                scale=kwargs["scale"],
                minval=kwargs["minval"],
                maxval=kwargs["maxval"],
            )(x)
        case DistributionType.DELTA:
            return lambda x: jax.lax.select(x == kwargs["value"], 0.0, -np.infty)
        case _:
            raise NotImplemented(f"Invalid distribution type {distribution_conf.type}")


DSPRITES_SCALES = np.linspace(0.5, 1.0, 6)
DSPRITES_ORIENTATIONS = np.linspace(0.0, 2 * np.pi, 40)
DSPRITES_X_POSITIONS = np.linspace(0.0, 1.0, 32)
DSPRITES_Y_POSITIONS = np.linspace(0.0, 1.0, 32)


NAME_TO_DOMAIN = {
    "scale": DSPRITES_SCALES,
    "orientation": DSPRITES_ORIENTATIONS,
    "x_position": DSPRITES_X_POSITIONS,
    "y_position": DSPRITES_Y_POSITIONS,
}


def visualise_latent_distribution_from_config(aug_dsprites_config: AugDspritesConfig):
    """
    A plotting utility to visualise the chosen latent distributions from a config.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        nrows=3, ncols=4, figsize=(10, 6), sharex="col", sharey="col"
    )

    for row_idx, (shape, shape_config) in enumerate(
        zip(
            ["heart", "ellipse", "square"],
            [
                aug_dsprites_config.heart_distribution,
                aug_dsprites_config.ellipse_distribution,
                aug_dsprites_config.square_distribution,
            ],
        )
    ):
        for col_idx, (latent_name, latent_config) in enumerate(
            zip(
                ["scale", "orientation", "x_position", "y_position"],
                [
                    shape_config.scale,
                    shape_config.orientation,
                    shape_config.x_position,
                    shape_config.y_position,
                ],
            )
        ):
            domain = NAME_TO_DOMAIN[latent_name]
            log_probs = jax.vmap(get_log_prob_func(latent_config))(domain)
            probs = jnp.exp(log_probs)
            non_zero_idxs = probs > 0.0
            axes[row_idx, col_idx].stem(domain[non_zero_idxs], probs[non_zero_idxs])

            xlim_buffer = 0.05 * (domain[-1] - domain[0])
            axes[row_idx, col_idx].set_xlim(
                [domain[0] - xlim_buffer, domain[-1] + xlim_buffer]
            )
            axes[row_idx, col_idx].set_yticks([])
            axes[row_idx, col_idx].set_xlabel(latent_name.capitalize())
        axes[row_idx, 0].set_ylabel(shape.capitalize() + " Distribution")
    return fig, axes


def plot_prototypes_by_shape(prototype_function, batch):
    import matplotlib.pyplot as plt

    rng = random.PRNGKey(0)

    def get_proto(x):
        η = prototype_function(x, rng)
        xhat = transform_image_with_affine_matrix(
            x, jnp.linalg.inv(gen_affine_matrix_no_shear(η)), order=3
        )
        return xhat

    square_images = batch["image"][batch["label"] == 0]
    square_prototypes = jax.vmap(get_proto)(square_images)
    ellipse_images = batch["image"][batch["label"] == 1]
    ellipse_prototypes = jax.vmap(get_proto)(ellipse_images)
    heart_images = batch["image"][batch["label"] == 2]
    heart_prototypes = jax.vmap(get_proto)(heart_images)

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
    fig, axes = plt.subplots(ncols=ncols, nrows=6, figsize=(ncols * 1.5, 1.5 * 6))
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
    return fig
