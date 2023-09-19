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
            raise NotImplemented(f"Invalid distribution type {distribution_type}")


class ShapeDistributionConfig(Protocol):
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
    value_orientation:ArrayLike
    value_x_position:  ArrayLike
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
    def normalise_log_probs_for_mask(log_p: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return log_p - scipy.special.logsumexp(np.where(mask, log_p, -np.infty)) * mask

    # Normalise for squares (0), ellipses (1), and hearts (2):
    log_probs = normalise_log_probs_for_mask(log_probs, dataset_examples.label_shape == 0)
    log_probs = normalise_log_probs_for_mask(log_probs, dataset_examples.label_shape == 1)
    log_probs = normalise_log_probs_for_mask(log_probs, dataset_examples.label_shape == 2)

    # Normalise once more to get the final probabilities (and for numerical stability)
    norm_log_probs = log_probs - scipy.special.logsumexp(log_probs)
    probs = np.exp(norm_log_probs)

    def example_idx_sampler(rng: random.PRNGKeyArray, num_vectorized_idx_samples: int = 50000) -> Iterable[int]:
        """
        Sample an example index from the dataset with probability proportional to `probs`.
        `num_batched_samples` helps with speed by vectorising the otherwise expenseive
        `jax.random.choice` function.
        """
        rng = np.random.default_rng(np.array(rng))
        num_examples = len(probs)
        while True:
            yield from rng.choice(num_examples, size=num_vectorized_idx_samples, p=probs)
    
    index_dataset = tf.data.Dataset.from_generator(
        example_idx_sampler,
        output_signature=tf.TensorSpec(shape=(), dtype=tf.int64),
        args=(sampler_rng, aug_dsprites_config.get("num_vectorized_idx_samples", 100000)),
    )

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
    continuous districtuions), it's fine.
    """
    config_per_shape: dict[int, ShapeDistributionConfig] = {
        0: config.square_distribution,
        1: config.ellipse_distribution,
        2: config.heart_distribution,  # TODO check order is right
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

    def per_latent_log_prob_funcs_to_joint_log_prob(log_probs_funcs: LatentLogProbFuncPerShape):
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
                + jnp.log(1 / 3)  # Shape probability
            )
        return joint_log_prob

    return jax.lax.switch(
        latent.label_shape,
        [
            per_latent_log_prob_funcs_to_joint_log_prob(log_probs_funcs_per_shape[shape])
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


def get_log_prob_func(
    distribution_conf: DistributionConfig,
) -> ScalarLogDensity:
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
            return lambda x: jnp.select(x == kwargs["value"], 0, -np.infty)
        case _:
            raise NotImplemented(f"Invalid distribution type {distribution_conf.type}")
