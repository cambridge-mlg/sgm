from ml_collections import ConfigDict
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Mapping, NamedTuple, Protocol
from jax import random
import jax
import jax.numpy as jnp
from jax._src import dtypes

from src.utils.algorithmic import get_closest_value_in_sorted_sequence


import enum
import functools
from typing import Callable, Optional, Union
import distrax
from jax import random
import jax.numpy as jnp
from ml_collections import ConfigDict


# --- Schema classes for the configuration of the augmented DSprites dataset:
class DistributionType(str, enum.Enum):
    UNIFORM = "uniform"
    TRUNCATED_NORMAL = "normal"
    TRUNCATED_BINORMAL = "truncated_binormal"
    BIUNIFORM = "biuniform"
    DELTA = "delta"


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


def construct_augmented_dsprites(
    aug_dsprites_config: AugDspritesConfig,
    sampler_rng: random.PRNGKeyArray,
) -> tf.data.Dataset:
    ds_builder = tfds.builder("dsprites")
    ds_builder.download_and_prepare()
    ds = ds_builder.as_dataset(split="train")

    # --- Load the dataset into memory so that it can be easily indexed into
    # (It takes about ~1min to decompress TFRecord files. Could save the dataset to disk
    # in a format that allows for easier random access.)
    dataset_list = list(ds)

    # Make a mapping so that we can easily look up the index for a given
    # configuration of augmentations:
    latent_to_index: Mapping[DspritesLatent, int] = {
        extract_latents_from_example(example): idx for idx, example in enumerate(dataset_list)
    }

    # --- Get all the unique configurations of latents
    # (all compositions of unique
    # orientations/scales/... exist in DSprites, so just get the unique values
    # for each latent variable):
    dsprite_unique_orientations = set()
    dsprite_unique_scales = set()
    dsprite_unique_x_positions = set()
    dsprite_unique_y_positions = set()
    dsprite_unique_shapes = set()

    for latent in latent_to_index.keys():
        dsprite_unique_orientations.add(latent.value_orientation)
        dsprite_unique_scales.add(latent.value_scale)
        dsprite_unique_x_positions.add(latent.value_x_position)
        dsprite_unique_y_positions.add(latent.value_y_position)
        dsprite_unique_shapes.add(latent.label_shape)

    dsprite_unique_orientations = sorted(dsprite_unique_orientations)
    dsprite_unique_scales = sorted(dsprite_unique_scales)
    dsprite_unique_x_positions = sorted(dsprite_unique_x_positions)
    dsprite_unique_y_positions = sorted(dsprite_unique_y_positions)

    # Function to retrieve the closest example in the dataset to a given latent (since we'll sample
    # latents possibly inbetween values of the discrete domain of the dataset)
    def get_closest_dsprites_example_given_latent(latent: DspritesLatent) -> int:
        closest_latent = DspritesLatent(
            value_orientation=get_closest_value_in_sorted_sequence(
                latent.value_orientation, dsprite_unique_orientations
            ),
            value_scale=get_closest_value_in_sorted_sequence(
                latent.value_scale, dsprite_unique_scales
            ),
            value_x_position=get_closest_value_in_sorted_sequence(
                latent.value_x_position, dsprite_unique_x_positions
            ),
            value_y_position=get_closest_value_in_sorted_sequence(
                latent.value_y_position, dsprite_unique_y_positions
            ),
            label_shape=latent.label_shape,
        )
        return latent_to_index[closest_latent]

    # Get a generator that samples latents according to the distributions specified in the configuration:

    latent_sampler: Callable[[random.PRNGKey], DspritesLatent] = get_dsprites_latent_sampler(
        aug_dsprites_config
    )

    # Make a generator that returns examples with latents sampled from the latent_sampler:
    def example_generator(rng: random.PRNGKey):
        while True:
            rng, rng_sample = random.split(rng)
            latent = latent_sampler(rng_sample)
            idx = get_closest_dsprites_example_given_latent(latent)
            yield dataset_list[idx]

    return tf.data.Dataset.from_generator(
        example_generator, output_signature=ds.element_spec, args=(sampler_rng,)
    )


def extract_latents_from_example(example: dict):
    return DspritesLatent(
        value_orientation=float(example["value_orientation"]),
        value_scale=float(example["value_scale"]),
        value_x_position=float(example["value_x_position"]),
        value_y_position=float(example["value_y_position"]),
        label_shape=int(example["label_shape"]),
    )


def get_dsprites_latent_sampler(config: AugDspritesConfig):
    shape_distribution = distrax.Categorical(probs=jnp.ones(3) / 3)
    config_per_shape: dict[int, ShapeDistributionConfig] = {
        0: config.square_distribution,
        1: config.ellipse_distribution,
        2: config.heart_distribution,  # TODO check order is right
    }

    def sample(rng: random.PRNGKey) -> DspritesLatent:
        rng_shape, rng = random.split(rng)
        rng_orient, rng_scale, rng_x, rng_y = random.split(rng, num=4)
        return DspritesLatent(
            # Sample the label
            label_shape=(shape := int(shape_distribution.sample(seed=rng_shape))),
            # Then, sample the other latents conditioned on the label:
            value_orientation=construct_sample_func(config_per_shape[shape].orientation)(rng_orient),
            value_scale=construct_sample_func(config_per_shape[shape].scale)(rng_scale),
            value_x_position=construct_sample_func(config_per_shape[shape].x_position)(rng_x),
            value_y_position=construct_sample_func(config_per_shape[shape].y_position)(rng_y),
        )

    return sample


def truncated_gaussian(key, loc, scale, minval, maxval, shape=None, dtype=dtypes.float_):
    """Truncated normal with changeable location and scale"""
    return loc + scale * random.truncated_normal(
        lower=(minval - loc) / scale,
        upper=(maxval - loc) / scale,
        key=key,
        shape=shape,
        dtype=dtype,
    )


def mixture_sample(
    key: random.PRNGKeyArray,
    prob_a: float,
    component_a: Callable[[random.PRNGKeyArray], float],
    component_b: Callable[[random.PRNGKeyArray], float],
) -> float:
    """Sample from a mixture of two distributions"""
    key_which_component, key_component_a, key_component_b = random.split(key, num=3)
    is_component_a = random.bernoulli(key_which_component, prob_a)
    a = component_a(key_component_a)
    b = component_b(key_component_b)
    return jnp.where(is_component_a, a, b)


def construct_sample_func(
    distribution_conf: DistributionConfig,
) -> distrax.Distribution:
    kwargs = distribution_conf.kwargs
    assert set(kwargs.keys()) == expected_kwargs_for_distribution_type(distribution_conf.type), (
        f"Expected kwargs {expected_kwargs_for_distribution_type(distribution_conf.type)} "
        f"for distribution type {distribution_conf.type}, got {set(kwargs.keys())}"
    )
    match distribution_conf.type:
        case DistributionType.UNIFORM:
            return lambda rng: float(
                random.uniform(key=rng, minval=kwargs["low"], maxval=kwargs["high"])
            )
        case DistributionType.BIUNIFORM:
            return lambda rng: float(
                mixture_sample(
                    key=rng,
                    prob_a=0.5,
                    component_a=lambda key: random.uniform(
                        key=key, minval=kwargs["low1"], maxval=kwargs["high1"]
                    ),
                    component_b=lambda key: random.uniform(
                        key=key, minval=kwargs["low2"], maxval=kwargs["high2"]
                    ),
                )
            )
        case DistributionType.TRUNCATED_NORMAL:
            return lambda rng: float(
                truncated_gaussian(
                    key=rng,
                    loc=kwargs["loc"],
                    scale=kwargs["scale"],
                    minval=kwargs["minval"],
                    maxval=kwargs["maxval"],
                )
            )
        case _:
            raise NotImplemented(f"Invalid distribution type {distribution_conf.type}")
