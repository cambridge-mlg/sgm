import enum
from enum import StrEnum
from typing import Callable, Mapping, NamedTuple, Protocol, Union

import requests
from pathlib import Path
from os import PathLike
import numpy as np
import distrax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import scipy.stats
from jax import random
from jax._src import dtypes

from src.utils.algorithmic import get_closest_value_in_sorted_sequence


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


def construct_dsprites(
    root: PathLike,
    download: bool = True,
) -> list[dict[str, np.ndarray]]:
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

    dataset_list = [
        {
            "image": img[:, :, None],  # Shape [64, 64, 1] (add channel dim.)
            "label_shape": shape,
            "value_scale": scale,
            "value_orientation": orientation,
            "value_x_position": x_position,
            "value_y_position": y_position,
        } for img, shape, scale, orientation, x_position, y_position in zip(
            imgs, shapes, scales, orientations, x_positions, y_positions
        )
    ]
    return dataset_list


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

    dataset_list = construct_dsprites(root="data/dsprites", download=True)

    # Make a mapping so that we can easily look up the index for a given
    # configuration of augmentations:
    latent_to_index: Mapping[DspritesLatent, int] = {
        extract_latents_from_example(example): idx
        for idx, example in enumerate(dataset_list)
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

    latent_sampler: Callable[
        [np.random.Generator], DspritesLatent
    ] = get_dsprites_latent_sampler(aug_dsprites_config)

    # Make a generator that returns examples with latents sampled from the latent_sampler:
    def example_generator(rng: random.PRNGKey):
        # Get a numpy rng
        rng = np.random.default_rng(np.array(rng))
        while True:
            latent = latent_sampler(rng)
            idx = get_closest_dsprites_example_given_latent(latent)
            yield dataset_list[idx]

    return tf.data.Dataset.from_generator(
        example_generator,
        output_signature=(
            {
                "image": tf.TensorSpec(shape=(64, 64, 1), dtype=tf.uint8),
                "label_shape": tf.TensorSpec(shape=(), dtype=tf.int64),
                "value_scale": tf.TensorSpec(shape=(), dtype=tf.float64),
                "value_orientation": tf.TensorSpec(shape=(), dtype=tf.float64),
                "value_x_position": tf.TensorSpec(shape=(), dtype=tf.float64),
                "value_y_position": tf.TensorSpec(shape=(), dtype=tf.float64),
            }
        ),
        args=(sampler_rng,),
    )


def extract_latents_from_example(example: dict):
    return DspritesLatent(
        value_orientation=float(example["value_orientation"]),
        value_scale=float(example["value_scale"]),
        value_x_position=float(example["value_x_position"]),
        value_y_position=float(example["value_y_position"]),
        label_shape=int(example["label_shape"]),
    )

ScalarSampler = Callable[[np.random.Generator], float]


def get_dsprites_latent_sampler(config: AugDspritesConfig):
    config_per_shape: dict[int, ShapeDistributionConfig] = {
        0: config.square_distribution,
        1: config.ellipse_distribution,
        2: config.heart_distribution,  # TODO check order is right
    }

    class LatentSamplerPerShape(NamedTuple):
        orientation: ScalarSampler
        scale: ScalarSampler
        x_position: ScalarSampler
        y_position: ScalarSampler

    sampler_per_shape = {
        shape: LatentSamplerPerShape(
            orientation=get_sample_func(config.orientation),
            scale=get_sample_func(config.scale),
            x_position=get_sample_func(config.x_position),
            y_position=get_sample_func(config.y_position),
        ) for shape, config in config_per_shape.items()
    }
    
    shapes, probs = [0, 1, 2], [1/3, 1/3, 1/3]
    shape_sampler: ScalarSampler = lambda rng: rng.choice(shapes, p=probs)
    def sample(rng: np.random.Generator) -> DspritesLatent:
        return DspritesLatent(
            # Sample the label
            label_shape=(shape := shape_sampler(rng)),
            # Then, sample the other latents conditioned on the label:
            value_orientation=sampler_per_shape[shape].orientation(rng),
            value_scale=sampler_per_shape[shape].scale(rng),
            value_x_position=sampler_per_shape[shape].x_position(rng),
            value_y_position=sampler_per_shape[shape].y_position(rng),
        )

    return sample


def truncated_gaussian(
    loc: float, scale: float, minval: float, maxval: float,
) -> ScalarSampler:
    distr = scipy.stats.truncnorm(
        a=(minval - loc) / scale,
        b=(maxval - loc) / scale,
        loc=loc,
        scale=scale,
    )
    return lambda rng: distr.rvs(
        random_state=rng,
    )


def mixture_sampler(
    prob_a: float,
    component_a: ScalarSampler,
    component_b: ScalarSampler,
) -> ScalarSampler:
    """Sample from a mixture of two distributions"""
    distr = scipy.stats.bernoulli(p=prob_a)
    return lambda rng: component_a(rng) if distr.rvs(random_state=rng) else component_b(rng)


def get_sample_func(
    distribution_conf: DistributionConfig,
) -> Callable[[random.PRNGKeyArray], float]:
    kwargs = distribution_conf.kwargs
    assert set(kwargs.keys()) == expected_kwargs_for_distribution_type(
        distribution_conf.type
    ), (
        f"Expected kwargs {expected_kwargs_for_distribution_type(distribution_conf.type)} "
        f"for distribution type {distribution_conf.type}, got {set(kwargs.keys())}"
    )
    match distribution_conf.type:
        case DistributionType.UNIFORM:
            distr = scipy.stats.uniform(loc=kwargs["low"], scale=kwargs["high"] - kwargs["low"])
            return lambda rng: distr.rvs(random_state=rng)
        case DistributionType.BIUNIFORM:
            component_a_distr = scipy.stats.uniform(loc=kwargs["low1"], scale=kwargs["high1"] - kwargs["low1"])
            component_b_distr = scipy.stats.uniform(loc=kwargs["low2"], scale=kwargs["high2"] - kwargs["low2"])
            return mixture_sampler(
                prob_a=0.5,
                component_a = lambda rng: component_a_distr.rvs(random_state=rng),
                component_b = lambda rng: component_b_distr.rvs(random_state=rng),
            )
        case DistributionType.TRUNCATED_NORMAL:
            return truncated_gaussian(
                loc=kwargs["loc"],
                scale=kwargs["scale"],
                minval=kwargs["minval"],
                maxval=kwargs["maxval"],
            )
        case DistributionType.DELTA:
            return lambda rng: kwargs["value"]
        case _:
            raise NotImplemented(f"Invalid distribution type {distribution_conf.type}")
