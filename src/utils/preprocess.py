import math
import dataclasses
from typing import Any, List, Optional

from clu import preprocess_spec
import tensorflow as tf
import tensorflow_addons as tfa


Features = preprocess_spec.Features
PRNGKey = Any


def all_ops():
    """Returns all preprocessing ops defined in this module."""
    return preprocess_spec.get_all_ops(__name__)


@dataclasses.dataclass
class ValueRange:
    """Transforms a [in_min, in_max] image to [vmin, vmax] range.
    Input ranges in_min/in_max can be equal-size lists to rescale the invidudal
    channels independently.
    Attributes:
      vmin: A scalar. Output max value.
      vmax: A scalar. Output min value.
      in_min: A scalar or a list of input min values to scale. If a list, the
        length should match to the number of channels in the image.
      in_max: A scalar or a list of input max values to scale. If a list, the
        length should match to the number of channels in the image.
      clip_values: Whether to clip the output values to the provided ranges.
      key: Key of the data to be processed.
      key_result: Key under which to store the result (same as `key` if None).
    """

    vmin: float = -1
    vmax: float = 1
    in_min: float = 0
    in_max: float = 255.0
    clip_values: bool = False
    key: str = "image"
    key_result: Optional[str] = None

    def __call__(self, features: Features) -> Features:
        image = features[self.key]
        in_min_t = tf.constant(self.in_min, tf.float32)
        in_max_t = tf.constant(self.in_max, tf.float32)
        image = tf.cast(image, tf.float32)
        image = (image - in_min_t) / (in_max_t - in_min_t)
        image = self.vmin + image * (self.vmax - self.vmin)
        if self.clip_values:
            image = tf.clip_by_value(image, self.vmin, self.vmax)
        features[self.key_result or self.key] = image  # type: ignore
        return features


@dataclasses.dataclass
class RandomRotate:
    """Randomly rotates an image uniformly in the range [θ_min, θ_max].

    Attributes:
      θ_min: A scalar. The minimum rotation in degrees.
      θ_max: A scalar. The maximum rotation in degrees.
      fill_mode: A string. The fill mode. One of 'constant', 'reflect', 'wrap', 'nearest'.
      fill_value: A scalar. The value to fill the empty pixels when using 'constant' fill mode.
      key: Key of the data to be processed.
      key_result: Key under which to store the result (same as `key` if None).
      rng_key: Key of the random number used for `tf.random.stateless_uniform`.
    """

    θ_min: float = -45
    θ_max: float = 45
    fill_mode: str = "nearest"
    fill_value: float = 0.0
    key: str = "image"
    key_result: Optional[str] = None
    rng_key: str = "rng"

    def __call__(self, features: Features) -> Features:
        image = features[self.key]
        rng = features[self.rng_key]
        self.θ_min = self.θ_min * math.pi / 180
        self.θ_max = self.θ_max * math.pi / 180
        θ = tf.random.stateless_uniform((), rng, self.θ_min, self.θ_max)  # type: ignore

        image = tfa.image.rotate(image, θ, "bilinear", fill_mode=self.fill_mode, fill_value=self.fill_value)  # type: ignore
        features[self.key_result or self.key] = image
        return features


@dataclasses.dataclass
class RandomZoom:
    """Randomly zooms an image.

    Attributes:
      x_min: A scalar. The minimum x-axis stretch factor.
      x_max: A scalar. The maximum x-axis stretch factor.
      y_min: A scalar. The minimum y-axis stretch factor.
      y_max: A scalar. The maximum y-axis stretch factor.
      fill_mode: A string. The fill mode. One of 'constant', 'reflect', 'wrap', 'nearest'.
      fill_value: A scalar. The value to fill the empty pixels when using 'constant' fill mode.
      key: Key of the data to be processed.
      key_result: Key under which to store the result (same as `key` if None).
      rng_key: Key of the random number used for `tf.random.stateless_uniform`.
    """

    x_min: float = 2 / 3
    x_max: float = 1.5
    y_min: float = 2 / 3
    y_max: float = 1.5
    fill_mode: str = "nearest"
    fill_value: float = 0.0
    key: str = "image"
    key_result: Optional[str] = None
    rng_key: str = "rng"

    def __call__(self, features: Features) -> Features:
        image = features[self.key]
        rng = features[self.rng_key]
        x_rng, y_rng = tf.unstack(tf.random.experimental.stateless_split(rng, 2))
        x_zoom = 1/tf.random.stateless_uniform((), x_rng, self.x_min, self.x_max)  # type: ignore
        y_zoom = 1/tf.random.stateless_uniform((), y_rng, self.y_min, self.y_max)  # type: ignore

        image_shape = tf.shape(image)
        image_height = tf.cast(image_shape[-3], tf.float32)
        image_width = tf.cast(image_shape[-2], tf.float32)
        x_offset = ((image_width - 1.0) / 2.0) * (1.0 - x_zoom)
        y_offset = ((image_height - 1.0) / 2.0) * (1.0 - y_zoom)

        transforms = tf.stack([x_zoom, 0, x_offset, 0, y_zoom, y_offset, 0, 0], axis=0)

        image = tfa.image.transform(image, transforms, "bilinear", fill_mode=self.fill_mode, fill_value=self.fill_value)  # type: ignore
        features[self.key_result or self.key] = image
        return features


@dataclasses.dataclass
class Keep:
    """Keeps only the given keys.
    Attributes:
      keys: List of string keys to keep.
    """

    keys: List[str]

    def __call__(self, features: Features) -> Features:
        return {k: v for k, v in features.items() if k in self.keys}
