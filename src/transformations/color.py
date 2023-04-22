"""Color transformations of images."""
import jax
from jax import numpy as jnp
from chex import Array, assert_shape, assert_rank


def color_transform_image(
    image: Array,
    η: Array,
):
    """Applies a color transformation to an image.

    Args:
        image: a rank-3 Array of shape (height, width, num channels) – i.e. Jax/TF image format.
        η: an Array with 4 entries:
        * η_0 controls hue.
        * η_1 controls saturation.
        * η_2 controls brightness.
        * η_3 controls contrast.

    Returns:
        A transformed image of same shape as the input.
    """
    assert_rank(image, 3)
    assert_shape(image, (None, None, 3))
    assert_shape(η, (4,))

    rgb_tuple = tuple(jax.tree_map(jnp.squeeze, jnp.split(image, 3, axis=-1)))

    # Convert to HSV.
    hsv_tuple = _rgb_to_hsv(*rgb_tuple)

    # Apply HSV transformations.
    hsv_tuple = _adjust_saturation(hsv_tuple, 1., η[1])
    hsv_tuple = _adjust_hue(hsv_tuple, η[0])

    # Convert back to RGB.
    rgb_tuple = _hsv_to_rgb(*hsv_tuple)

    # Apply RGB transformations.
    rgb_tuple = _adjust_contrast(rgb_tuple, η[3])
    rgb_tuple = _adjust_brightness(rgb_tuple, η[2])

    return jnp.stack(rgb_tuple, axis=-1)


def hsv_transform_image(
    image: Array,
    η: Array,
):
    """Applies a hue transformation to an image.

    Args:
        image: a rank-3 Array of shape (height, width, num channels) – i.e. Jax/TF image format.
        η: an Array with 2 entries:
        * η_0 controls hue.
        * η_1 controls saturation.

    Returns:
        A transformed image of same shape as the input.
    """
    assert_shape(η, (2,))
    η = jnp.array([η[0], η[1], 0, 0])
    return color_transform_image(image, η)


def rgb_transform_image(
    image: Array,
    η: Array,
):
    """Applies a brightness and contrast transformation to an image.

    Args:
        image: a rank-3 Array of shape (height, width, num channels) – i.e. Jax/TF image format.
        η: an Array with 2 entries:
        * η_0 controls brightness.
        * η_1 controls contrast.

    Returns:
        A transformed image of same shape as the input.
    """
    assert_shape(η, (2,))
    η = jnp.array([0, 0, η[0], η[1]])
    return color_transform_image(image, η)


# Adapted from https://github.com/deepmind/deepmind-research/blob/master/byol/utils/augmentations.py.
def _rgb_to_hsv(r, g, b):
    """Converts R, G, B  values to H, S, V values.
    Reference TF implementation:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/adjust_saturation_op.cc
    Only input values between 0 and 1 are guaranteed to work properly, but this
    function complies with the TF implementation outside of this range.
    Args:
        r: A tensor representing the red color component as floats.
        g: A tensor representing the green color component as floats.
        b: A tensor representing the blue color component as floats.
    Returns:
        H, S, V values, each as tensors of shape [...] (same as the input without
        the last dimension).
    """
    vv = jnp.maximum(jnp.maximum(r, g), b)
    range_ = vv - jnp.minimum(jnp.minimum(r, g), b)
    sat = jnp.where(vv > 0, range_ / vv, 0.0)
    norm = jnp.where(range_ != 0, 1.0 / (6.0 * range_), 1e9)

    hr = norm * (g - b)
    hg = norm * (b - r) + 2.0 / 6.0
    hb = norm * (r - g) + 4.0 / 6.0

    hue = jnp.where(r == vv, hr, jnp.where(g == vv, hg, hb))
    hue = hue * (range_ > 0)
    hue = hue + (hue < 0)

    return hue, sat, vv


def _hsv_to_rgb(h, s, v):
    """Converts H, S, V values to an R, G, B tuple.
    Reference TF implementation:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/adjust_saturation_op.cc
    Only input values between 0 and 1 are guaranteed to work properly, but this
    function complies with the TF implementation outside of this range.
    Args:
        h: A float tensor of arbitrary shape for the hue (0-1 values).
        s: A float tensor of the same shape for the saturation (0-1 values).
        v: A float tensor of the same shape for the value channel (0-1 values).
    Returns:
        An (r, g, b) tuple, each with the same dimension as the inputs.
    """
    c = s * v
    m = v - c
    dh = (h % 1.0) * 6.0
    fmodu = dh % 2.0
    x = c * (1 - jnp.abs(fmodu - 1))
    hcat = jnp.floor(dh).astype(jnp.int32)
    rr = jnp.where((hcat == 0) | (hcat == 5), c, jnp.where((hcat == 1) | (hcat == 4), x, 0)) + m
    gg = jnp.where((hcat == 1) | (hcat == 2), c, jnp.where((hcat == 0) | (hcat == 3), x, 0)) + m
    bb = jnp.where((hcat == 3) | (hcat == 4), c, jnp.where((hcat == 2) | (hcat == 5), x, 0)) + m
    return rr, gg, bb


def _adjust_brightness(rgb_tuple, delta):
    return jax.tree_map(lambda x: jnp.clip(x + delta, 0.0, 1.0), rgb_tuple)


def _adjust_contrast(rgb_tuple, factor):
    def _adjust_contrast_channel(channel):
        mean = jnp.mean(channel, axis=(-2, -1), keepdims=True)
        return jnp.clip(factor * (channel - mean) + mean, 0.0, 1.0)

    return jax.tree_map(_adjust_contrast_channel, rgb_tuple)


def _adjust_saturation(hsv_tuple, factor, delta):
    h, s, v = hsv_tuple
    return h, jnp.clip((s + delta) * factor, 0.0, 1.0), v


def _adjust_hue(hsv_tuple, delta):
    # Note: this method exactly matches TF"s adjust_hue (combined with the hsv/rgb
    # conversions) when running on GPU. When running on CPU, the results will be
    # different if all RGB values for a pixel are outside of the [0, 1] range.
    h, s, v = hsv_tuple
    return (h + delta) % 1.0, s, v
