import jax.numpy as jnp
from chex import Array, assert_rank, assert_shape

from src.transformations.affine import affine_transform_image, rotate_image
from src.transformations.color import (
    color_transform_image,
    hsv_transform_image,
    hue_transform_image,
    rgb_transform_image,
)

__all__ = [
    "affine_transform_image",
    "rotate_image",
    "color_transform_image",
    "hsv_transform_image",
    "hue_transform_image",
    "rgb_transform_image",
    "transform_image",
    "full_transform_image",
]


def transform_image(
    image: Array,
    η: Array,
    fill_mode: str = "constant",
    fill_value: float = -1.0,
    order: int = 3,
) -> Array:
    """Applies transformations to an image.

    Args:
        image: a rank-3 Array of shape (height, width, num channels) – i.e. Jax/TF image format.

        η: an Array with 5 entries:
        * η_0 controls translation in x.
        * η_1 controls translation in y.
        * η_2 is the angle of rotation.
        * η_3 is the scaling factor in x.
        * η_4 is the scaling factor in y.

    Returns:
        A transformed image of same shape as the input.
    """
    assert_rank(image, 3)
    assert_shape(η, (5,))

    # Apply affine transformations
    η_affine = jnp.concatenate((η, jnp.zeros(1, dtype=η.dtype)))
    image = affine_transform_image(
        image,
        η_affine,
        fill_mode=fill_mode,
        fill_value=fill_value,
        order=order,
    )

    return image


def full_transform_image(
    image: Array,
    η: Array,
    fill_mode: str = "constant",
    fill_value: float = -1.0,
) -> Array:
    """Applies transformations to an image.

    Args:
        image: a rank-3 Array of shape (height, width, num channels) – i.e. Jax/TF image format.

        η: an Array with 5 entries:
        * η_0 controls translation in x.
        * η_1 controls translation in y.
        * η_2 is the angle of rotation.
        * η_3 is the scaling factor in x.
        * η_4 is the scaling factor in y.
        * η_5 is the shearing.
        * η_6 is the hue.
        * η_7 is the saturation.

    Returns:
        A transformed image of same shape as the input.
    """
    assert_rank(image, 3)
    assert_shape(η, (8,))

    # Apply affine transformations
    image = affine_transform_image(
        image, η[:6], fill_mode=fill_mode, fill_value=fill_value
    )

    # Apply color transformations
    image = hsv_transform_image(image, η[6:])

    return image
