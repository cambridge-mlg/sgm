"""Affine transformations of images."""
import functools
from jax import numpy as jnp
import jax
from jax.scipy.linalg import expm
from chex import Array, assert_shape, assert_rank

from src.transformations.map_coords import map_coordinates


def create_generator_matrices() -> Array:
    """Creates the generator matrices for affine transformations.
    See https://en.wikipedia.org/wiki/Affine_transformation#Image_transformation.

    Args:
        None

    Returns:
        A 6x3x3 array containing the 6 3x3 generator matrices, in order:
        translation in x, translation in y, rotation, scale in x, scale in y, shearing in x, and shearing in y.
    """
    G_trans_x = jnp.zeros((3, 3), jnp.float32).at[0, 2].set(1)
    G_trans_y = jnp.zeros((3, 3), jnp.float32).at[1, 2].set(1)
    G_rot = jnp.zeros((3, 3), jnp.float32).at[1, 0].set(1).at[0, 1].set(-1)
    G_scale_x = jnp.zeros((3, 3), jnp.float32).at[0, 0].set(1)
    G_scale_y = jnp.zeros((3, 3), jnp.float32).at[1, 1].set(1)
    G_shear = jnp.zeros((3, 3), jnp.float32).at[0, 1].set(1).at[1, 0].set(1)
    Gs = jnp.array([G_trans_x, G_trans_y, G_rot, G_scale_x, G_scale_y, G_shear])

    return Gs


def gen_affine_matrix(
    η: Array,
) -> Array:
    """Generates an affine transformation matrix which can be used to translate,
    rotate, scale, and shear a 2D image.

    See App. E of "Learning Invariant Weights in Neural Networks" by van der
    Ouderaa and van der Wilk.

    Args:
        η: an Array with 6 entries:
        * η_0 controls translation in x.
        * η_1 controls translation in y.
        * η_2 is the angle of rotation.
        * η_3 is the scaling factor in x.
        * η_4 is the scaling factor in y.
        * η_5 controls shearing in x and y.

    Returns:
        A 3x3 affine transformation array.
    """
    assert_shape(η, (6,))

    Gs = create_generator_matrices()

    T = expm((η[:, jnp.newaxis, jnp.newaxis] * Gs).sum(axis=0))

    return T


def gen_affine_matrix_no_shear(
    η: Array,
) -> Array:
    """Generates an affine transformation matrix which can be used to translate,
    rotate and scale a 2D image.

    See App. E of "Learning Invariant Weights in Neural Networks" by van der
    Ouderaa and van der Wilk.

    Args:
        η: an Array with 5 entries:
        * η_0 controls translation in x.
        * η_1 controls translation in y.
        * η_2 is the angle of rotation.
        * η_3 is the scaling factor in x.
        * η_4 is the scaling factor in y.

    Returns:
        A 3x3 affine transformation array.
    """
    assert_shape(η, (5,))
    return gen_affine_matrix(
        jnp.concatenate((η, jnp.zeros(1, dtype=η.dtype)))
    )


def transform_image_with_affine_matrix(
    image: Array,
    T: Array,
    fill_mode: str = "constant",
    fill_value: float = -1.0,
    order: int = 3,
) -> Array:
    """Applies an affine transformation to an image.

    See Sec 3.2 in "Spatial Transformer Networks" by Jaderberg et al.

    Args:
        image: a rank-3 Array of shape (height, width, num channels) – i.e. Jax/TF image format.

        T: a 3x3 affine transformation Array.

    Returns:
        A transformed image of same shape as the input.
    """
    assert_rank(image, 3)
    assert_shape(T, (3, 3))

    height, width, num_channels = image.shape
    A = T[:2, :]

    # (x_t, y_t, 1), (1) in Jaderberg et al.
    x_t, y_t = jnp.meshgrid(jnp.linspace(-1, 1, width), jnp.linspace(-1, 1, height))
    ones = jnp.ones(x_t.size)
    input_pts = jnp.vstack([x_t.flatten(), y_t.flatten(), ones])

    # (x_s, y_s) = A x (x_t, y_t, 1)^T
    transformed_pts = A @ input_pts
    transformed_pts = (transformed_pts + 1) / 2
    transformed_pts = transformed_pts * jnp.array([[width - 1], [height - 1]])

    # Transform the image by moving the pixels to their new locations
    output = jnp.stack(
        [
            map_coordinates(
                image[:, :, i],
                transformed_pts[::-1],
                order=order,
                mode=fill_mode,
                cval=fill_value,
            )
            for i in range(num_channels)
        ],
        axis=-1,
    )
    output = jax.vmap(
        functools.partial(
            map_coordinates,
            order=order,
            mode=fill_mode,
            cval=fill_value,
        ),
        in_axes=(2, None),
        out_axes=1,
    )(
        image,
        transformed_pts[::-1],
    )  # shape [height * width, num_channels]
    output = jnp.reshape(output, image.shape)

    return output


def affine_transform_image(
    image: Array,
    η: Array,
    fill_mode: str = "constant",
    fill_value: float = -1.0,
    order: int = 3,
) -> Array:
    """Applies an affine transformation to an image.

    See Sec 3.2 in "Spatial Transformer Networks" by Jaderberg et al. and
    App. E of "Learning Invariant Weights in Neural Networks" by van der
    Ouderaa and van der Wilk.

    Args:
        image: a rank-3 Array of shape (height, width, num channels) – i.e. Jax/TF image format.

        η: an Array with 6 entries:
        * η_0 controls translation in x.
        * η_1 controls translation in y.
        * η_2 is the angle of rotation.
        * η_3 is the scaling factor in x.
        * η_4 is the scaling factor in y.
        * η_5 controls shearing in x and y.

    Returns:
        A transformed image of same shape as the input.
    """
    assert_rank(image, 3)
    assert_shape(η, (6,))

    T = gen_affine_matrix(η)
    return transform_image_with_affine_matrix(image, T, fill_mode=fill_mode, fill_value=fill_value, order=order)


def rotate_image(
    image: Array,
    θ: float,
    fill_value: float = 0.0,
) -> Array:
    """Rotates an image by an angle θ.

    Args:
        image: a rank-3 Array of shape (height, width, num channels) – i.e. Jax/TF image format.

        θ: the angle of rotation in radians.

    Returns:
        A rotated image of same shape as the input.
    """
    η = jnp.array([0, 0, θ, 0, 0, 0, 0])
    return affine_transform_image(image, η, fill_value=fill_value)
