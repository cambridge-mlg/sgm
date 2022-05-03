"""Affine transformations of images."""
import jax
from jax import numpy as jnp
from jax.scipy.linalg import expm
from chex import Array, assert_shape, assert_rank, assert_equal_shape


def create_generator_matrices() -> Array:
    """Creates the generator matrices for affine transformations.
    See https://en.wikipedia.org/wiki/Affine_transformation#Image_transformation.

    Args:
        None

    Returns:
        A 6x3x3 array containing the 6 3x3 generator matrices, in order:
        translation in x, translation in y, rotation, scale in x, scale in y, and shearing.
    """
    G_trans_x = jnp.zeros((3, 3), jnp.float32).at[0, 2].set(1)
    G_trans_y = jnp.zeros((3, 3), jnp.float32).at[1, 2].set(1)
    G_rot = jnp.zeros((3, 3), jnp.float32).at[1, 0].set(1).at[0, 1].set(-1)
    G_scale_x = jnp.zeros((3, 3), jnp.float32).at[0, 0].set(1)
    G_scale_y = jnp.zeros((3, 3), jnp.float32).at[1, 1].set(1)
    G_shear = jnp.zeros((3, 3), jnp.float32).at[1, 0].set(1).at[0, 1].set(1)
    Gs = jnp.array([G_trans_x, G_trans_y, G_rot, G_scale_x, G_scale_y, G_shear])

    return Gs


def gen_transform_mat(
    η: Array,
    ε: Array,
) -> Array:
    """Generates an affine transformation matrix which can be used to translate,
    rotate, scale, and shear a 2D image.

    See App. E of "Learning Invariant Weights in Neural Networks" by van der
    Ouderaa and van der Wilk.

    Args:
        η: an Array with six entries:
        * η_0 controls translation in x.
        * η_1 controls translation in y.
        * η_2 is the angle of rotation.
        * η_3 is the scaling factor in x.
        * η_4 is the scaling factor in y.
        * η_5 controls shearing.

        ε: an Array of the same shape as `η` which is used for
        sampling transformations via a reparameterisation trick (ε ~ U[-1, 1]^6)

    Returns:
        A 3x3 affine transformation array.
    """
    assert_equal_shape([η, ε])
    assert_shape(η, (6,))

    Gs = create_generator_matrices()

    T = expm((ε[:, jnp.newaxis, jnp.newaxis] * η[:, jnp.newaxis, jnp.newaxis] * Gs).sum(axis=0))

    return T


def transform_image(
    image: Array,
    T: Array
) -> Array:
    """Applies an affine transformation to an image.

    See Sec 3.2 in "Spatial Transformer Networks" by Jaderberg et al.

    Args:
        image: a rank-3 Array of shape (num channels, height, width).

        T: a 3x3 affine transformation Array.

    Returns:
        A transformed image of same shape as the input.
    """
    assert_rank(image, 3)
    assert_shape(T, (3, 3))

    num_channels, height, width = image.shape
    A = T[:2, :]

    # (x_t, y_t, 1), eq (1) in Jaderberg et al.
    x_t, y_t = jnp.meshgrid(jnp.linspace(-1, 1, width),
                            jnp.linspace(-1, 1, height))
    ones = jnp.ones(x_t.size)
    input_pts = jnp.vstack([x_t.flatten(), y_t.flatten(), ones])

    # (x_s, y_s) = A x (x_t, y_t, 1)^T
    transformed_pts = A @ input_pts
    transformed_pts = (transformed_pts + 1) / 2
    transformed_pts = transformed_pts * jnp.array([[width], [height]])

    # Transform the image by moving the pixels to their new locations
    output = jnp.stack(
        [
            jax.scipy.ndimage.map_coordinates(image[i], transformed_pts[::-1], order=1, cval=0)
            # Note: usually we would use bicubic interpolation (order=3), but this isn't available
            # in jax, so we have to use linear interpolation.
            for i in range(num_channels)
        ]
    )
    output = jnp.reshape(output, (num_channels, height, width))
    return output
