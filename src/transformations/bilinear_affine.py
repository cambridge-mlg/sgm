import functools
import jax
import jax.numpy as jnp
from chex import Array, assert_shape, assert_rank

def compute_interpolation_weights(transformed_pt, width, height):
    x, y = transformed_pt
    x0 = jnp.floor(jax.lax.stop_gradient(x)).astype(jnp.int32)
    x1 = x0 + 1
    y0 = jnp.floor(jax.lax.stop_gradient(y)).astype(jnp.int32)
    y1 = y0 + 1

    w00 = (x1 - x) * (y1 - y)
    w01 = (x1 - x) * (y - y0)
    w10 = (x - x0) * (y1 - y)
    w11 = (x - x0) * (y - y0)
    # Check for out ofb ounds
    x0_oob = jnp.logical_or(x0 < 0, x0 >= width)
    x1_oob = jnp.logical_or(x1 < 0, x1 >= width)
    y0_oob = jnp.logical_or(y0 < 0, y0 >= height)
    y1_oob = jnp.logical_or(y1 < 0, y1 >= height)
    # return x0, x1, y0, y1, w00, w01, w10, w11
    return x0, x1, y0, y1, w00, w01, w10, w11, x0_oob, x1_oob, y0_oob, y1_oob



def bilinear_map_coordinate(
    img,  # shape [height, width]
    transformed_pt,  # shape [2]
    fill_value: float,
): # shape []
    height, width = img.shape
    # x0, x1, y0, y1, w00, w01, w10, w11 = compute_interpolation_weights(transformed_pt)
    (
        x0, x1, y0, y1, w00, w01, w10, w11, x0_oob, x1_oob, y0_oob, y1_oob
    ) = compute_interpolation_weights(transformed_pt, width, height)
    # Check for out ofb ounds
    # x0_oob = jnp.logical_or(x0 < 0, x0 >= width)
    # x1_oob = jnp.logical_or(x1 < 0, x1 >= width)
    # y0_oob = jnp.logical_or(y0 < 0, y0 >= height)
    # y1_oob = jnp.logical_or(y1 < 0, y1 >= height)

    f00 = jax.lax.select(
        jnp.logical_or(x0_oob, y0_oob),
        fill_value,
        img[y0, x0],
    )
    f01 = jax.lax.select(
        jnp.logical_or(x0_oob, y1_oob),
        fill_value,
        img[y1, x0]
    )
    f10 = jax.lax.select(
        jnp.logical_or(x1_oob, y0_oob),
        fill_value,
        img[y0, x1]
    )
    f11 = jax.lax.select(
        jnp.logical_or(x1_oob, y1_oob),
        fill_value,
        img[y1, x1],
    )

    interpolated_f = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11

    return interpolated_f, (x0, x1, y0, y1, w00, w01, w10, w11)


def bilinear_transform_image_with_affine_matrix(
    image: Array,
    T: Array,
    fill_value: float = -1.0,
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

    def get_transformed_pts(height, width, affine_mat):
        A = affine_mat[:2, :]

        # (x_t, y_t, 1), (1) in Jaderberg et al.
        x_t, y_t = jnp.meshgrid(jnp.linspace(-1, 1, width), jnp.linspace(-1, 1, height))
        ones = jnp.ones(x_t.size)
        input_pts = jnp.vstack([x_t.flatten(), y_t.flatten(), ones])  # shape [3, width * height]

        # (x_s, y_s) = A x (x_t, y_t, 1)^T
        transformed_pts = A @ input_pts  # shape [2, width * height]
        transformed_pts = (transformed_pts + 1) / 2
        transformed_pts = transformed_pts * jnp.array([width - 1, height - 1])[:, jnp.newaxis]
        return transformed_pts

    transformed_pts = get_transformed_pts(height, width, T)

    def bilinear_map_coordinates(
        img, # shape [height, width]
        transformed_pts,  # shape [2, width * height]
    ):  # shape [height * width]
        interpolated_f, *_ = jax.vmap(
            functools.partial(bilinear_map_coordinate, fill_value=fill_value),
            in_axes=(None, 1),
        )(img, transformed_pts)
        return interpolated_f
        
    output = jax.vmap(
        bilinear_map_coordinates,
        in_axes=(2, None),
    )(
        image,
        transformed_pts,
        # transformed_pts[::-1],
    )  # shape [height * width, num_channels]

    output = jnp.reshape(output, image.shape)
    return output
