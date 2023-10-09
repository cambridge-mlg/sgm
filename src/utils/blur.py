import jax
import jax.numpy as jnp
import chex


def _get_log_gaussian_kernel(sigma: float, filter_size):
    """Compute 1D Gaussian kernel."""
    x = jnp.arange(filter_size, dtype=jnp.float32) - ((filter_size - 1) / 2)
    y = -x**2 / (2.0 * (sigma**2))
    return y


def gaussian_filter2d(
    image,
    filter_shape: tuple[int, int] = (3, 3),
    sigma: tuple[float, float] | float = 1.0,
    constant_values: float = -1.,
):
    """Perform Gaussian blur on image(s).

    Args:
      image: Either a 2-D `Tensor` of shape `[height, width]`,
        a 3-D `Tensor` of shape `[height, width, channels]`,
        or a 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: `tuple` of 2 integers, specifying
        the height and width of the 2-D gaussian filter. Can be a single
        integer to specify the same value for all spatial dimensions.
      sigma: A `float` or `tuple`/`list` of 2 floats, specifying
        the standard deviation in x and y direction the 2-D gaussian filter.
        Can be a single float to specify the same value for all spatial
        dimensions. This is relative to the pixel-spacing of the image, i.e. if the image width
        is 64 and `sigma` is 32, the gaussian kernel will have standard deviation
        of length equal to half the image width.
    """
    chex.assert_rank(image, 3)

    sigma_tuple: tuple[float, float]
    if isinstance(sigma, tuple):
        if len(sigma) != 2:
            raise ValueError("sigma should be a float or a tuple/list of 2 floats")
        sigma_tuple = sigma
    elif isinstance(sigma, float):
        sigma_tuple = (sigma, sigma)
    else:
        raise TypeError
    
    # if any(filter_size % 2 == 0 for filter_size in filter_shape):
    #     raise ValueError("filter_shape should be odd integers, otherwise the filter will not have a center to convolve on, and the output image will be skewed")

    if any(filter_size <= 0 for filter_size in filter_shape):
        raise ValueError("filter_shape should be positive integers")

    if any(s < 0 for s in sigma_tuple):
        raise ValueError("sigma should be greater than or equal to 0.")

    log_gaussian_kernel_x = _get_log_gaussian_kernel(sigma_tuple[1], filter_shape[1])
    log_gaussian_kernel_x = log_gaussian_kernel_x[jnp.newaxis, :]

    log_gaussian_kernel_y = _get_log_gaussian_kernel(sigma_tuple[0], filter_shape[0])
    log_gaussian_kernel_y = log_gaussian_kernel_y[:, jnp.newaxis]

    log_gaussian_kernel = log_gaussian_kernel_x + log_gaussian_kernel_y

    gaussian_kernel = jnp.exp(log_gaussian_kernel)
    gaussian_kernel /= gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel[jnp.newaxis, jnp.newaxis, :, :]

    # Convolve the image with the gaussian kernel:
    out = jax.vmap(
        lambda img_channel: jax.lax.conv(
            jnp.transpose(img_channel[None, ..., None], [0, 3, 1, 2]),  # lhs = NCHW image tensor
            gaussian_kernel,  # rhs = OIHW conv kernel tensor
            (1, 1),  # window strides
            "SAME",  # padding mode
        ).squeeze(0).squeeze(0),
        in_axes=2,
        out_axes=-1,
    )(image - constant_values)
    return out + constant_values