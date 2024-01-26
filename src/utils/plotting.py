import math

import matplotlib.pyplot as plt
from jax import numpy as jnp


def rescale_for_imshow(img):
    return jnp.clip((img + 1.0) * 127.0 + 0.5, 0, 255).astype(jnp.uint8)


def put_in_grid(array, ncol=16, padding=2, pad_value=0.0, return_maps=False):
    array = jnp.asarray(array)

    if array.ndim == 4 and array.shape[-1] == 1:  # single-channel images
        array = jnp.concatenate((array, array, array), -1)

    # Make the mini-batch of images into a grid.
    nmaps = array.shape[0]
    xmaps = min(ncol, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(array.shape[1] + padding), int(array.shape[2] + padding)
    num_channels = array.shape[3]
    grid = jnp.full(
        (height * ymaps + padding, width * xmaps + padding, num_channels), pad_value
    ).astype(jnp.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid = grid.at[
                y * height + padding : (y + 1) * height,
                x * width + padding : (x + 1) * width,
            ].set(rescale_for_imshow(array[k]))
            k = k + 1

    if return_maps:
        return grid, xmaps, ymaps

    return grid


def plot_img_array(array, ncol=16, padding=2, pad_value=0.0, title=None, dpi=100):
    grid, xmaps, ymaps = put_in_grid(array, ncol, padding, pad_value, return_maps=True)

    fig = plt.figure(figsize=(2 * xmaps, 2 * ymaps), dpi=dpi)
    plt.imshow(grid)
    plt.axis("off")
    plt.tight_layout()
    if title is not None:
        plt.title(title, fontsize=32)
    plt.show()

    return fig
