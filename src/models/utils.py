from typing import Optional, Tuple

import jax
from jax import numpy as jnp
from jax import random
from chex import Array, PRNGKey, assert_rank, assert_equal_shape
import distrax
import optax

from src.transformations import transform_image


def reset_metrics(state):
    return state.replace(metrics=state.metrics.empty())


def transform_η(η: Array, bounds: Array, offset: Optional[Array] = None) -> Array:
    """Converts η to the correct range.

    Args:
        η: a rank-1 array of length 7.
        bounds: a rank-1 array of length 7.
        offset: an optional rank-1 array of length 7.
    """
    assert_rank(η, 1)
    assert_rank(bounds, 1)
    assert_equal_shape(η, bounds)
    if offset is not None:
        assert_rank(offset, 1)
        assert_equal_shape(η, offset)
    else:
        offset = jnp.zeros_like(bounds)

    η = jax.nn.tanh(η) * bounds + offset

    return η


def approximate_mode(
    distribution: distrax.Distribution, num_samples: int, rng: PRNGKey
) -> Array:
    """Approximates the mode of a distribution by taking a number of samples and returning the most likely.

    Args:
        distribution: A distribution.
        num_samples: The number of samples to take.
        rng: A PRNG key.

    Returns:
        An approximate mode.
    """
    samples, log_probs = distribution.sample_and_log_prob(
        seed=rng, sample_shape=(num_samples,)
    )
    return samples[jnp.argmax(log_probs)]


def make_approx_invariant(
    p_Z_given_X, x: Array, num_samples: int, rng: PRNGKey, bounds: Tuple[float, ...]
) -> distrax.Normal:
    """Construct an approximately invariant distribution by sampling transformations then averaging.

    Args:
        p_Z_given_X: A distribution whose parameters are a function of x.
        x: An image.
        num_samples: The number of samples to use for the approximation.
        rng: A random number generator.

    Returns:
        An approximately invariant distribution of the same type as p.
    """
    p_Η = distrax.Uniform(low=-1 * jnp.array(bounds), high=jnp.array(bounds))
    rngs = random.split(rng, num_samples)
    # TODO: investigate scaling of num_samples with the size of η.

    # TODO: this function is not aware of the bounds for η.
    def sample_params(x, rng):
        η = p_Η.sample(seed=rng)
        x_ = transform_image(x, -η)
        p_Z_given_x_ = p_Z_given_X(x_)
        assert type(p_Z_given_x_) == distrax.Normal
        # TODO: generalise to other distributions.
        return p_Z_given_x_.loc, p_Z_given_x_.scale

    params = jax.vmap(sample_params, in_axes=(None, 0))(x, rngs)
    params = jax.tree_map(lambda x: jnp.mean(x, axis=0), params)

    return distrax.Normal(*params)


def clipped_adamw(learning_rate, norm, weight_decay: float = 1e-4):
    return optax.chain(
        optax.clip_by_global_norm(norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )


def huber_loss(target: float, pred: float, slope: float = 1.0, radius: float = 1.0) -> float:
    """Huber loss. Separate out delta (which normally controls both the slope of the linear behaviour, 
    and the radius of the quadratic behaviour) into 2 separate terms.

    Args:
        target: ground truth
        pred: predictions
        slope: slope of linear behaviour
        radius: radius of quadratic behaviour

    Returns:
        loss value

    References:
        https://en.wikipedia.org/wiki/Huber_loss
    """
    abs_diff = jnp.abs(target - pred)
    return jnp.where(
        abs_diff > radius,
        slope * abs_diff - 0.5 * slope * radius,
        (0.5 * slope / radius) * abs_diff ** 2,
    )
