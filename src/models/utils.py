import distrax
import optax
from chex import Array, PRNGKey
from jax import numpy as jnp


def reset_metrics(state):
    return state.replace(metrics=state.metrics.empty())


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


def clipped_adamw(learning_rate, norm, weight_decay: float = 1e-4):
    return optax.chain(
        optax.clip_by_global_norm(norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )


def huber_loss(
    target: float, pred: float, slope: float = 1.0, radius: float = 1.0
) -> float:
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
        (0.5 * slope / radius) * abs_diff**2,
    )
