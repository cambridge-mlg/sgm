from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
import jax
from jax import numpy as jnp
from jax import random
from jax.tree_util import tree_map
from chex import Array, assert_rank, assert_shape
from flax import linen as nn
import flax.linen.initializers as init
import distrax

from src.transformations.affine import transform_image


INV_SOFTPLUS_1 = jnp.log(jnp.exp(1) - 1.0)
# ^ this value is softplus^{-1}(1), i.e. if we get σ as softplus(σ_),
# and we init σ_ to this value, we effectively init σ to 1.
PRNGKey = Any
KwArgs = Mapping[str, Any]


class DenseEncoder(nn.Module):
    """p(z|x) = N(μ(x), σ(x)), where μ(x) and σ(x) are dense neural networks."""

    latent_dim: int
    hidden_dims: Optional[Sequence[int]] = None
    act_fn: Callable = nn.relu
    σ_min: float = 1e-2

    @nn.compact
    def __call__(self, x: Array) -> distrax.Normal:
        hidden_dims = self.hidden_dims if self.hidden_dims else [64, 32]

        h = x.reshape(-1)
        for i, hidden_dim in enumerate(hidden_dims):
            h = self.act_fn(nn.Dense(hidden_dim, name=f"hidden_{i}")(h))

        # We initialize these dense layers so that we get μ=0 and σ=1 at the start.
        μ = nn.Dense(self.latent_dim, kernel_init=init.zeros, bias_init=init.zeros, name="μ")(h)
        σ = jax.nn.softplus(
            nn.Dense(
                self.latent_dim,
                kernel_init=init.zeros,
                bias_init=init.constant(INV_SOFTPLUS_1),
                name="σ_",
            )(h)
        )

        return distrax.Normal(loc=μ, scale=σ.clip(min=self.σ_min))


class ConvEncoder(nn.Module):
    """p(z|x) = N(μ(x), σ(x)), where μ(x) and σ(x) are convolutional neural networks."""

    latent_dim: int
    hidden_dims: Optional[Sequence[int]] = None
    act_fn: Callable = nn.relu
    norm_cls: nn.Module = nn.LayerNorm
    σ_min: float = 1e-2

    @nn.compact
    def __call__(self, x: Array) -> distrax.Normal:
        hidden_dims = self.hidden_dims if self.hidden_dims else [64, 128, 256]

        h = x
        for i, hidden_dim in enumerate(hidden_dims):
            h = nn.Conv(
                hidden_dim,
                kernel_size=(3, 3),
                strides=(2, 2) if i == 0 else (1, 1),
                name=f"hidden{i}",
            )(h)
            h = self.norm_cls(name=f"norm{i}")(h)
            h = self.act_fn(h)

        h = h.flatten()

        # TODO: should we init these to 0 and 1?
        μ = nn.Dense(self.latent_dim, name=f"μ")(h)
        σ = jax.nn.softplus(nn.Dense(self.latent_dim, name="σ_")(h))

        return distrax.Normal(loc=μ, scale=σ.clip(min=self.σ_min))


class ConvDecoder(nn.Module):
    """p(x|z) = N(μ(z), σ), where μ(z) is a convolutional neural network."""

    image_shape: Tuple[int, int, int]
    hidden_dims: Optional[Sequence[int]] = None
    σ_init: Callable = init.constant(INV_SOFTPLUS_1)
    σ_min: float = 1e-2
    act_fn: Callable = nn.relu
    norm_cls: nn.Module = nn.LayerNorm

    @nn.compact
    def __call__(self, z: Array) -> distrax.Normal:
        hidden_dims = self.hidden_dims if self.hidden_dims else [256, 128, 64]

        assert self.image_shape[0] == self.image_shape[1], "Images should be square."
        output_size = self.image_shape[0]
        first_hidden_size = output_size // 2

        h = nn.Dense(first_hidden_size * first_hidden_size * hidden_dims[0], name=f"resize")(z)
        h = h.reshape(first_hidden_size, first_hidden_size, hidden_dims[0])

        for i, hidden_dim in enumerate(hidden_dims):
            h = nn.ConvTranspose(
                hidden_dim,
                kernel_size=(3, 3),
                strides=(2, 2) if i == 0 else (1, 1),
                name=f"hidden{i}",
            )(h)
            h = self.norm_cls(name=f"norm{i}")(h)
            h = self.act_fn(h)

        μ = nn.Conv(self.image_shape[-1], kernel_size=(3, 3), strides=(1, 1), name=f"μ")(h)
        σ = jax.nn.softplus(self.param("σ_", self.σ_init, self.image_shape))

        return distrax.Normal(loc=μ, scale=σ.clip(min=self.σ_min))


# Adapted from https://github.com/deepmind/distrax/blob/master/examples/flow.py.
class Conditioner(nn.Module):
    """A neural network that predicts the parameters of a flow given an input."""

    output_dim: int
    hidden_dims: Optional[Sequence[int]] = None
    act_fn: Callable = nn.relu
    dropout_rate: float = 0.1
    train: Optional[bool] = None

    @nn.compact
    def __call__(self, x: Array, train: Optional[bool] = None) -> Array:
        train = nn.merge_param("train", self.train, train)
        if train is None:
            train = False
        # TODO: we probably shouldn't have a default here, but this would break existing code.

        hidden_dims = self.hidden_dims if self.hidden_dims else [64, 32]

        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        h = x.reshape(-1)
        for i, hidden_dim in enumerate(hidden_dims):
            h = self.act_fn(nn.Dense(hidden_dim, name=f"hidden{i}")(h))

        # We initialize this dense layer to zero so that the flow is initialized to the identity function.
        y = nn.Dense(
            self.output_dim,
            kernel_init=init.zeros,
            bias_init=init.zeros,
            name="final",
        )(h)

        return y


# Adapted from https://github.com/deepmind/distrax/blob/master/examples/flow.py.
class Flow(nn.Module):
    event_shape: Sequence[int]
    num_layers: int = 2
    num_bins: int = 4
    cond_hidden_dims: Optional[Sequence[int]] = None
    bounds_array: Optional[Array] = None
    dropout_rate: float = 0.1
    base: Optional[KwArgs] = None

    @nn.compact
    def __call__(self, x: Array, train=False) -> distrax.Transformed:
        cond_hidden_dims = self.cond_hidden_dims if self.cond_hidden_dims else [64, 128]

        mask = jnp.arange(np.prod(self.event_shape)) % 2
        mask = jnp.reshape(mask, self.event_shape)
        mask = mask.astype(bool)

        def bijector_fn(params: Array):
            return distrax.RationalQuadraticSpline(params, range_min=-2.0, range_max=2.0)

        # Number of parameters for the rational-quadratic spline:
        # - `num_bins` bin widths
        # - `num_bins` bin heights
        # - `num_bins + 1` knot slopes
        # for a total of `3 * num_bins + 1` parameters.
        num_bijector_params = 3 * self.num_bins + 1

        layers = [distrax.Block(distrax.Lambda(lambda x: x), len(self.event_shape))]
        for i in range(self.num_layers):
            layer = distrax.MaskedCoupling(
                mask=mask,
                bijector=bijector_fn,
                conditioner=Conditioner(
                    num_bijector_params,
                    cond_hidden_dims,
                    dropout_rate=self.dropout_rate,
                    train=train,
                    name=f"cond_{i}",
                ),
            )
            layers.append(layer)

            mask = jnp.logical_not(mask)

        bijector = distrax.Chain(
            [
                distrax.Block(
                    distrax.ScalarAffine(
                        shift=jnp.zeros_like(self.bounds_array), scale=self.bounds_array
                    ),
                    len(self.event_shape),
                ),
                distrax.Block(distrax.Tanh(), len(self.event_shape)),
                # We invert the flow so that the `forward` method is called with `log_prob`.
                distrax.Inverse(distrax.Chain(layers)),
            ]
        )

        base = distrax.Independent(
            DenseEncoder(
                latent_dim=len(self.bounds_array),
                **(self.base or {}),
            )(x),
            reinterpreted_batch_ndims=len(self.event_shape),
        )

        return distrax.Transformed(base, bijector)


def make_approx_invariant(
    p_Z_given_X: ConvEncoder,
    x: Array,
    num_samples: int,
    rng: PRNGKey,
    bounds: Tuple[float, float, float, float, float, float, float],
    α: float = 1.0,
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
    p_Η = distrax.Uniform(low=-1 * jnp.array(bounds) * α, high=jnp.array(bounds) * α)
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
    params = tree_map(lambda x: jnp.mean(x, axis=0), params)

    return distrax.Normal(*params)


def make_η_bounded(η: Array, bounds: Array):
    """Converts η to a bounded representation.

    Args:
        η: a rank-1 array of length 7.
        bounds: a rank-1 array of length 7.
    """
    assert_rank(η, 1)
    assert_shape(η, (7,))
    assert_rank(bounds, 1)
    assert_shape(bounds, (7,))

    # η = bounds * jnp.sin(η * 0.5 * (jnp.pi + 1e-8) / (bounds + 1e-8))
    # η = η.clip(-bounds, bounds)
    η = jax.nn.tanh(η) * bounds

    return η


def approximate_mode(distribution: distrax.Distribution, num_samples: int, rng: PRNGKey) -> Array:
    """Approximates the mode of a distribution by taking a number of samples and returning the most likely.

    Args:
        distribution: A distribution.

    Returns:
        An approximate mode.
    """
    samples, log_probs = distribution.sample_and_log_prob(seed=rng, sample_shape=(num_samples,))
    return samples[jnp.argmax(log_probs)]
