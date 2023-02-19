from typing import Any, Callable, Optional, Sequence, Tuple

import jax
from jax import numpy as jnp
from jax import random
from jax.tree_util import tree_map
from chex import Array
from flax import linen as nn
import flax.linen.initializers as init
import distrax

from src.transformations.affine import transform_image


INV_SOFTPLUS_1 = jnp.log(jnp.exp(1) - 1.0)
# ^ this value is softplus^{-1}(1), i.e. if we get σ as softplus(σ_),
# and we init σ_ to this value, we effectively init σ to 1.
PRNGKey = Any


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
            h = self.act_fn(nn.Dense(hidden_dim, name=f"hidden{i}")(h))

        μ = nn.Dense(self.latent_dim, name="μ")(h)
        σ = jax.nn.softplus(nn.Dense(self.latent_dim, name="σ_")(h))

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

    @nn.compact
    def __call__(self, x: Array) -> Array:
        hidden_dims = self.hidden_dims if self.hidden_dims else [64, 32]

        h = x.reshape(-1)
        for i, hidden_dim in enumerate(hidden_dims):
            h = self.act_fn(nn.Dense(hidden_dim, name=f"hidden{i}")(h))

        y = nn.Dense(
            self.output_dim,
            kernel_init=init.zeros,
            bias_init=init.zeros,
            name="final",
        )(h)

        return y


# Adapted from https://github.com/deepmind/distrax/blob/master/examples/flow.py.
class Bijector(nn.Module):
    num_layers: int = 2
    num_bins: int = 4
    hidden_dims: Optional[Sequence[int]] = None

    @nn.compact
    def __call__(self, x: Array) -> distrax.BijectorLike:
        hidden_dims = self.hidden_dims if self.hidden_dims else [64, 128]

        def bijector_fn(params: Array):
            return distrax.RationalQuadraticSpline(params, range_min=0.0, range_max=1.0)

        # Number of parameters for the rational-quadratic spline:
        # - `num_bins` bin widths
        # - `num_bins` bin heights
        # - `num_bins + 1` knot slopes
        # for a total of `3 * num_bins + 1` parameters.
        num_bijector_params = 3 * self.num_bins + 1

        layers = []
        for i in range(self.num_layers):
            params = Conditioner(num_bijector_params, hidden_dims, name=f"cond{i}")(x)
            layers.append(bijector_fn(params))

        # We invert the flow so that the `forward` method is called with `log_prob`.
        bijector = distrax.Inverse(distrax.Chain(layers))
        return bijector


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
