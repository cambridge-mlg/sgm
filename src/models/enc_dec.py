from typing import Callable, List, Optional, Union
from math import prod

import jax
import jax.numpy as jnp
from flax import linen as nn
import flax.linen.initializers as init
import tensorflow_probability.substrates.jax.distributions as dists


_LIKELIHOODS = [
    'iso-normal',  # Homoskedastic (i.e. not input dependant) isotropic Guassian.
    'unit-iso-normal',  # unit-variance isotropic Guassian.
    'diag-normal',  # Homoskedastic diagonal Guassian.
    'hetero-diag-normal',  # Heteroskedastic (i.e., predicted per input by an NN) isotropic Guassian.
    'hetero-iso-normal',  # Heteroskedastic diagonal Guassian.
    'bernoulli',
    # TODO: support Categorical.
]


_POSTERIORS = [
    'hetero-diag-normal',
    # TODO: support Full Cov Normal.
]


def _get_act_fn(act_fn):
    if isinstance(act_fn, str):
        return getattr(nn, act_fn)
    else:
        return act_fn


class FCEncoder(nn.Module):
    latent_dim: int
    posterior: str = 'hetero-diag-normal'
    hidden_dims: Optional[List[int]] = None
    act_fn: Union[Callable, str] = nn.relu

    @nn.compact
    def __call__(self, x, train):
        if self.posterior not in _POSTERIORS:
            msg = f'`self.posterior` should be one of `{_POSTERIORS}` but was `{self.posterior}` instead.'
            raise RuntimeError(msg)

        if self.hidden_dims is None:
            self.hidden_dims = [500,]

        act_fn = _get_act_fn(self.act_fn)

        h = x
        for i, hidden_dim in enumerate(self.hidden_dims):
            h = act_fn(nn.Dense(hidden_dim, name=f'hidden{i}')(h))

        μ = nn.Dense(self.latent_dim, name=f'μ')(h)
        σ = jax.nn.softplus(nn.Dense(self.latent_dim, name=f'σ')(h))

        return dists.Normal(loc=μ, scale=σ)


class FCDecoder(nn.Module):
    image_shape: int
    likelihood: str = 'iso-normal'
    hidden_dims: Optional[List[int]] = None
    act_fn: Union[Callable, str] = nn.relu
    σ_init: Callable = init.constant(jnp.log(jnp.exp(1) - 1.))
    # ^ this value is softplus^{-1}(1), i.e., σ starts at 1.
    σ_min: float = 1e-2

    @nn.compact
    def __call__(self, z, train):
        if self.likelihood not in _LIKELIHOODS:
            msg = f'`self.likelihood` should be one of `{_LIKELIHOODS}` but was `{self.likelihood}` instead.'
            raise RuntimeError(msg)

        if self.hidden_dims is None:
            self.hidden_dims = [500,]

        act_fn = _get_act_fn(self.act_fn)

        output_dim = prod(self.image_shape)

        h = z
        for i, hidden_dim in enumerate(self.hidden_dims):
            h = act_fn(nn.Dense(hidden_dim, name=f'hidden{i}')(h))

        if self.likelihood == 'bernoulli':
            logits = nn.Dense(output_dim, name=f'logits')(h)

            return dists.Bernoulli(logits=logits)

        else:
            μ = nn.Dense(output_dim, name=f'μ')(h)

            if 'iso' in self.likelihood:
                σ_size = 1
            else:
                σ_size = output_dim

            if 'hetero' in self.likelihood:
                σ = jax.nn.softplus(nn.Dense(σ_size, name=f'σ_')(h))
            else:
                σ = jax.nn.softplus(self.param(
                    'σ_',
                    self.σ_init,
                    (σ_size,)
                ))
                if 'unit' in self.likelihood:
                    σ = jax.lax.stop_gradient(σ)

            return dists.Normal(loc=μ, scale=σ.clip(min=self.σ_min))


class ConvEncoder(nn.Module):
    latent_dim: int
    posterior: str = 'hetero-diag-normal'
    hidden_dims: Optional[List[int]] = None
    act_fn: Union[Callable, str] = nn.relu
    norm_cls: nn.Module = nn.LayerNorm

    @nn.compact
    def __call__(self, x, train):
        if self.posterior not in _POSTERIORS:
            msg = f'`self.posterior` should be one of `{_POSTERIORS}` but was `{self.posterior}` instead.'
            raise RuntimeError(msg)

        if self.hidden_dims is None:
            self.hidden_dims = [64, 128]

        act_fn = _get_act_fn(self.act_fn)

        h = x
        for i, hidden_dim in enumerate(self.hidden_dims):
            h = nn.Conv(
                hidden_dim,
                kernel_size=(3, 3),
                strides=(2, 2) if i==0 else (1, 1),
                name=f'hidden{i}'
            )(h)
            h = self.norm_cls(name=f'norm{i}')(h)
            h = act_fn(h)

        h = h.flatten()

        μ = nn.Dense(self.latent_dim, name=f'μ')(h)
        σ = jax.nn.softplus(nn.Dense(self.latent_dim, name='σ')(h))

        return dists.Normal(loc=μ, scale=σ)


class ConvDecoder(nn.Module):
    image_shape: int
    likelihood: str = 'iso-normal'
    hidden_dims: Optional[List[int]] = None
    act_fn: Union[Callable, str] = nn.relu
    σ_init: Callable = init.constant(jnp.log(jnp.exp(1) - 1.))
    # ^ this value is softplus^{-1}(1), i.e., σ starts at 1.
    σ_min: float = 1e-2
    norm_cls: nn.Module = nn.LayerNorm

    @nn.compact
    def __call__(self, z, train):
        if self.likelihood not in _LIKELIHOODS:
            msg = f'`self.likelihood` should be one of `{_LIKELIHOODS}` but was `{self.likelihood}` instead.'
            raise RuntimeError(msg)

        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]

        assert self.image_shape[0] == self.image_shape[1], "Images should be square."
        output_size = self.image_shape[0]
        first_hidden_size = output_size // 2

        act_fn = _get_act_fn(self.act_fn)

        h = nn.Dense(
            first_hidden_size * first_hidden_size * self.hidden_dims[0], name=f'resize'
        )(z)
        h = h.reshape(first_hidden_size, first_hidden_size, self.hidden_dims[0])

        for i, hidden_dim in enumerate(self.hidden_dims):
            h = nn.ConvTranspose(
                hidden_dim,
                kernel_size=(3, 3),
                strides=(2, 2) if i == 0 else (1, 1),
                name=f'hidden{i}'
            )(h)
            h = self.norm_cls(name=f'norm{i}')(h)
            h = act_fn(h)

        if self.likelihood == 'bernoulli':
            logits = nn.Conv(
                self.image_shape[-1],
                kernel_size=(3, 3),
                strides=(1, 1),
                name=f'logits'
            )(h)

            return dists.Bernoulli(logits=logits)

        else:
            μ = nn.Conv(self.image_shape[-1], (3, 3), 1, name=f'μ')(h)

            if 'hetero' in self.likelihood:
                if not 'iso' in self.likelihood:
                    σ = jax.nn.softplus(nn.Conv(
                            self.image_shape[-1],
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            name=f'σ_'
                        )(h))
                else:
                    σ = jax.nn.softplus(nn.Dense(1, name=f'σ_')(h))

            else:
                σ = jax.nn.softplus(self.param(
                    'σ_',
                    self.σ_init,
                    (1,) if 'iso' in self.likelihood else self.image_shape
                ))
                if 'unit' in self.likelihood:
                    σ = jax.lax.stop_gradient(σ)

            return dists.Normal(loc=μ, scale=σ.clip(min=self.σ_min))


_convnext_initializer = nn.initializers.variance_scaling(
    0.2, "fan_in", distribution="truncated_normal"
)

class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block.

    Adapted from: https://github.com/DarshanDeshpande/jax-models/blob/ae5750540f142572ff7f276b927a9cdb5195fd23/jax_models/models/convnext.py#L5
    """
    dim: int = 64
    layer_scale_init_value: float = 1e-6

    @nn.compact
    def __call__(self, inputs):
        x = nn.Conv(
            self.dim,
            kernel_size=(7, 7),
            feature_group_count=self.dim,
            kernel_init=_convnext_initializer,
            name="dw3x3conv"
        )(inputs)
        x = nn.LayerNorm(name="norm")(x)
        x = nn.Dense(
            4 * self.dim,
            kernel_init=_convnext_initializer,
            name="pw1x1conv1"
        )(x)
        x = nn.gelu(x)
        x = nn.Dense(
            self.dim,
            kernel_init=_convnext_initializer,
            name="pw1x1conv2"
        )(x)
        if self.layer_scale_init_value > 0:
            gamma = self.param(
                "gamma", init.constant(self.layer_scale_init_value), (self.dim,),
            )
            x = gamma * x

        x = inputs + x
        # NOTE: ^ this should have DropPath if it was being used in an actual ConvNeXt
        return x


class ConvNeXtEncoder(nn.Module):
    latent_dim: int
    posterior: str = 'hetero-diag-normal'
    hidden_dims: Optional[List[int]] = None

    @nn.compact
    def __call__(self, x, train):
        if self.posterior not in _POSTERIORS:
            msg = f'`self.posterior` should be one of `{_POSTERIORS}` but was `{self.posterior}` instead.'
            raise RuntimeError(msg)

        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 256, 512]

        h = x
        for i, hidden_dim in enumerate(self.hidden_dims):
            h = nn.Conv(
                hidden_dim,
                kernel_size=(2, 2),
                strides=(2, 2) if i == 0 else (1, 1),
                kernel_init=_convnext_initializer,
                name=f"downsample_conv{i}",
            )(h)
            h = nn.LayerNorm(name=f"downsample_norm{i}")(h)

            h = ConvNeXtBlock(hidden_dim)(h)
            # TODO: ^ maybe we want more of these?


        h = h.flatten()

        μ = nn.Dense(self.latent_dim, name=f'μ')(h)
        σ = jax.nn.softplus(nn.Dense(self.latent_dim, name='σ')(h))

        return dists.Normal(loc=μ, scale=σ)


class ConvNeXtDecoder(nn.Module):
    image_shape: int
    likelihood: str = 'iso-normal'
    hidden_dims: Optional[List[int]] = None
    σ_init: Callable = init.constant(jnp.log(jnp.exp(1) - 1.))
    # ^ this value is softplus^{-1}(1), i.e., σ starts at 1.
    σ_min: float = 1e-2

    @nn.compact
    def __call__(self, z, train):
        if self.likelihood not in _LIKELIHOODS:
            msg = f'`self.likelihood` should be one of `{_LIKELIHOODS}` but was `{self.likelihood}` instead.'
            raise RuntimeError(msg)

        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128, 64]

        assert self.image_shape[0] == self.image_shape[1], "Images should be square."
        output_size = self.image_shape[0]
        first_hidden_size = output_size // 2

        h = nn.Dense(first_hidden_size * first_hidden_size * self.hidden_dims[0], name=f'resize')(z)
        h = h.reshape(first_hidden_size, first_hidden_size, self.hidden_dims[0])

        for i, hidden_dim in enumerate(self.hidden_dims):
            h = nn.ConvTranspose(
                hidden_dim,
                kernel_size=(2, 2),
                strides=(2, 2) if i == 0 else (1, 1),
                kernel_init=_convnext_initializer,
                name=f"upsample_convt{i}",
            )(h)
            h = nn.LayerNorm(name=f"upsample_norm{i}")(h)

            h = ConvNeXtBlock(hidden_dim)(h)
            # TODO: ^ maybe we want more of these?


        if self.likelihood == 'bernoulli':
            logits = nn.Conv(
                self.image_shape[-1],
                kernel_size=(3, 3),
                strides=(1, 1),
                name=f'logits'
            )(h)

            return dists.Bernoulli(logits=logits)

        else:
            μ = nn.Conv(
                self.image_shape[-1],
                kernel_size=(3, 3),
                strides=(1, 1),
                name=f'μ'
            )(h)

            if 'hetero' in self.likelihood:
                if not 'iso' in self.likelihood:
                    σ = jax.nn.softplus(nn.Conv(
                            self.image_shape[-1],
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            name=f'σ_'
                        )(h))
                else:
                    σ = jax.nn.softplus(nn.Dense(1, name=f'σ_')(h))

            else:
                σ = jax.nn.softplus(self.param(
                    'σ_',
                    self.σ_init,
                    (1,) if 'iso' in self.likelihood else self.image_shape
                ))
                if 'unit' in self.likelihood:
                    σ = jax.lax.stop_gradient(σ)

            return dists.Normal(loc=μ, scale=σ.clip(min=self.σ_min))
