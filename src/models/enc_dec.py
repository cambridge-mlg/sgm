from functools import partial
from typing import Callable, List, Optional, Union
from math import prod

import jax
import jax.numpy as jnp
from flax import linen as nn
import flax.linen.initializers as init
import distrax

from src.models.common import raise_if_not_in_list, INV_SOFTPLUS_1


_LIKELIHOODS = [
    'iso-normal',  # Homoskedastic (i.e. not input dependant) isotropic Guassian.
    'unit-iso-normal',  # Unit-variance isotropic Guassian.
    'diag-normal',  # Homoskedastic diagonal Guassian.
    'hetero-diag-normal',  # Heteroskedastic (i.e., predicted per input by an NN) diagonal Guassian.
    'hetero-iso-normal',  # Heteroskedastic isotropic Guassian.
    'bernoulli',
    # TODO: support Categorical.
]


_POSTERIORS = [
    'hetero-diag-normal',
    # TODO: support Full Cov Normal.
]


# This function allows us to either specify activation functions callables,
# as seen in the default for FCEncoder below, or as string names,
# which is useful for commandline args or config files.
def _get_act_fn(act_fn):
    if isinstance(act_fn, str):
        return getattr(nn, act_fn)
    else:
        return act_fn


def create_likelihood(obj, hidden, output_layer, output_shape):
    if obj.likelihood == 'bernoulli':
        logits = output_layer(name=f'logits')(hidden)
        return distrax.Bernoulli(logits=logits)

    else:
        μ = output_layer(name=f'μ')(hidden)

        if obj.likelihood == 'hetero-diag-normal':
            σ_ = output_layer(name=f'σ_')(hidden)

        elif obj.likelihood == 'hetero-iso-normal':
            σ_ = nn.Dense(1, name=f'σ_')(hidden.flatten())

        elif obj.likelihood == 'iso-normal':
            σ_ = obj.param('σ_', obj.σ_init, (1,))

        elif obj.likelihood == 'unit-iso-normal':
            σ_ = jax.lax.stop_gradient(obj.param('σ_', init.constant(INV_SOFTPLUS_1), (1,)))

        else:
            assert obj.likelihood == 'diag-normal'
            σ_ = obj.param('σ_', obj.σ_init, output_shape)

        return distrax.Normal(loc=μ, scale=jax.nn.softplus(σ_).clip(min=obj.σ_min))


class FCEncoder(nn.Module):
    latent_dim: int
    posterior: str = 'hetero-diag-normal'
    hidden_dims: Optional[List[int]] = None
    act_fn: Union[Callable, str] = nn.relu

    @nn.compact
    def __call__(self, x, train=True):
        raise_if_not_in_list(self.posterior, _POSTERIORS, 'self.posterior')

        if self.hidden_dims is None:
            self.hidden_dims = [500,]

        act_fn = _get_act_fn(self.act_fn)

        h = x
        for i, hidden_dim in enumerate(self.hidden_dims):
            h = act_fn(nn.Dense(hidden_dim, name=f'hidden{i}')(h))

        μ = nn.Dense(self.latent_dim, name=f'μ')(h)
        σ = jax.nn.softplus(nn.Dense(self.latent_dim, name=f'σ')(h))

        return distrax.Normal(loc=μ, scale=σ)


class FCDecoder(nn.Module):
    image_shape: int
    likelihood: str = 'iso-normal'
    hidden_dims: Optional[List[int]] = None
    σ_init: Callable = init.constant(INV_SOFTPLUS_1)
    σ_min: float = 1e-2
    act_fn: Union[Callable, str] = nn.relu

    @nn.compact
    def __call__(self, z, train=True):
        raise_if_not_in_list(self.likelihood, _LIKELIHOODS, 'self.likelihood')

        if self.hidden_dims is None:
            self.hidden_dims = [500,]

        act_fn = _get_act_fn(self.act_fn)

        h = z
        for i, hidden_dim in enumerate(self.hidden_dims):
            h = act_fn(nn.Dense(hidden_dim, name=f'hidden{i}')(h))

        output_dim = prod(self.image_shape)
        return create_likelihood(self, h, partial(nn.Dense, output_dim), (output_dim,))


class ConvEncoder(nn.Module):
    latent_dim: int
    posterior: str = 'hetero-diag-normal'
    hidden_dims: Optional[List[int]] = None
    act_fn: Union[Callable, str] = nn.relu
    norm_cls: nn.Module = nn.LayerNorm

    @nn.compact
    def __call__(self, x, train=True):
        raise_if_not_in_list(self.posterior, _POSTERIORS, 'self.posterior')

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

        return distrax.Normal(loc=μ, scale=σ)


class ConvDecoder(nn.Module):
    image_shape: int
    likelihood: str = 'iso-normal'
    hidden_dims: Optional[List[int]] = None
    σ_init: Callable = init.constant(INV_SOFTPLUS_1)
    σ_min: float = 1e-2
    act_fn: Union[Callable, str] = nn.relu
    norm_cls: nn.Module = nn.LayerNorm

    @nn.compact
    def __call__(self, z, train=True):
        raise_if_not_in_list(self.likelihood, _LIKELIHOODS, 'self.likelihood')

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

        output_conv = partial(
            nn.Conv,
            self.image_shape[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
        )

        return create_likelihood(self, h, output_conv, self.image_shape)


_convnext_initializer = nn.initializers.variance_scaling(
    0.2, "fan_in", distribution="truncated_normal"
)


class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block.

    Adapted from: https://github.com/DarshanDeshpande/jax-models/blob/ae5750540f142572ff7f276b927a9cdb5195fd23/jax_models/models/convnext.py#L5
    """
    dim: int = 64
    layer_scale_init_value: float = 1e-6
    act_fn: Callable = nn.gelu
    norm_cls: nn.Module = nn.LayerNorm

    @nn.compact
    def __call__(self, inputs):
        x = nn.Conv(
            self.dim,
            kernel_size=(7, 7),
            feature_group_count=self.dim,
            kernel_init=_convnext_initializer,
            name="dw3x3conv"
        )(inputs)
        x = self.norm_cls(name="norm")(x)
        x = nn.Dense(
            4 * self.dim,
            kernel_init=_convnext_initializer,
            name="pw1x1conv1"
        )(x)
        x = self.act_fn(x)
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
        # NOTE: ^ this should have DropPath if it was being used in an full ConvNeXt ResNet.
        return x


class ConvNeXtEncoder(nn.Module):
    latent_dim: int
    posterior: str = 'hetero-diag-normal'
    hidden_dims: Optional[List[int]] = None
    act_fn: Union[Callable, str] = nn.gelu
    norm_cls: nn.Module = nn.LayerNorm

    @nn.compact
    def __call__(self, x, train=True):
        raise_if_not_in_list(self.posterior, _POSTERIORS, 'self.posterior')

        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 256, 512]

        act_fn = _get_act_fn(self.act_fn)

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

            h = ConvNeXtBlock(hidden_dim, act_fn=act_fn, norm_cls=self.norm_cls)(h)

        h = h.flatten()

        μ = nn.Dense(self.latent_dim, name=f'μ')(h)
        σ = jax.nn.softplus(nn.Dense(self.latent_dim, name='σ')(h))

        return distrax.Normal(loc=μ, scale=σ)


class ConvNeXtDecoder(nn.Module):
    image_shape: int
    likelihood: str = 'iso-normal'
    hidden_dims: Optional[List[int]] = None
    σ_init: Callable = init.constant(INV_SOFTPLUS_1)
    σ_min: float = 1e-2
    act_fn: Union[Callable, str] = nn.gelu
    norm_cls: nn.Module = nn.LayerNorm


    @nn.compact
    def __call__(self, z, train=True):
        raise_if_not_in_list(self.likelihood, _LIKELIHOODS, 'self.likelihood')

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
            h = self.norm_cls(name=f"upsample_norm{i}")(h)

            h = ConvNeXtBlock(hidden_dim)(h)

        output_conv = partial(
            nn.Conv,
            self.image_shape[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
        )

        return create_likelihood(self, h, output_conv, self.image_shape)
