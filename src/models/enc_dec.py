from typing import Callable, List, Optional, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
import flax.linen.initializers as init
import tensorflow_probability.substrates.jax.distributions as dists


_LIKELIHOODS = [
    'iso-normal',  # Homoskedastic (i.e. not input dependant) isotropic Guassian.
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
    # TODO: this ^ is a bit gross


class FCEncoder(nn.Module):
    latent_dim: int
    posterior: str = 'hetero-diag-normal'
    hidden_dims: Optional[List[int]] = None
    act_fn: Union[Callable, str] = nn.relu

    @nn.compact
    def __call__(self, x, train):
        if self.posterior not in _POSTERIORS:
            msg = f'`self.posterior` should be one of `{_POSTERIORS} but was `{self.posterior}` instead.'
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


class ConvEncoder(nn.Module):
    latent_dim: int
    posterior: str = 'hetero-diag-normal'
    hidden_dims: Optional[List[int]] = None
    act_fn: Union[Callable, str] = nn.relu
    norm_cls: nn.Module = nn.LayerNorm

    @nn.compact
    def __call__(self, x, train):
        if self.posterior not in _POSTERIORS:
            msg = f'`self.posterior` should be one of `{_POSTERIORS} but was `{self.posterior}` instead.'
            raise RuntimeError(msg)

        if self.hidden_dims is None:
            self.hidden_dims = [64, 128]

        act_fn = _get_act_fn(self.act_fn)

        h = x
        for i, hidden_dim in enumerate(self.hidden_dims):
            h = nn.Conv(hidden_dim, (3, 3), 2 if i==0 else 1, name=f'hidden{i}')(h)
            h = self.norm_cls(name=f'norm{i}')(h)
            h = act_fn(h)

        h = h.flatten()

        μ = nn.Dense(self.latent_dim, name=f'μ')(h)
        σ = jax.nn.softplus(nn.Dense(self.latent_dim, name='σ')(h))

        return dists.Normal(loc=μ, scale=σ)


class FCDecoder(nn.Module):
    image_shape: int
    likelihood: str = 'iso-normal'
    hidden_dims: Optional[List[int]] = None
    act_fn: Union[Callable, str] = nn.relu
    σ_init: Callable = init.constant(jnp.log(jnp.exp(1) - 1.))
    # ^ this value is softplus^{-1}(1), i.e., σ starts at 1.

    @nn.compact
    def __call__(self, z, train):
        if self.likelihood not in _LIKELIHOODS:
            msg = f'`self.likelihood` should be one of `{_LIKELIHOODS} but was `{self.likelihood}` instead.'
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

            return dists.Normal(loc=μ, scale=σ)


class ConvDecoder(nn.Module):
    image_shape: int
    likelihood: str = 'iso-normal'
    hidden_dims: Optional[List[int]] = None
    act_fn: Union[Callable, str] = nn.relu
    σ_init: Callable = init.constant(jnp.log(jnp.exp(1) - 1.))
    # ^ this value is softplus^{-1}(1), i.e., σ starts at 1.
    norm_cls: nn.Module = nn.LayerNorm

    @nn.compact
    def __call__(self, z, train):
        if self.likelihood not in _LIKELIHOODS:
            msg = f'`self.likelihood` should be one of `{_LIKELIHOODS} but was `{self.likelihood}` instead.'
            raise RuntimeError(msg)

        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]

        assert self.image_shape[0] == self.image_shape[1], "Images should be square."
        output_size = self.image_shape[0]
        first_hidden_size = output_size // 2

        act_fn = _get_act_fn(self.act_fn)

        h = nn.Dense(first_hidden_size * first_hidden_size * self.hidden_dims[0], name=f'resize')(z)
        h = h.reshape(first_hidden_size, first_hidden_size, self.hidden_dims[0])

        for i, hidden_dim in enumerate(self.hidden_dims):
            h = nn.ConvTranspose(hidden_dim, (3, 3), (2, 2) if i == 0 else (1, 1), name=f'hidden{i}')(h)
            h = self.norm_cls(name=f'norm{i}')(h)
            h = act_fn(h)

        if self.likelihood == 'bernoulli':
            logits = nn.Conv(self.image_shape[-1], (3, 3), 1, name=f'logits')(h)

            return dists.Bernoulli(logits=logits)

        else:
            μ = nn.Conv(self.image_shape[-1], (3, 3), 1, name=f'μ')(h)

            if 'hetero' in self.likelihood:
                if not 'iso' in self.likelihood:
                    σ = jax.nn.softplus(nn.Conv(
                            self.image_shape[-1], (3, 3), 1, name=f'σ_'
                        )(h))
                else:
                    σ = jax.nn.softplus(nn.Dense(1, name=f'σ_')(h))

            else:
                σ = jax.nn.softplus(self.param(
                    'σ_',
                    self.σ_init,
                    (1,) if 'iso' in self.likelihood else self.image_shape
                ))

            return dists.Normal(loc=μ, scale=σ)