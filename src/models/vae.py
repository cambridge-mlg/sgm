from typing import Callable, List, Optional
from functools import partial
from math import prod

import jax.numpy as jnp
import jax
from jax import random, lax
from chex import Array
from flax import linen as nn
import flax.linen.initializers as init
import tensorflow_probability.substrates.jax.distributions as dists

# TODO: support Categorical.
_supported_likelihoods = [
    'iso-normal',  # Homoskedastic (i.e. not input dependant) isotropic Guassian.
    'diag-normal',  # Homoskedastic diagonal Guassian.
    'hetero-diag-normal',  # Heteroskedastic (i.e., predicted per input by an NN) isotropic Guassian.
    'hetero-iso-normal',  # Heteroskedastic diagonal Guassian.
    'bernoulli'
]

# TODO: support Full Cov Normal.
_supported_posteriors = ['diag-normal']


class FCEncoder(nn.Module):
    latent_dim: int
    posterior: str
    hidden_dims: Optional[List[int]] = None
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x, train):
        assert self.posterior in _supported_posteriors
        if self.hidden_dims is None:
            self.hidden_dims = [500,]

        h = x
        for i, hidden_dim in enumerate(self.hidden_dims):
            h = self.act_fn(nn.Dense(hidden_dim, name=f'fc{i}')(h))

        μ = nn.Dense(self.latent_dim, name=f'fc{i+1}_μ')(h)
        σ = jax.nn.softplus(nn.Dense(self.latent_dim, name=f'fc{i+1}_σ')(h))

        return dists.Normal(μ, σ)


class FCDecoder(nn.Module):
    output_dim: int
    likelihood: str
    hidden_dims: Optional[List[int]] = None
    act_fn: Callable = nn.relu
    σ_init: Callable = init.constant(jnp.log(jnp.exp(1) - 1.))
    # ^ this value is softplus^{-1}(1), i.e., σ starts at 1.

    @nn.compact
    def __call__(self, z, train):
        assert self.likelihood in _supported_likelihoods
        if self.hidden_dims is None:
            self.hidden_dims = [500,]

        h = z
        for i, hidden_dim in enumerate(self.hidden_dims):
            h = self.act_fn(nn.Dense(hidden_dim, name=f'fc{i}')(h))

        if self.likelihood == 'bernoulli':
            logits = nn.Dense(self.output_dim, name=f'fc{i+1}')(h)

            return dists.Bernoulli(logits=logits)

        else:
            μ = nn.Dense(self.output_dim, name=f'fc{i+1}_μ')(h)

            if 'iso' in self.likelihood:
                σ_size = 1
            else:
                σ_size = self.output_dim

            if 'hetero' in self.likelihood:
                σ = jax.nn.softplus(nn.Dense(σ_size, name=f'fc3\{i+1}_σ')(h))
            else:
                σ = jax.nn.softplus(self.param(
                    'σ_',
                    self.σ_init,
                    (σ_size,)
                ))

            return dists.Normal(loc=μ, scale=σ)


class VAE(nn.Module):
    output_dim: int
    latent_dim: int = 20
    likelihood: str = 'diag-normal'
    posterior: str = 'diag-normal'
    # TODO: support other priors? e.g.:
    # prior: str = 'diag-normal'
    learn_prior: bool = False
    convolutional: bool = False
    enc_hidden_dims: Optional[List[int]] = None
    dec_hidden_dims: Optional[List[int]] = None
    act_fn: Callable = nn.relu

    def setup(self):
        if self.likelihood not in _supported_likelihoods:
            msg = f'`self.likelihood` should be one of `{_supported_likelihoods} but was `{self.likelihood}` instead.'
            raise RuntimeError(msg)

        if self.posterior not in _supported_posteriors:
            msg = f'`self.posterior` should be one of `{_supported_posteriors} but was `{self.posterior}` instead.'
            raise RuntimeError(msg)

        # TODO: support convolutional VAE.
        if self.convolutional:
            msg = 'Convolutional VAE is not yet supported.'

        self.encoder = FCEncoder(self.latent_dim, self.posterior, self.enc_hidden_dims, self.act_fn)
        self.decoder = FCDecoder(self.output_dim, self.likelihood, self.dec_hidden_dims, self.act_fn)

        self.prior_loc = self.param(
            'prior_loc',
            init.zeros,
            (self.latent_dim,)
        )
        self.prior_scale = jax.nn.softplus(self.param(
            'prior_scale_',
            init.constant(jnp.log(jnp.exp(1) - 1.)),
            # ^ this value is softplus^{-1}(1), i.e., σ starts at 1.
            (self.latent_dim,)
        ))
        if not self.learn_prior:
            self.prior_loc = jax.lax.stop_gradient(self.prior_loc)
            self.prior_scale = jax.lax.stop_gradient(self.prior_scale)
        self.p_z = dists.Normal(loc=self.prior_loc, scale=self.prior_scale)

    def __call__(self, x, rng, train=True):
        q_z_x = self.encoder(x, train=train)
        z = q_z_x.sample(seed=rng)

        p_x_z = self.decoder(z, train=train)

        return q_z_x, p_x_z, self.p_z

    def generate(self, z, rng, sample_shape=(1,)):
        p_x_z = self.decoder(z, train=False)
        return p_x_z.sample(seed=rng, sample_shape=sample_shape)


def make_VAE_loss(
    model: VAE,
    x_batch: Array,
    train: bool = True,
) -> Callable:
    """Creates a loss function for training a VAE."""
    def batch_loss(params, state, batch_rng):
        # Define loss func for 1 example.
        def loss_fn(x):
            rng = random.fold_in(batch_rng, lax.axis_index('batch'))
            (q_z_x, p_x_z, p_z), new_state = model.apply(
                {'params': params, **state}, x, rng,
                mutable=list(state.keys()) if train else {},
            )

            elbo = _calculate_elbo(x, q_z_x, p_x_z, p_z)['elbo']

            return -elbo, new_state

        # Broadcast over batch and take mean.
        batch_losses, new_state = jax.vmap(
            loss_fn, out_axes=(0, None), in_axes=(0), axis_name='batch'
        )(x_batch)
        return batch_losses.mean(axis=0), (new_state,)

    return batch_loss


def make_VAE_eval(
    model: VAE,
    x_batch: Array,
    zs: Array,
    num_recons: int = 16
) -> Callable:
    """Creates a function for evaluating a VAE."""
    def batch_eval(params, state, batch_rng):
        eval_rng, sample_rng = random.split(batch_rng)

        # Define eval func for 1 example.
        def eval_fn(x):
            rng = random.fold_in(eval_rng, lax.axis_index('batch'))
            q_z_x, p_x_z, p_z = model.apply(
                {'params': params, **state}, x, rng, train=False
            )

            metrics = _calculate_elbo(x, q_z_x, p_x_z, p_z)

            return metrics, p_x_z.mode()

        # Broadcast over batch and take mean.
        batch_metrics, batch_x_recon= jax.vmap(
            eval_fn, out_axes=(0, 0), in_axes=(0,), axis_name='batch'
        )(x_batch)
        batch_metrics = jax.tree_util.tree_map(lambda x: x.mean(axis=0), batch_metrics)

        recon_comparison = jnp.concatenate([
            x_batch[:num_recons].reshape(-1, 28, 28, 1),
            batch_x_recon[:num_recons].reshape(-1, 28, 28, 1)
        ])

        @partial(jax.vmap, axis_name='batch')
        def sample_fn(z):
            rng = random.fold_in(sample_rng, lax.axis_index('batch'))
            x_sample = model.apply(
                {'params': params, **state}, z, rng,
                method=VAE.generate
            )

            return x_sample

        sampled_images = sample_fn(zs).reshape(-1, 28, 28, 1)

        return batch_metrics, recon_comparison, sampled_images

    return batch_eval


def _calculate_elbo(x, q_z_x, p_x_z, p_z):
    x_size = prod(p_x_z.batch_shape)
    z_size = prod(p_z.batch_shape)

    ll = p_x_z.log_prob(x).sum()
    kld = q_z_x.kl_divergence(p_z).sum()
    elbo = ll - kld

    return {
        'll': ll,
        'kld': kld,
        'elbo': elbo / (x_size * z_size),
        # ^ We normalise the ELBO by the data and latent size to (hopefully) make LR, etc., more general.
        'elbo_unnorm': elbo,
    }
