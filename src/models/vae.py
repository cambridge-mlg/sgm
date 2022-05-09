from typing import Any, Callable, Mapping, Optional, Tuple
from functools import partial
from math import prod

import jax
import jax.numpy as jnp
from jax import random, lax
from jax.tree_util import tree_map
from chex import Array
from flax import linen as nn
import flax.linen.initializers as init
import tensorflow_probability.substrates.jax.distributions as dists

from src.models.enc_dec import FCDecoder, FCEncoder, ConvDecoder, ConvEncoder


KwArgs = Mapping[str, Any]


_ARCHITECTURES = [
    'MLP',
    'ConvNet',
    'ResNet',
]


class VAE(nn.Module):
    latent_dim: int = 20
    # TODO: support other priors? e.g.:
    # prior: str = 'diag-normal'
    learn_prior: bool = False
    convolutional: bool = False
    encoder: Optional[KwArgs] = None
    decoder: Optional[KwArgs] = None

    def setup(self):
        # TODO: support convolutional VAE.
        if self.architecture == 'ConvNet':
            Encoder = ConvEncoder
            Decoder = ConvDecoder
        elif self.architecture == 'MLP':
            Encoder = FCEncoder
            Decoder = FCDecoder

        self.enc = Encoder(latent_dim=self.latent_dim, **(self.encoder or {}))
        self.dec = Decoder(**(self.decoder or {}))

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
        q_z_x = self.enc(x, train=train)
        z = q_z_x.sample(seed=rng)

        p_x_z = self.dec(z, train=train)

        return q_z_x, p_x_z, self.p_z

    def generate(self, z, rng, sample_shape=(1,), return_mode=False):
        p_x_z = self.dec(z, train=False)

        if not return_mode:
            return p_x_z.sample(seed=rng, sample_shape=sample_shape)
        else:
            return p_x_z.mode()


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

            metrics = _calculate_metrics(x, q_z_x, p_x_z, p_z)
            elbo = metrics['elbo']

            return -elbo, new_state, metrics

        # Broadcast over batch and take mean.
        batch_losses, new_state, batch_metrics = jax.vmap(
            loss_fn, out_axes=(0, None, 0), in_axes=(0), axis_name='batch'
        )(x_batch)
        batch_metrics = tree_map(lambda x: x.mean(axis=0), batch_metrics)
        return batch_losses.mean(axis=0), (new_state, batch_metrics)

    return jax.jit(batch_loss)


def make_VAE_eval(
    model: VAE,
    x_batch: Array,
    zs: Array,
    img_shape: Tuple,
    num_recons: int = 16,
) -> Callable:
    """Creates a function for evaluating a VAE."""
    def batch_eval(params, state, batch_rng):
        eval_rng, sample_rng = random.split(batch_rng)

        # Define eval func for 1 example.
        def eval_fn(x):
            rng1, rng2 = random.split(random.fold_in(eval_rng, lax.axis_index('batch')))
            q_z_x, p_x_z, p_z = model.apply(
                {'params': params, **state}, x, rng1, train=False
            )

            metrics = _calculate_metrics(x, q_z_x, p_x_z, p_z)

            return metrics, p_x_z.mode(), p_x_z.sample(seed=rng2, sample_shape=(1,))

        # Broadcast over batch and take mean.
        batch_metrics, batch_x_recon_mode, batch_x_recon_sample = jax.vmap(
            eval_fn, out_axes=(0, 0, 0), in_axes=(0,), axis_name='batch'
        )(x_batch)
        batch_metrics = tree_map(lambda x: x.mean(axis=0), batch_metrics)

        recon_comparison = jnp.concatenate([
            x_batch[:num_recons].reshape(-1, *img_shape),
            batch_x_recon_mode[:num_recons].reshape(-1, *img_shape),
            batch_x_recon_sample[:num_recons].reshape(-1, *img_shape),
        ])

        @partial(jax.vmap, axis_name='batch')
        def sample_fn(z):
            rng = random.fold_in(sample_rng, lax.axis_index('batch'))
            x_sample = model.apply(
                {'params': params, **state}, z, rng,
                method=VAE.generate
            )

            x_mode = model.apply(
                {'params': params, **state}, z, rng, return_mode=True,
                method=VAE.generate
            )

            return x_sample, x_mode

        sampled_images, image_modes = sample_fn(zs)
        samples = jnp.concatenate([
            image_modes.reshape(-1, *img_shape),
            sampled_images.reshape(-1, *img_shape),
        ])

        return batch_metrics, recon_comparison, samples

    return jax.jit(batch_eval)


def _calculate_metrics(x, q_z_x, p_x_z, p_z):
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
