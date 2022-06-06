from typing import Any, Callable, Mapping, Optional, Tuple
from functools import partial
from math import prod

import jax
import jax.numpy as jnp
from jax import random, lax
from jax.tree_util import tree_map
from chex import Array
import flax.linen as nn
import flax.linen.initializers as init
import distrax

from src.models.vae import VAE, get_enc_dec
from src.models.common import (
    sample_transformed_data, make_invariant_encoder,
    raise_if_not_in_list, get_agg_fn,
    MAX_η, MIN_η, INV_SOFTPLUS_1,
)
from src.transformations.affine import gen_transform_mat, transform_image

KwArgs = Mapping[str, Any]


_ENCODER_INVARIANCE_MODES = ['full', 'partial', 'none']


# TODO: generalise to more than just rotations
class LIVAE(VAE):
    encoder_invariance: str = 'partial'
    invariance_samples: Optional[int] = None
    learn_η_prior: bool = False
    recon_title: str = "Reconstructions: original – $\\hat{x}$ mode – $x$ mode - $\\hat{x}$ sample - $x$ sample"
    sample_title: str = "Prior Samples: $\\hat{x}$ mode – $x$ mode - $\\hat{x}$ sample - $x$ sample"

    def setup(self):
        super().setup()

        self.η_prior_μ = self.param(
            'η_prior_μ',
            init.zeros,
            (1,)
        )
        self.η_prior_σ = jax.nn.softplus(self.param(
            'η_prior_σ_',
            init.constant(INV_SOFTPLUS_1),
            (1,)
        ))
        # we always fix loc=0 to avoid identifiability issues
        self.η_prior_μ = jax.lax.stop_gradient(self.η_prior_μ)
        if not self.learn_η_prior:
            self.η_prior_σ = jax.lax.stop_gradient(self.η_prior_σ)
        self.p_η = distrax.Normal(loc=self.η_prior_μ, scale=self.η_prior_σ)

        Encoder, _ = get_enc_dec(self.architecture)
        self.η_enc = Encoder(latent_dim=1, **(self.encoder or {}))

    def __call__(self, x, rng, train=True, invariance_samples=None):
        raise_if_not_in_list(self.encoder_invariance, _ENCODER_INVARIANCE_MODES, 'self.encoder_invariance')
        z_rng, η_rng, xhat_rng = random.split(rng, 3)

        # TODO: add invariance
        q_z_x = self.enc(x, train=train)
        z = q_z_x.sample(seed=z_rng)

        # TODO: add equivariance
        q_η_x = self.η_enc(x, train=train)
        η = q_η_x.sample(seed=η_rng)

        p_xhat_z = self.dec(z, train=train)
        xhat = p_xhat_z.sample(seed=xhat_rng)

        T = gen_transform_mat(jnp.array([0., 0., η[0], 0., 0., 0., 0.]))
        x_ = transform_image(xhat, T)
        # TODO: make noisy?
        p_x_xhat_η = distrax.Normal(x_, 1.)
        # TODO: learn scale param here?

        return q_z_x, q_η_x, p_x_xhat_η, p_xhat_z, self.p_z, self.p_η

    def generate(self, rng, sample_shape=(), return_mode=False, return_xhat=False):
        z_rng, η_rng, xhat_rng, transform_rng = random.split(rng, 4)
        z = self.p_z.sample(seed=z_rng)
        p_xhat_z = self.dec(z, train=False)

        if not return_mode:
            xhat = p_xhat_z.sample(seed=xhat_rng, sample_shape=sample_shape)
        else:
            xhat = p_xhat_z.mode()

        if return_xhat:
            return xhat
        else:
            if not return_mode:
                η = self.p_η.sample(seed=η_rng)
            else:
                η = self.p_η.mode()

            T = gen_transform_mat(jnp.array([0., 0., η[0], 0., 0., 0., 0.]))
            x = transform_image(xhat, T)
            # TODO: make noisy?
            # TODO: vmap this to deal with more than 1 sample
            return x


def make_LIVAE_loss(
    model: LIVAE,
    x_batch: Array,
    train: bool = True,
    aggregation: str = 'mean',
) -> Callable:
    """Creates a loss function for training a VAE."""
    def batch_loss(params, state, batch_rng, β=1.):
        # TODO: this loss function is a 1 sample estimate, add an option for more samples?
        # Define loss func for 1 example.
        def loss_fn(x):
            rng = random.fold_in(batch_rng, lax.axis_index('batch'))
            (q_z_x, q_η_x, p_x_xhat_η, _, p_z, p_η), new_state = model.apply(
                {'params': params, **state}, x, rng,
                mutable=list(state.keys()) if train else {},
            )

            metrics = _calculate_elbo_and_metrics(x, q_z_x, q_η_x, p_x_xhat_η, p_z, p_η, β)
            elbo = metrics['elbo']

            return -elbo, new_state, metrics

        # Broadcast over batch and take aggregate.
        agg = get_agg_fn(aggregation)
        batch_losses, new_state, batch_metrics = jax.vmap(
            loss_fn, out_axes=(0, None, 0), in_axes=(0), axis_name='batch'
        )(x_batch)
        batch_metrics = tree_map(lambda x: agg(x, axis=0), batch_metrics)
        return agg(batch_losses, axis=0), (new_state, batch_metrics)

    return jax.jit(batch_loss)


def make_LIVAE_eval(
    model: LIVAE,
    x_batch: Array,
    img_shape: Tuple,
    num_recons: int = 16,
    aggregation: str = 'mean',
) -> Callable:
    """Creates a function for evaluating a VAE."""
    def batch_eval(params, state, batch_rng, β=1.):
        eval_rng, sample_rng = random.split(batch_rng)

        # Define eval func for 1 example.
        def eval_fn(x):
            z_rng, x_hat_rng, η_rng = random.split(random.fold_in(eval_rng, lax.axis_index('batch')), 3)
            q_z_x, q_η_x, p_x_xhat_η, p_xhat_z, p_z, p_η = model.apply(
                {'params': params, **state}, x, z_rng, train=False
            )

            metrics = _calculate_elbo_and_metrics(x, q_z_x, q_η_x, p_x_xhat_η, p_z, p_η, β)

            x_hat_mode = p_xhat_z.mode()
            x_hat_sample = p_xhat_z.sample(seed=x_hat_rng, sample_shape=())

            η = q_η_x.mode()
            T = gen_transform_mat(jnp.array([0., 0., η[0], 0., 0., 0., 0.]))
            x_mode = transform_image(x_hat_mode, T)

            η = q_η_x.sample(seed=η_rng)
            T = gen_transform_mat(jnp.array([0., 0., η[0], 0., 0., 0., 0.]))
            x_sample = transform_image(x_hat_sample, T)

            return metrics, x_hat_mode, x_hat_sample, x_mode, x_sample

        # Broadcast over batch and aggregate.
        agg = get_agg_fn(aggregation)
        batch_metrics, batch_x_hat_mode, batch_x_hat_sample, batch_x_mode, batch_x_sample = jax.vmap(
            eval_fn, out_axes=(0, 0, 0, 0, 0), in_axes=(0,), axis_name='batch'
        )(x_batch)
        batch_metrics = tree_map(lambda x: agg(x, axis=0), batch_metrics)

        recon_array = jnp.concatenate([
            x_batch[:num_recons].reshape(-1, *img_shape),
            batch_x_hat_mode[:num_recons].reshape(-1, *img_shape),
            batch_x_mode[:num_recons].reshape(-1, *img_shape),
            batch_x_hat_sample[:num_recons].reshape(-1, *img_shape),
            batch_x_sample[:num_recons].reshape(-1, *img_shape),
        ])

        @partial(jax.vmap, axis_name='batch')
        def sample_fn(batch_rng):
            xhat_sample = model.apply(
                {'params': params, **state}, batch_rng, return_mode=False, return_xhat=True,
                method=model.generate
            )

            x_sample = model.apply(
                {'params': params, **state}, batch_rng, return_mode=False, return_xhat=False,
                method=model.generate
            )

            xhat_mode = model.apply(
                {'params': params, **state}, batch_rng, return_mode=True, return_xhat=True,
                method=model.generate
            )

            x_mode = model.apply(
                {'params': params, **state}, batch_rng, return_mode=True, return_xhat=False,
                method=model.generate
            )

            return xhat_mode, x_mode, xhat_sample, x_sample

        xhat_modes, x_modes, xhat_samples, x_samples = sample_fn(random.split(sample_rng, num_recons))
        sample_array = jnp.concatenate([
            xhat_modes.reshape(-1, *img_shape),
            x_modes.reshape(-1, *img_shape),
            xhat_samples.reshape(-1, *img_shape),
            x_samples.reshape(-1, *img_shape),
        ])

        return batch_metrics, recon_array, sample_array

    return jax.jit(batch_eval)


def _calculate_elbo_and_metrics(x, q_z_x, q_η_x, p_x_xhat_η, p_z, p_η, β=1.):
    x_size = prod(p_x_xhat_η.batch_shape)

    ll = p_x_xhat_η.log_prob(x).sum()
    z_kld = q_z_x.kl_divergence(p_z).sum()
    η_kld = q_η_x.kl_divergence(p_η).sum()
    elbo = ll - β * z_kld - η_kld
    # TODO: add beta term for η_kld? Use same beta?

    return {
        'll': ll,
        'kld': z_kld,
        'η_kld': η_kld,
        'elbo': elbo / x_size,
        # ^ We normalise the ELBO by the data size to (hopefully) make LR, etc., more general.
        'elbo_unnorm': elbo,
    }
