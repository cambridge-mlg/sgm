from re import X
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union
from functools import partial
from math import prod

import jax
import jax.numpy as jnp
from jax import random, lax
from jax.tree_util import tree_map
from chex import Array
import flax.linen as nn

from src.models.vae import VAE
from src.models.common import sample_transformed_data, make_invariant_encoder


KwArgs = Mapping[str, Any]


_ENCODER_INVARIANCE_MODES = ['full', 'partial', 'none']


class invVAE(VAE):
    η_min: Optional[Union[Array, List]] = None
    η_max: Optional[Union[Array, List]] = None
    encoder_invariance: str = 'partial'
    invariance_samples: Optional[int] = None

    def setup(self):
        super().setup()

        if self.η_min is None or self.η_max is None:
            msg = f'`self.η_min` and self.η_max` must be specified, but were ({self.η_min}, {self.η_max}). See src.transformations.affine.gen_transform_mat for specification details.'
            raise RuntimeError(msg)

        self.η_min = jnp.array(self.η_min)
        self.η_max = jnp.array(self.η_max)

    def __call__(self, xhat, rng, train=True, invariance_samples=None):
        z_rng, transform_rng, inv_rng = random.split(rng, 3)
        x = sample_transformed_data(xhat, transform_rng, self.η_min, self.η_max)

        if self.encoder_invariance not in _ENCODER_INVARIANCE_MODES:
            msg = f'`self.encoder_invariance` should be one of `{_ENCODER_INVARIANCE_MODES}` but was `{self.encoder_invariance}` instead.'
            raise RuntimeError(msg)

        if self.encoder_invariance in ['full', 'partial']:
            inv_rot = self.max_rotation if self.encoder_invariance == 'partial' else jnp.pi/2
            invariance_samples = nn.merge_param(
                'invariance_samples', self.invariance_samples, invariance_samples
            )
            q_z_x = make_invariant_encoder(self.enc, x, self.η_min, self.η_max, invariance_samples, inv_rng, train)
        else:
            q_z_x = self.enc(x, train=train)

        z = q_z_x.sample(seed=z_rng)

        p_xhat_z = self.dec(z, train=train)

        return q_z_x, p_xhat_z, self.p_z

    def generate(self, z, rng, sample_shape=(), return_mode=False, return_xhat=False):
        x_rng, transform_rng = random.split(rng)
        p_xhat_z = self.dec(z, train=False)

        if not return_mode:
            xhat = p_xhat_z.sample(seed=x_rng, sample_shape=sample_shape)
        else:
            xhat = p_xhat_z.mode()

        if return_xhat:
            return xhat
        else:
            return sample_transformed_data(xhat, transform_rng, self.η_min, self.η_max)
            # TODO: vmap this to handle more than 1 sample of x_hat



def make_invVAE_loss(
    model: invVAE,
    x_batch: Array,
    train: bool = True,
) -> Callable:
    """Creates a loss function for training a VAE."""
    def batch_loss(params, state, batch_rng, β=1.):
        # TODO: this loss function is a 1 sample estimate, add an option for more samples?
        # Define loss func for 1 example.
        def loss_fn(xhat):
            rng = random.fold_in(batch_rng, lax.axis_index('batch'))
            (q_z_x, p_xhat_z, p_z), new_state = model.apply(
                {'params': params, **state}, xhat, rng,
                mutable=list(state.keys()) if train else {},
            )

            metrics = _calculate_elbo_and_metrics(xhat, q_z_x, p_xhat_z, p_z, β)
            elbo = metrics['elbo']

            return -elbo, new_state, metrics

        # Broadcast over batch and take mean.
        batch_losses, new_state, batch_metrics = jax.vmap(
            loss_fn, out_axes=(0, None, 0), in_axes=(0), axis_name='batch'
        )(x_batch)
        batch_metrics = tree_map(lambda x: x.mean(axis=0), batch_metrics)
        return batch_losses.mean(axis=0), (new_state, batch_metrics)

    return jax.jit(batch_loss)


def make_invVAE_eval(
    model: invVAE,
    x_batch: Array,
    zs: Array,
    img_shape: Tuple,
    num_recons: int = 16,
) -> Callable:
    """Creates a function for evaluating a VAE."""
    def batch_eval(params, state, batch_rng, β=1.):
        eval_rng, sample_rng = random.split(batch_rng)

        # Define eval func for 1 example.
        def eval_fn(xhat):
            rng1, rng2 = random.split(random.fold_in(eval_rng, lax.axis_index('batch')))
            q_z_x, p_xhat_z, p_z = model.apply(
                {'params': params, **state}, xhat, rng1, train=False
            )

            metrics = _calculate_elbo_and_metrics(xhat, q_z_x, p_xhat_z, p_z, β)

            return metrics, p_xhat_z.mode(), p_xhat_z.sample(seed=rng2, sample_shape=(1,))

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
            xhat_sample = model.apply(
                {'params': params, **state}, z, rng, return_mode=False, return_xhat=True,
                method=model.generate
            )

            x_sample = model.apply(
                {'params': params, **state}, z, rng, return_mode=False, return_xhat=False,
                method=model.generate
            )

            xhat_mode = model.apply(
                {'params': params, **state}, z, rng, return_mode=True, return_xhat=True,
                method=model.generate
            )

            x_mode = model.apply(
                {'params': params, **state}, z, rng, return_mode=True, return_xhat=False,
                method=model.generate
            )

            return xhat_mode, x_mode, xhat_sample, x_sample

        xhat_modes, x_modes, xhat_samples, x_samples = sample_fn(zs)
        samples = jnp.concatenate([
            xhat_modes.reshape(-1, *img_shape),
            x_modes.reshape(-1, *img_shape),
            xhat_samples.reshape(-1, *img_shape),
            x_samples.reshape(-1, *img_shape),
        ])

        return batch_metrics, recon_comparison, samples

    return jax.jit(batch_eval)


def _calculate_elbo_and_metrics(xhat, q_z_x, p_xhat_z, p_z, β=1.):
    x_size = prod(p_xhat_z.batch_shape)

    ll = p_xhat_z.log_prob(xhat).sum()
    kld = q_z_x.kl_divergence(p_z).sum()
    elbo = ll - β * kld

    return {
        'll': ll,
        'kld': kld,
        'elbo': elbo / x_size,
        # ^ We normalise the ELBO by the data size to (hopefully) make LR, etc., more general.
        'elbo_unnorm': elbo,
    }
