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
    make_invariant_encoder,
    raise_if_not_in_list, get_agg_fn,
    MAX_η, MIN_η, INV_SOFTPLUS_1, NUM_TRANSFORM_PARAMS,
    apply_mask
)
from src.transformations.affine import gen_transform_mat, transform_image

KwArgs = Mapping[str, Any]


_ENCODER_INVARIANCE_MODES = ['full', 'partial', 'none']


# TODO: generalise to more than just rotations
class LIVAE(VAE):
    encoder_invariance: str = 'partial'
    invariance_samples: Optional[int] = None
    learn_η_loc: bool = False
    learn_η_scale: bool = True
    η_encoder: Optional[KwArgs] = None
    η_mask: Array = jnp.ones((NUM_TRANSFORM_PARAMS,))
    recon_title: str = "Reconstructions: original – $\\hat{x}$ mean – $x$ mean - $\\hat{x}$ sample - $x$ sample"
    sample_title: str = "Prior Samples: $\\hat{x}$ mean – $x$ mean - $\\hat{x}$ sample - $x$ sample"

    def setup(self):
        super().setup()

        self.η_prior_μ = self.param(
            'η_prior_μ', init.zeros, (NUM_TRANSFORM_PARAMS,)
        )
        self.η_prior_σ = jax.nn.softplus(self.param(
            'η_prior_σ_',
            # init.constant(3.1),
            init.constant(INV_SOFTPLUS_1),
            (NUM_TRANSFORM_PARAMS,)
        ))

        if not self.learn_η_loc:
            # Note: this is helpful for identifiability, however, it isn't suitable
            # when the transformation is not symetric (e.g. a rotation between 0 and π/4)
            self.η_prior_μ = jax.lax.stop_gradient(self.η_prior_μ)
        if not self.learn_η_scale:
            self.η_prior_σ = jax.lax.stop_gradient(self.η_prior_σ)

        self.η_prior_μ = apply_mask(self.η_prior_μ, self.η_mask)
        self.η_prior_σ = apply_mask(self.η_prior_σ, self.η_mask, 1e-18)

        if 'normal' in self.η_encoder['posterior']:
            self.p_η = distrax.Normal(loc=self.η_prior_μ, scale=self.η_prior_σ)
        else:
            assert self.η_encoder['posterior'] == 'uniform'
            high = self.η_prior_μ + self.η_prior_σ
            low = self.η_prior_μ - self.η_prior_σ
            self.p_η = distrax.Uniform(low=low, high=high)

        Encoder, _ = get_enc_dec(self.architecture)
        self.η_enc = Encoder(
            latent_dim=7, **(self.η_encoder or {}), prior=self.p_η, output_mask=self.η_mask,
        )

    def __call__(self, x, rng, train=True, invariance_samples=None):
        raise_if_not_in_list(self.encoder_invariance, _ENCODER_INVARIANCE_MODES, 'self.encoder_invariance')
        z_rng, η_rng, xhat_rng, inv_rng = random.split(rng, 4)

        if self.encoder_invariance in ['full', 'partial']:
            # # Here we choose between having an encoder that is fully invariant to transformations
            # # e.g. in the case of rotations, for angles between -π and π, or partially invariant
            # # as specified by the prior on η.
            # if self.encoder_invariance == 'full':
            #     p_η = distrax.Uniform(low=MIN_η, high=MAX_η)
            # else:
            #     assert self.encoder_invariance == 'partial'
            #     p_η = self.p_η

            # invariance_samples = nn.merge_param(
            #     'invariance_samples', self.invariance_samples, invariance_samples
            # )
            # # The encoder should be invariant to transformations of the observed data x that result in sample x'
            # # the range [η_low, η_high] / [η_min, η_max] *relative to the prototype xhat*. If this was relative to x,
            # # applying two transformations could result in some samples x' being outside of the data distribution /
            # # the allowed maximum transformation ranges.
            # η_ = q_η_x.mean()
            # T = gen_transform_mat(-η_)
            # # ^ -ve transformation when going from x to xhat
            # xhat_ = transform_image(x, T)
            # # TODO: ^ make stochastic?
            # q_z_x = make_invariant_encoder(self.enc, xhat_, p_η, invariance_samples, inv_rng, train)
            pass
        else:
            q_z_x = self.enc(x, train=train)

        z = q_z_x.sample(seed=z_rng)

        # TODO: add equivariance
        q_η_x = self.η_enc(x, train=train)
        η = q_η_x.sample(seed=η_rng)

        p_xhat_z = self.dec(z, train=train)
        xhat = p_xhat_z.sample(seed=xhat_rng)

        T = gen_transform_mat(η)
        x_ = transform_image(xhat, T)
        # TODO: make stochastic?
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
            xhat = p_xhat_z.mean()

        if return_xhat:
            return xhat
        else:
            if not return_mode:
                η = self.p_η.sample(seed=η_rng)
            else:
                η = self.p_η.mean()

            T = gen_transform_mat(η)
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
    zs_rng: Optional[jax.random.PRNGKey] = None,
) -> Callable:
    """Creates a function for evaluating a VAE."""
    def batch_eval(params, state, eval_rng, β=1.):
        if zs_rng is None:
            eval_rng, sample_rng = random.split(eval_rng)
        else:
            sample_rng = zs_rng

        # Define eval func for 1 example.
        def eval_fn(x):
            z_rng, x_hat_rng, η_rng = random.split(random.fold_in(eval_rng, lax.axis_index('batch')), 3)
            q_z_x, q_η_x, p_x_xhat_η, p_xhat_z, p_z, p_η = model.apply(
                {'params': params, **state}, x, z_rng, train=False
            )

            metrics = _calculate_elbo_and_metrics(x, q_z_x, q_η_x, p_x_xhat_η, p_z, p_η, β)

            x_hat_mode = p_xhat_z.mean()
            x_hat_sample = p_xhat_z.sample(seed=x_hat_rng, sample_shape=())

            η = q_η_x.mean()
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
