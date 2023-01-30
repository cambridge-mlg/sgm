"""Definition of the Learnt-Invariance VAE.

A note on notation. In order to distinguish between random variables and their values, we use upper
and lower case variable names. I.e., p(Z) or `p_Z` is the distribution over the r.v. Z, and is a
function, while p(z) or `p_z` is the probability that Z=z. Similarly, p(X|Z) or `p_X_given_Z` is a
a function which returns another function p(X|z) or `p_X_given_z`, which would return the proability
that X=x|Z=z a.k.k `p_x_given_z`.
"""

from typing import Any, Callable, Mapping, Optional, Tuple

import jax
from jax import numpy as jnp
from jax import random, lax
from jax.tree_util import tree_map
from chex import Array
from flax import linen as nn
import flax.linen.initializers as init
import distrax

from src.transformations.affine import rotate_image
from src.models.common import ConvEncoder, ConvDecoder, DenseEncoder, Bijector, INV_SOFTPLUS_1

KwArgs = Mapping[str, Any]


class LIVAE(nn.Module):
    latent_dim: int = 20
    image_shape: Tuple[int, int, int] = (28, 28, 1)
    encoder: Optional[KwArgs] = None
    decoder: Optional[KwArgs] = None
    θ_encoder: Optional[KwArgs] = None
    θ_decoder: Optional[KwArgs] = None
    θ_bijector: Optional[KwArgs] = None

    def setup(self):
        # p(Z)
        self.p_Z = distrax.Normal(
            loc=self.param(
                'prior_μ',
                init.zeros,
                (self.latent_dim,)
            ),
            scale=jax.nn.softplus(self.param(
                'prior_σ_',
                init.constant(INV_SOFTPLUS_1),
                # ^ this value is softplus^{-1}(1), i.e., σ starts at 1.
                (self.latent_dim,)
            ))  # type: ignore
        )
        # q(Z|X)
        self.q_Z_given_X = ConvEncoder(latent_dim=self.latent_dim, **(self.encoder or {}))
        # p(X̂|Z)
        self.p_Xhat_given_Z = ConvDecoder(image_shape=self.image_shape, **(self.decoder or {}))
        # p(Θ|Z)
        p_Θ_given_Z_base = DenseEncoder(latent_dim=1, **(self.θ_decoder or {}))
        p_Θ_given_Z_bij = Bijector(num_layers=2, num_bins=4, **(self.θ_bijector or {}))
        self.p_Θ_given_Z = lambda z: distrax.Transformed(p_Θ_given_Z_base(z), p_Θ_given_Z_bij(z))
        # q(Θ|X)
        q_Θ_given_X_base = DenseEncoder(latent_dim=1, **(self.θ_decoder or {}))
        q_Θ_given_X_bij = Bijector(num_layers=2, num_bins=4, **(self.θ_bijector or {}))
        self.q_Θ_given_X = lambda x: distrax.Transformed(q_Θ_given_X_base(x), q_Θ_given_X_bij(x))

    def __call__(self, x, rng):
        q_Z_given_x = make_approx_invariant(self.q_Z_given_X, x, 10, rng)
        z = q_Z_given_x.sample(seed=rng)

        p_Θ_given_z = self.p_Θ_given_Z(z)

        q_Θ_given_x = self.q_Θ_given_X(x)
        θ = q_Θ_given_x.sample(seed=rng)[0]

        p_Xhat_given_z = self.p_Xhat_given_Z(z)
        xhat = p_Xhat_given_z.sample(seed=rng)

        x_ = rotate_image(xhat, θ, -1)
        p_X_given_xhat_θ = distrax.Normal(x_, 1.)

        return q_Z_given_x, q_Θ_given_x, p_X_given_xhat_θ, p_Xhat_given_z, p_Θ_given_z, self.p_Z

    def sample(self, rng, prototype=False, sample_xhat=False, sample_θ=False):
        z = self.p_Z.sample(seed=rng)

        p_Xhat_given_z = self.p_Xhat_given_Z(z)
        xhat = p_Xhat_given_z.sample(seed=rng) if sample_xhat else p_Xhat_given_z.mean()
        if prototype:
            return xhat

        p_Θ_given_z = self.p_Θ_given_Z(z)
        θ = p_Θ_given_z.sample(seed=rng)[0] if sample_θ else p_Θ_given_z.mean()[0]

        x = rotate_image(xhat, θ, -1)
        return x

    def reconstruct(self, x, rng, prototype=False, sample_z=False,
                    sample_xhat=False, sample_θ=False):
        q_Z_given_x = make_approx_invariant(self.q_Z_given_X, x, 10, rng)
        z = q_Z_given_x.sample(seed=rng) if sample_z else q_Z_given_x.mean()

        p_Xhat_given_z = self.p_Xhat_given_Z(z)
        xhat = p_Xhat_given_z.sample(seed=rng) if sample_xhat else p_Xhat_given_z.mean()
        if prototype:
            return xhat

        q_Θ_given_x = self.q_Θ_given_X(x)
        θ = q_Θ_given_x.sample(seed=rng)[0] if sample_θ else q_Θ_given_x.mean()[0]

        x_recon = rotate_image(xhat, θ, -1)
        return x_recon


# TODO: generalize to other transformations.
def make_approx_invariant(p_Z_given_X, x, num_samples, rng):
    """Construct an approximately invariant distribution by sampling parameters
    of the distribution for rotated inputs and then averaging.

    Args:
        p_Z_given_X: A distribution whose parameters are a function of x.
        x: An image.
        num_samples: The number of samples to take.
        rng: A random number generator.

    Returns:
        An approximately invariant distribution of the same type as p.
    """
    p_Θ = distrax.Uniform(low=-jnp.pi, high=jnp.pi)
    rngs = random.split(rng, num_samples)

    def sample_params(x, rng):
        θ = p_Θ.sample(seed=rng)
        x_ = rotate_image(x, θ, -1)
        p_Z_given_x_ = p_Z_given_X(x_)
        assert type(p_Z_given_x_) == distrax.Normal
        # TODO: generalise to other distributions.
        return p_Z_given_x_.loc, p_Z_given_x_.scale

    params = jax.vmap(sample_params, in_axes=(None, 0))(x, rngs)
    params = tree_map(lambda x: jnp.mean(x, axis=0), params)

    return distrax.Normal(*params)


def calculate_livae_elbo(x, q_Z_given_x, q_Θ_given_x, p_X_given_xhat_θ, p_Θ_given_z, p_Z, β=1.):
    ll = p_X_given_xhat_θ.log_prob(x).sum()
    z_kld = q_Z_given_x.kl_divergence(p_Z).sum()
    θ_kld = q_Θ_given_x.kl_divergence(p_Θ_given_z).sum()

    elbo = ll - β * z_kld - θ_kld
    # TODO: add beta term for θ_kld? Use same beta?

    return {'loss': -elbo, 'elbo': elbo, 'll': ll, 'z_kld': z_kld, 'θ_kld': θ_kld}


def make_livae_loss(
    model: LIVAE,
    x_batch: Array,
) -> Callable:
    def batch_loss(params, batch_rng, β: float = 1.):
        # TODO: this loss function is a 1 sample estimate, add an option for more samples?
        # Define loss func for 1 example.
        def loss_fn(x):
            rng = random.fold_in(batch_rng, lax.axis_index('batch'))
            q_Z_given_x, q_Θ_given_x, p_X_given_xhat_θ, _, p_Θ_given_z, p_Z = model.apply(
                {'params': params}, x, rng,
            )

            metrics = calculate_livae_elbo(x, q_Z_given_x, q_Θ_given_x, p_X_given_xhat_θ, p_Θ_given_z, p_Z, β)

            return metrics

        # Broadcast over batch and take aggregate.
        batch_metrics = jax.vmap(
            loss_fn, out_axes=(0), in_axes=(0), axis_name='batch'
        )(x_batch)
        batch_metrics = tree_map(lambda x: x.mean(axis=0), batch_metrics)
        return batch_metrics['loss'], (batch_metrics)

    return jax.jit(batch_loss)
