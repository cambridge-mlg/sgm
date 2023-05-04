"""VAE implementation.

A note on notation. In order to distinguish between random variables and their values, we use upper
and lower case variable names. I.e., p(Z) or `p_Z` is the distribution over the r.v. Z, and is a
function, while p(z) or `p_z` is the probability that Z=z. Similarly, p(X|Z) or `p_X_given_Z` is a
a function which returns another function p(X|z) or `p_X_given_z`, which would return the proability
that X=x|Z=z a.k.k `p_x_given_z`.
"""

from typing import Any, Mapping, Optional, Tuple
from functools import partial

import jax
from jax import numpy as jnp
from jax import random, lax
from chex import Array
from flax import linen as nn
import flax.linen.initializers as init
import distrax

from src.models.common import Encoder, Decoder, INV_SOFTPLUS_1
import src.utils.plotting as plot_utils

KwArgs = Mapping[str, Any]
PRNGKey = Any


class VAE(nn.Module):
    latent_dim: int = 20
    image_shape: Tuple[int, int, int] = (28, 28, 1)
    Z_given_X: Optional[KwArgs] = None
    X_given_Z: Optional[KwArgs] = None
    σ_min: float = 1e-2

    def setup(self):
        # p(Z)
        self.p_Z = distrax.Independent(
            distrax.Normal(
                loc=self.param("prior_μ", init.zeros, (self.latent_dim,)),
                scale=jax.nn.softplus(
                    self.param(
                        "prior_σ_",
                        init.constant(INV_SOFTPLUS_1),
                        # ^ this value is softplus^{-1}(1), i.e., σ starts at 1.
                        (self.latent_dim,),
                    )
                ).clip(
                    min=self.σ_min
                ),  # type: ignore
            ),
            reinterpreted_batch_ndims=1,
        )
        # q(Z|X)
        self.q_Z_given_X = Encoder(latent_dim=self.latent_dim, **(self.Z_given_X or {}))
        # p(X|Z)
        self.p_X_given_Z = Decoder(image_shape=self.image_shape, **(self.X_given_Z or {}))

    def __call__(
        self, x: Array, rng: PRNGKey, train: bool = True
    ) -> Tuple[distrax.Distribution, ...]:
        q_Z_given_x = self.q_Z_given_X(x, train=train)
        z = q_Z_given_x.sample(seed=rng)

        p_X_given_z = self.p_X_given_Z(z, train=train)

        return q_Z_given_x, p_X_given_z, self.p_Z

    def logp_x_given_z(self, x: Array, z: Array, train: bool = True) -> Array:
        return self.p_X_given_Z(z, train=train).log_prob(x)

    def logq_z_given_x(self, x: Array, z: Array, train: bool = True) -> Array:
        return self.q_Z_given_X(x, train=train).log_prob(z)

    def logp_z(self, z: Array) -> Array:
        return self.p_Z.log_prob(z)

    def sample(
        self,
        rng: PRNGKey,
        sample_x: bool = False,
        train: bool = True,
    ) -> Array:
        z_rng, x_rng = random.split(rng, 2)
        z = self.p_Z.sample(seed=z_rng)

        p_X_given_z = self.p_X_given_Z(z, train=train)
        if sample_x:
            x = p_X_given_z.sample(seed=x_rng)
        else:
            x = p_X_given_z.mode()

        return x

    def reconstruct(
        self,
        x: Array,
        rng: PRNGKey,
        sample_z: bool = False,
        sample_xrecon: bool = False,
        train: bool = True,
    ) -> Array:
        z_rng, x_rng = random.split(rng, 2)
        q_Z_given_x = self.q_Z_given_X(x, train=train)
        if sample_z:
            z = q_Z_given_x.sample(seed=z_rng)
        else:
            z = q_Z_given_x.mode()

        p_X_given_z = self.p_X_given_Z(z, train=train)
        if sample_xrecon:
            x_recon = p_X_given_z.sample(seed=x_rng)
        else:
            x_recon = p_X_given_z.mode()

        return x_recon


def calculate_vae_elbo(
    x: Array,
    q_Z_given_x: distrax.Distribution,
    p_X_given_z: distrax.Distribution,
    p_Z: distrax.Distribution,
    β: float = 1.0,
) -> Tuple[float, Mapping[str, float]]:
    ll = p_X_given_z.log_prob(x) / x.shape[-1]
    z_kld = q_Z_given_x.kl_divergence(p_Z)

    elbo = ll - β * z_kld

    return -elbo, {"elbo": elbo, "ll": ll, "z_kld": z_kld}


def vae_loss_fn(
    model: nn.Module,
    params: nn.FrozenDict,
    x: Array,
    rng: PRNGKey,
    β: float = 1.0,
    train: bool = True,
    **kwargs
) -> Tuple[float, Mapping[str, float]]:
    """Single example loss function for VAE."""
    # TODO: this loss function is a 1 sample estimate, add an option for more samples?
    rng_local = random.fold_in(rng, lax.axis_index("batch"))
    q_Z_given_x, p_X_given_z, p_Z = model.apply(
        {"params": params},
        x,
        rng_local,
        train,
    )

    loss, metrics = calculate_vae_elbo(x, q_Z_given_x, p_X_given_z, p_Z, β)

    return loss, metrics


def make_vae_batch_loss(model, agg=jnp.mean, train=True):
    def batch_loss(params, x_batch, mask, rng, state):
        # Broadcast loss over batch and aggregate.
        loss, metrics = jax.vmap(
            vae_loss_fn, in_axes=(None, None, 0, None, None, None), axis_name="batch"  # type: ignore
        )(model, params, x_batch, rng, state.β, train)
        loss, metrics, mask = jax.tree_util.tree_map(partial(agg, axis=0), (loss, metrics, mask))
        return loss, (metrics, mask)

    return batch_loss


def make_vae_reconstruction_plot(x, n_visualize, model, state, visualisation_rng):
    def reconstruct(x, sample_z=False, sample_xrecon=False):
        rng = random.fold_in(visualisation_rng, jax.lax.axis_index("image"))  # type: ignore
        return model.apply(
            {"params": state.params},
            x,
            rng,
            sample_z=sample_z,
            sample_xrecon=sample_xrecon,
            method=model.reconstruct,
        )

    x_recon_modes = jax.vmap(
        reconstruct, axis_name="image", in_axes=(0, None, None)  # type: ignore
    )(x, False, True)

    recon_fig = plot_utils.plot_img_array(
        jnp.concatenate((x, x_recon_modes), axis=0),
        ncol=n_visualize,  # type: ignore
        pad_value=1,
        padding=1,
        title="Original | Reconstruction",
    )

    return recon_fig


def make_vae_sampling_plot(n_visualize, model, state, visualisation_rng):
    def sample(rng, sample_x=False):
        return model.apply(
            {"params": state.params},
            rng,
            sample_x=sample_x,
            method=model.sample,
        )

    sampled_data = jax.vmap(sample, in_axes=(0, None))(  # type: ignore
        jax.random.split(visualisation_rng, n_visualize), True  # type: ignore
    )

    sample_fig = plot_utils.plot_img_array(
        sampled_data,
        ncol=n_visualize,  # type: ignore
        pad_value=1,
        padding=1,
        title="Sampled data",
    )
    return sample_fig


def make_vae_summary_plot(config, final_state, x, rng):
    return None
