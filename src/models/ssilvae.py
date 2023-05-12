"""Self-Supervised Invariance Learner VAE.

A note on notation. In order to distinguish between random variables and their values, we use upper
and lower case variable names. I.e., p(Z) or `p_Z` is the distribution over the r.v. Z, and is a
function, while p(z) or `p_z` is the probability that Z=z. Similarly, p(X|Z) or `p_X_given_Z` is a
a function which returns another function p(X|z) or `p_X_given_z`, which would return the proability
that X=x|Z=z a.k.k `p_x_given_z`.
"""

from typing import Any, Mapping, Optional, Sequence, Tuple
from functools import partial

import jax
from jax import numpy as jnp
from jax import random, lax
from chex import Array
from flax import linen as nn
import flax.linen.initializers as init
import distrax
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from src.transformations import transform_image
from src.models.common import (
    Encoder,
    Decoder,
    Flow,
    INV_SOFTPLUS_1,
    approximate_mode,
)
import src.utils.plotting as plot_utils

KwArgs = Mapping[str, Any]
PRNGKey = Any


class SSILVAE(nn.Module):
    latent_dim: int = 20
    image_shape: Tuple[int, int, int] = (28, 28, 1)
    Η_given_Xhat: Optional[KwArgs] = None
    Η_given_X: Optional[KwArgs] = None
    Z_given_Xhat: Optional[KwArgs] = None
    Xhat_given_Z: Optional[KwArgs] = None
    bounds: Optional[Sequence[float]] = None
    offset: Optional[Sequence[float]] = None
    σ_min: float = 1e-2

    def setup(self):
        self.bounds_array = jnp.array(self.bounds) if self.bounds else None
        self.offset_array = jnp.array(self.offset) if self.offset else None
        # p(Η|Xhat)
        self.p_Η_given_Xhat = Flow(
            **(self.Η_given_Xhat or {}),
            bounds_array=self.bounds_array,
            offset_array=self.offset_array,
            event_shape=self.bounds_array.shape,
        )
        # q(Η|X)
        self.q_Η_given_X = Flow(
            **(self.Η_given_X or {}),
            bounds_array=self.bounds_array,
            offset_array=self.offset_array,
            event_shape=self.bounds_array.shape,
        )
        self.σ = jax.nn.softplus(
            self.param(
                "σ_",
                init.constant(INV_SOFTPLUS_1),
                # ^ this value is softplus^{-1}(1), i.e., σ starts at 1.
                self.image_shape,
            )
        ).clip(min=self.σ_min)
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
        # q(Z|Xhat)
        self.q_Z_given_Xhat = Encoder(latent_dim=self.latent_dim, **(self.Z_given_Xhat or {}))
        # p(Xhat|Z)
        self.p_Xhat_given_Z = Decoder(image_shape=self.image_shape, **(self.Xhat_given_Z or {}))

    def __call__(
        self, x: Array, rng: PRNGKey, α: float = 1.0, train: bool = True
    ) -> Tuple[distrax.Distribution, ...]:
        η_rng, η_unif_rng, η_tot_rng, z_rng = random.split(rng, 4)

        Η_uniform = distrax.Uniform(
            low=-α * self.bounds_array + self.offset_array,
            high=α * self.bounds_array + self.offset_array,
        )
        η_uniform = Η_uniform.sample(seed=η_unif_rng)

        x_uniform = transform_image(x, η_uniform)

        q_Η_given_x_uniform = self.q_Η_given_X(x_uniform, train=train)
        η_tot = q_Η_given_x_uniform.sample(seed=η_tot_rng)

        xhat = transform_image(x_uniform, -η_tot)

        p_Η_given_xhat = self.p_Η_given_Xhat(xhat, train=train)
        η_ = p_Η_given_xhat.sample(seed=η_rng)

        q_Η_given_x = self.q_Η_given_X(x, train=train)
        η = q_Η_given_x.sample(seed=η_rng)

        p_X_given_xhat_and_η = distrax.Independent(
            distrax.Normal(transform_image(xhat, η), self.σ),
            reinterpreted_batch_ndims=len(x.shape),
        )
        # TODO: this requires applying transform_image twice to the same input, but we could do it as a single call to two different inputs.

        q_Z_given_xhat = self.q_Z_given_Xhat(lax.stop_gradient(xhat), train=train)
        z = q_Z_given_xhat.sample(seed=z_rng)

        p_Xhat_given_z = self.p_Xhat_given_Z(z, train=train)

        return (
            η,
            η_,
            z,
            p_X_given_xhat_and_η,
            p_Η_given_xhat,
            p_Xhat_given_z,
            self.p_Z,
            q_Η_given_x,
            q_Z_given_xhat,
        )

    def sample(
        self,
        rng: PRNGKey,
        return_xhat: bool = False,
        sample_xhat: bool = True,
        sample_η: bool = True,
        sample_x: bool = False,
        train: bool = True,
    ) -> Array:
        z_rng, xhat_rng, η_rng = random.split(rng, 3)
        z = self.p_Z.sample(seed=z_rng)

        p_Xhat_given_z = self.p_Xhat_given_Z(z, train=train)
        if sample_xhat:
            xhat = p_Xhat_given_z.sample(seed=xhat_rng)
        else:
            xhat = p_Xhat_given_z.mode()

        if return_xhat:
            return xhat

        p_Η_given_xhat = self.p_Η_given_Xhat(xhat, train=train)
        if sample_η:
            η = p_Η_given_xhat.sample(seed=η_rng)
        else:
            η = approximate_mode(p_Η_given_xhat, 100, rng=η_rng)

        p_X_given_xhat_and_η = distrax.Independent(
            distrax.Normal(transform_image(xhat, η), self.σ),
            reinterpreted_batch_ndims=len(xhat.shape),
        )
        if sample_x:
            x = p_X_given_xhat_and_η.sample(seed=rng)
        else:
            x = p_X_given_xhat_and_η.mode()

        return x

    def reconstruct(
        self,
        x: Array,
        rng: PRNGKey,
        return_xhat: bool = False,
        reconstruct_xhat: bool = True,
        sample_η_proto: bool = False,
        sample_η_recon: bool = False,
        sample_z: bool = False,
        sample_xrecon: bool = False,
        sample_xhat: bool = True,
        α: float = 1.0,
        train: bool = True,
    ) -> Array:
        η1_rng, η2_rng, xrecon_rng, xhat_rng, z_rng = random.split(rng, 5)

        q_Η_given_x = self.q_Η_given_X(x, train=train)
        # TODO: should this be a randomly transformed x?
        if sample_η_proto:
            η1 = q_Η_given_x.sample(seed=η1_rng)
        else:
            η1 = approximate_mode(q_Η_given_x, 100, rng=η1_rng)

        xhat_ = transform_image(x, -η1)
        if reconstruct_xhat:
            q_Z_given_xhat = self.q_Z_given_Xhat(xhat_, train=train)
            if sample_z:
                z = q_Z_given_xhat.sample(seed=z_rng)
            else:
                z = q_Z_given_xhat.mode()

            p_Xhat_given_z = self.p_Xhat_given_Z(z, train=train)
            if sample_xhat:
                xhat = p_Xhat_given_z.sample(seed=xhat_rng)
            else:
                xhat = p_Xhat_given_z.mode()
        else:
            xhat = xhat_

        if return_xhat:
            return xhat

        p_Η_given_xhat = self.p_Η_given_Xhat(xhat, train=train)
        # TODO: should we use a randomly transformed x here (and in the ELBO)? I.e., single sample monte carlo estimate of the invariant function.
        if sample_η_recon:
            η2 = p_Η_given_xhat.sample(seed=η2_rng)
        else:
            η2 = approximate_mode(p_Η_given_xhat, 100, rng=η2_rng)

        p_Xrecon_given_xhat_and_η = distrax.Independent(
            distrax.Normal(transform_image(xhat, η2), self.σ),
            reinterpreted_batch_ndims=len(x.shape),
        )
        if sample_xrecon:
            xrecon = p_Xrecon_given_xhat_and_η.sample(seed=xrecon_rng)
        else:
            xrecon = p_Xrecon_given_xhat_and_η.mean()

        return xrecon


def calculate_ssilvae_elbo(
    x: Array,
    η: Array,
    p_X_given_xhat_and_η: distrax.Distribution,
    p_Η_given_xhat: distrax.Distribution,
    p_Xhat_given_z: distrax.Distribution,
    p_Z: distrax.Distribution,
    q_Η_given_x: distrax.Distribution,
    q_Z_given_xhat: distrax.Distribution,
    rng: PRNGKey,
    bounds: Array,
    offset: Array,
    β: float = 1.0,
    γ: float = 1.0,
    n: int = 100,
    # TODO: more? less?
    ε: float = 1e-6,
) -> Tuple[float, Mapping[str, float]]:
    ll_unorm = p_X_given_xhat_and_η.log_prob(x)
    # Normalise by the number of channels. Maybe we should normalise by the number of pixels instead?
    ll = ll_unorm / x.shape[-1]

    η_qs, q_Η_log_probs = q_Η_given_x.sample_and_log_prob(seed=rng, sample_shape=(n,))
    q_H_entropy = -jnp.mean(q_Η_log_probs, axis=0)

    p_Η_log_probs = jax.vmap(p_Η_given_xhat.log_prob)(
        η_qs.clip(min=(1 - ε) * (-bounds) + offset, max=(1 - ε) * (bounds) + offset)
    )
    p_q_H_cross_entropy = -jnp.mean(p_Η_log_probs, axis=0)

    η_kld = p_q_H_cross_entropy - q_H_entropy

    def norm(η: Array) -> float:
        return jnp.sum(η**2) / len(η)

    q_η_norm = jax.vmap(norm)(η_qs).mean()

    η_ps = p_Η_given_xhat.sample(seed=rng, sample_shape=(n,))
    p_η_norm = jax.vmap(norm)(η_ps).mean()

    z_kld = q_Z_given_xhat.kl_divergence(p_Z)
    xhat_ll_unorm = p_Xhat_given_z.log_prob(lax.stop_gradient(transform_image(x, -η)))
    xhat_ll = xhat_ll_unorm / x.shape[-1]

    elbo = ll - η_kld - γ * (q_η_norm + p_η_norm) + q_H_entropy + xhat_ll - β * z_kld

    return -elbo, {
        "elbo": elbo,
        "ll": ll,
        "ll_unorm": ll_unorm,
        "η_kld": η_kld,
        "q_η_ce": q_η_norm,
        "p_η_ce": p_η_norm,
        "q_H_ent": q_H_entropy,
        "p_q_H_cent": p_q_H_cross_entropy,
        "xhat_ll": xhat_ll,
        "xhat_ll_unorm": xhat_ll_unorm,
        "z_kld": z_kld,
    }


def ssil_loss_fn(
    model: nn.Module,
    params: nn.FrozenDict,
    x: Array,
    rng: PRNGKey,
    α: float = 1.0,
    β: float = 1.0,
    γ: float = 1.0,
    train: bool = True,
) -> Tuple[float, Mapping[str, float]]:
    """Single example loss function for Contrastive Invariance Learner."""
    # TODO: this loss function is a 1 sample estimate, add an option for more samples?
    rng_local = random.fold_in(rng, lax.axis_index("batch"))
    rng_apply, rng_loss, rng_dropout = random.split(rng_local, 3)
    (
        η,
        _,
        _,
        p_X_given_xhat_and_η,
        p_Η_given_xhat,
        p_Xhat_given_z,
        p_Z,
        q_Η_given_x,
        q_Z_given_xhat,
    ) = model.apply({"params": params}, x, rng_apply, α, train=train, rngs={"dropout": rng_dropout})

    loss, metrics = calculate_ssilvae_elbo(
        x,
        η,
        p_X_given_xhat_and_η,
        p_Η_given_xhat,
        p_Xhat_given_z,
        p_Z,
        q_Η_given_x,
        q_Z_given_xhat,
        rng_loss,
        jnp.array(model.bounds),
        jnp.array(model.offset),
        β,
        γ,
    )

    return loss, metrics


def make_ssilvae_batch_loss(model, agg=jnp.mean, train=True):
    def batch_loss(params, x_batch, mask, rng, state):
        # Broadcast loss over batch and aggregate.
        loss, metrics = jax.vmap(
            ssil_loss_fn, in_axes=(None, None, 0, None, None, None, None, None), axis_name="batch"  # type: ignore
        )(model, params, x_batch, rng, state.α, state.β, state.γ, train)
        loss, metrics, mask = jax.tree_util.tree_map(partial(agg, axis=0), (loss, metrics, mask))
        return loss, (metrics, mask)

    return batch_loss


def make_ssilvae_reconstruction_plot(x, n_visualize, model, state, visualisation_rng, train=False):
    def reconstruct(x, return_xhat, reconstruct_xhat):
        rng = random.fold_in(visualisation_rng, jax.lax.axis_index("image"))  # type: ignore
        rng_apply, rng_dropout = random.split(rng)
        return model.apply(
            {"params": state.params},
            x,
            rng_apply,
            return_xhat=return_xhat,
            reconstruct_xhat=reconstruct_xhat,
            sample_η_proto=False,
            sample_η_recon=False,
            sample_z=False,
            sample_xrecon=False,
            sample_xhat=False,
            α=state.α,
            train=train,
            method=model.reconstruct,
            rngs={"dropout": rng_dropout},
        )

    x_proto = jax.vmap(reconstruct, axis_name="image", in_axes=(0, None, None))(  # type: ignore
        x, True, False
    )

    x_recon = jax.vmap(reconstruct, axis_name="image", in_axes=(0, None, None))(  # type: ignore
        x, False, False
    )

    x_proto_vae = jax.vmap(reconstruct, axis_name="image", in_axes=(0, None, None))(  # type: ignore
        x, True, True
    )

    recon_fig = plot_utils.plot_img_array(
        jnp.concatenate((x, x_proto, x_recon, x_proto_vae), axis=0),
        ncol=n_visualize,  # type: ignore
        pad_value=1,
        padding=1,
        title="Orig | Proto | Recon | Proto (recon)",
    )

    return recon_fig


def make_ssilvae_sampling_plot(n_visualize, model, state, visualisation_rng, train=False):
    def sample(rng, return_xhat, sample_xhat):
        rng_apply, rng_dropout = random.split(rng)
        return model.apply(
            {"params": state.params},
            rng_apply,
            return_xhat=return_xhat,
            sample_xhat=sample_xhat,
            sample_η=True,
            sample_x=False,
            train=train,
            method=model.sample,
            rngs={"dropout": rng_dropout},
        )

    sampled_xhats = jax.vmap(sample, in_axes=(0, None, None))(  # type: ignore
        jax.random.split(visualisation_rng, n_visualize), True, True  # type: ignore
    )

    sampled_xs = jax.vmap(sample, in_axes=(0, None, None))(  # type: ignore
        jax.random.split(visualisation_rng, n_visualize), False, True  # type: ignore
    )

    sampled_xs_mode = jax.vmap(sample, in_axes=(0, None, None))(  # type: ignore
        jax.random.split(visualisation_rng, n_visualize), False, False  # type: ignore
    )

    sample_fig = plot_utils.plot_img_array(
        jnp.concatenate((sampled_xhats, sampled_xs, sampled_xs_mode), axis=0),
        ncol=n_visualize,  # type: ignore
        pad_value=1,
        padding=1,
        title="Proto | X | X (proto mode)",
    )
    return sample_fig


def make_ssilvae_summary_plot(config, final_state, x, rng):
    """Make a summary plot of the model."""
    return None
