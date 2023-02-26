"""Learnt-Invariance VAE definition. Without Conjecture 3, i.e., using a P(Η) rather than P(Η|Z).

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

from src.transformations.affine import transform_image
from src.models.common import (
    ConvEncoder,
    ConvDecoder,
    DenseEncoder,
    Bijector,
    INV_SOFTPLUS_1,
    make_approx_invariant,
    make_η_bounded,
)
from src.models.livae import (
    make_livae_reconstruction_plot as make_livae_no_conj3_reconstruction_plot,
)
from src.models.livae import make_livae_sampling_plot as make_livae_no_conj3_sampling_plot

KwArgs = Mapping[str, Any]
PRNGKey = Any


class LIVAE_NO_CONJ3(nn.Module):
    latent_dim: int = 20
    image_shape: Tuple[int, int, int] = (28, 28, 1)
    Z_given_X: Optional[KwArgs] = None
    Xhat_given_Z: Optional[KwArgs] = None
    Eta: Optional[KwArgs] = None
    Eta_given_X: Optional[KwArgs] = None
    bounds: Tuple[float, float, float, float, float, float, float] = (
        0.25,
        0.25,
        jnp.pi,
        0.25,
        0.25,
        jnp.pi / 6,
        jnp.pi / 6,
    )

    def setup(self):
        self.bounds_array = jnp.array(self.bounds)
        # p(Z)
        self.p_Z = distrax.Normal(
            loc=self.param("prior_μ", init.zeros, (self.latent_dim,)),
            scale=jax.nn.softplus(
                self.param(
                    "prior_σ_",
                    init.constant(INV_SOFTPLUS_1),
                    # ^ this value is softplus^{-1}(1), i.e., σ starts at 1.
                    (self.latent_dim,),
                )
            ),  # type: ignore
        )
        # q(Z|X)
        self.q_Z_given_X = ConvEncoder(latent_dim=self.latent_dim, **(self.Z_given_X or {}))
        # p(X̂|Z)
        self.p_Xhat_given_Z = ConvDecoder(image_shape=self.image_shape, **(self.Xhat_given_Z or {}))
        # p(Η|Z)
        self.p_Η_base = DenseEncoder(latent_dim=7, **(self.Eta or {}).get("base", {}))
        self.p_Η_bij = Bijector(**(self.Eta or {}).get("bijector", {}))
        # q(Η|X)
        self.q_Η_given_X_base = DenseEncoder(
            latent_dim=7, **(self.Eta_given_X or {}).get("base", {})
        )
        self.q_Η_given_X_bij = Bijector(**(self.Eta_given_X or {}).get("bijector", {}))

    def __call__(self, x: Array, rng: PRNGKey, α: float = 1.0) -> Tuple[distrax.Distribution, ...]:
        inv_rng, z_rng, xhat_rng, η_rng = random.split(rng, 4)
        q_Z_given_x = make_approx_invariant(self.q_Z_given_X, x, 10, inv_rng, self.bounds_array, α)
        z = q_Z_given_x.sample(seed=z_rng)

        x_ = jnp.sum(z) * 0 + x
        # TODO: this ^ is a hack to get jaxprs to match up. It isn't clear why this is necessary.
        # But without the hack the jaxpr for the p_Η_given_Z_bij is different from the jaxpr for the
        # q_Η_given_X_bij, where the difference comes from registers being on device0 vs not being
        # commited to a device. This is a problem because the take the KLD between the p_Η_given_Z
        # and the q_Η_given_x, the jaxprs must match up.

        p_Η = distrax.Transformed(self.p_Η_base(jnp.zeros_like(z)), self.p_Η_bij(jnp.zeros_like(z)))

        q_Η_given_x = distrax.Transformed(self.q_Η_given_X_base(x_), self.q_Η_given_X_bij(x_))
        η = q_Η_given_x.sample(seed=η_rng)
        η = make_η_bounded(η, self.bounds_array * α)

        p_Xhat_given_z = self.p_Xhat_given_Z(z)
        xhat = p_Xhat_given_z.sample(seed=xhat_rng)

        p_X_given_xhat_η = distrax.Normal(transform_image(xhat, η), 1.0)

        return q_Z_given_x, q_Η_given_x, p_X_given_xhat_η, p_Xhat_given_z, p_Η, self.p_Z

    def sample(
        self,
        rng: PRNGKey,
        prototype: bool = False,
        sample_xhat: bool = False,
        sample_η: bool = False,
        α: float = 1.0,
    ) -> Array:
        z_rng, xhat_rng, η_rng = random.split(rng, 3)
        z = self.p_Z.sample(seed=z_rng)

        p_Xhat_given_z = self.p_Xhat_given_Z(z)
        xhat = p_Xhat_given_z.sample(seed=xhat_rng) if sample_xhat else p_Xhat_given_z.mean()
        if prototype:
            return xhat

        p_Η = distrax.Transformed(self.p_Η_base(jnp.zeros_like(z)), self.p_Η_bij(jnp.zeros_like(z)))
        η = p_Η.sample(seed=η_rng) if sample_η else p_Η.mean()
        η = make_η_bounded(η, self.bounds_array * α)

        x = transform_image(xhat, η)
        return x

    def reconstruct(
        self,
        x: Array,
        rng: PRNGKey,
        prototype: bool = False,
        sample_z: bool = False,
        sample_xhat: bool = False,
        sample_η: bool = False,
        α: float = 1.0,
    ) -> Array:
        z_rng, xhat_rng, η_rng = random.split(rng, 3)
        q_Z_given_x = make_approx_invariant(self.q_Z_given_X, x, 100, z_rng, self.bounds, α)
        z = q_Z_given_x.sample(seed=rng) if sample_z else q_Z_given_x.mean()

        p_Xhat_given_z = self.p_Xhat_given_Z(z)
        xhat = p_Xhat_given_z.sample(seed=xhat_rng) if sample_xhat else p_Xhat_given_z.mean()
        if prototype:
            return xhat

        q_Η_given_x = distrax.Transformed(self.q_Η_given_X_base(x), self.q_Η_given_X_bij(x))
        η = q_Η_given_x.sample(seed=η_rng) if sample_η else q_Η_given_x.mean()
        η = make_η_bounded(η, self.bounds_array * α)

        x_recon = transform_image(xhat, η)
        return x_recon


def calculate_livae_no_conj3_elbo(
    x: Array,
    q_Z_given_x: distrax.Distribution,
    q_Η_given_x: distrax.Distribution,
    p_X_given_xhat_η: distrax.Distribution,
    p_Η: distrax.Distribution,
    p_Z: distrax.Distribution,
    β: float = 1.0,
) -> Tuple[float, Mapping[str, float]]:
    ll = p_X_given_xhat_η.log_prob(x).sum()
    z_kld = q_Z_given_x.kl_divergence(p_Z).sum()

    η_kld = q_Η_given_x.kl_divergence(p_Η).sum()

    elbo = ll - β * z_kld - η_kld
    # TODO: add beta term for η_kld? Use same beta?

    return -elbo, {"elbo": elbo, "ll": ll, "z_kld": z_kld, "η_kld": η_kld}


def livae_no_conj3_loss_fn(
    model: nn.Module, params: nn.FrozenDict, x: Array, rng: PRNGKey, β: float = 1.0, α: float = 1.0
) -> Tuple[float, Mapping[str, float]]:
    """Single example loss function for LIVAE."""
    # TODO: this loss function is a 1 sample estimate, add an option for more samples?
    rng_local = random.fold_in(rng, lax.axis_index("batch"))
    q_Z_given_x, q_Η_given_x, p_X_given_xhat_η, _, p_Η, p_Z = model.apply(
        {"params": params},
        x,
        rng_local,
        α,
    )

    loss, metrics = calculate_livae_no_conj3_elbo(
        x, q_Z_given_x, q_Η_given_x, p_X_given_xhat_η, p_Η, p_Z, β
    )

    return loss, metrics


def make_livae_no_conj3_batch_loss(model, agg=jnp.mean):
    @jax.jit
    def batch_loss(params, x_batch, mask, rng, state):
        # Broadcast loss over batch and aggregate.
        loss, metrics = jax.vmap(
            livae_no_conj3_loss_fn, in_axes=(None, None, 0, None, None, None), axis_name="batch"  # type: ignore
        )(model, params, x_batch, rng, state.β, state.α)
        loss, metrics, mask = jax.tree_util.tree_map(partial(agg, axis=0), (loss, metrics, mask))
        return loss, (metrics, mask)

    return batch_loss
