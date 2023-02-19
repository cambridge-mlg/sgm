"""Invariant VAE definition.

A note on notation. In order to distinguish between random variables and their values, we use upper
and lower case variable names. I.e., p(Z) or `p_Z` is the distribution over the r.v. Z, and is a
function, while p(z) or `p_z` is the probability t Z=z. Similarly, p(X|Z) or `p_X_given_Z` is a
a function which returns another function p(X|z) or `p_X_given_z`, which would return the proability
t X=x|Z=z a.k.k `p_x_given_z`.
"""

from typing import Any, Mapping, Optional, Tuple

import jax
from jax import numpy as jnp
from jax import random, lax
from chex import Array
from flax import linen as nn
import flax.linen.initializers as init
import distrax

from src.models.common import ConvEncoder, ConvDecoder, INV_SOFTPLUS_1, make_approx_invariant

KwArgs = Mapping[str, Any]
PRNGKey = Any


class LIVAE(nn.Module):
    latent_dim: int = 20
    image_shape: Tuple[int, int, int] = (28, 28, 1)
    Z_given_X: Optional[KwArgs] = None
    X_given_Z: Optional[KwArgs] = None
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
        # p(X|Z)
        self.p_X_given_Z = ConvDecoder(image_shape=self.image_shape, **(self.X_given_Z or {}))

    def __call__(self, x: Array, rng: PRNGKey, α: float = 1.0) -> Tuple[distrax.Distribution, ...]:
        inv_rng, z_rng, x_rng, η_rng = random.split(rng, 4)
        q_Z_given_x = make_approx_invariant(self.q_Z_given_X, x, 10, inv_rng, self.bounds_array, α)
        z = q_Z_given_x.sample(seed=z_rng)

        p_X_given_z = self.p_X_given_Z(z)

        return q_Z_given_x, p_X_given_z, self.p_Z

    def sample(
        self,
        rng: PRNGKey,
        sample_x: bool = False,
        α: float = 1.0,
    ) -> Array:
        z_rng, x_rng, η_rng = random.split(rng, 3)
        z = self.p_Z.sample(seed=z_rng)

        p_X_given_z = self.p_X_given_Z(z)
        x = p_X_given_z.sample(seed=x_rng) if sample_x else p_X_given_z.mean()

        return x

    def reconstruct(
        self,
        x: Array,
        rng: PRNGKey,
        sample_z: bool = False,
        sample_x: bool = False,
        α: float = 1.0,
    ) -> Array:
        z_rng, x_rng = random.split(rng, 2)
        q_Z_given_x = make_approx_invariant(self.q_Z_given_X, x, 100, z_rng, self.bounds, α)
        z = q_Z_given_x.sample(seed=rng) if sample_z else q_Z_given_x.mean()

        p_X_given_z = self.p_X_given_Z(z)
        x_recon = p_X_given_z.sample(seed=x_rng) if sample_x else p_X_given_z.mean()

        return x_recon


def calculate_ivae_elbo(
    x: Array,
    q_Z_given_x: distrax.Distribution,
    p_X_given_z: distrax.Distribution,
    p_Z: distrax.Distribution,
    β: float = 1.0,
) -> Tuple[float, Mapping[str, float]]:
    ll = p_X_given_z.log_prob(x).sum()
    z_kld = q_Z_given_x.kl_divergence(p_Z).sum()

    elbo = ll - β * z_kld

    return -elbo, {"elbo": elbo, "ll": ll, "z_kld": z_kld}


def livae_loss_fn(
    model: nn.Module, params: nn.FrozenDict, x: Array, rng: PRNGKey, β: float = 1.0, α: float = 1.0
) -> Tuple[float, Mapping[str, float]]:
    """Single example loss function for LIVAE."""
    # TODO: this loss function is a 1 sample estimate, add an option for more samples?
    rng_local = random.fold_in(rng, lax.axis_index("batch"))
    q_Z_given_x, q_Η_given_x, p_X_given_x_η, _, p_Η_given_z, p_Z = model.apply(
        {"params": params},
        x,
        rng_local,
        α,
    )

    loss, metrics = calculate_ivae_elbo(
        x, q_Z_given_x, q_Η_given_x, p_X_given_x_η, p_Η_given_z, p_Z, β
    )

    return loss, metrics
