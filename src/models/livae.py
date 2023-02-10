"""Definition of the Learnt-Invariance VAE.

A note on notation. In order to distinguish between random variables and their values, we use upper
and lower case variable names. I.e., p(Z) or `p_Z` is the distribution over the r.v. Z, and is a
function, while p(z) or `p_z` is the probability that Z=z. Similarly, p(X|Z) or `p_X_given_Z` is a
a function which returns another function p(X|z) or `p_X_given_z`, which would return the proability
that X=x|Z=z a.k.k `p_x_given_z`.
"""

from typing import Any, Mapping, Optional, Tuple

import jax
from jax import numpy as jnp
from jax import random, lax
from jax.tree_util import tree_map
from chex import Array, assert_rank, assert_shape
from flax import linen as nn
import flax.linen.initializers as init
import distrax

from src.transformations.affine import transform_image
from src.models.common import ConvEncoder, ConvDecoder, DenseEncoder, Bijector, INV_SOFTPLUS_1

KwArgs = Mapping[str, Any]
PRNGKey = Any


def make_η_bounded(η, bounds):
    """Converts η to a bounded representation.

    Args:
        η: a rank-1 array of length 7.
        bounds: a rank-1 array of length 7.
    """
    assert_rank(η, 1)
    assert_shape(η, (7,))
    assert_rank(bounds, 1)
    assert_shape(bounds, (7,))

    # η = bounds * jnp.sin(η * 0.5 * (jnp.pi + 1e-8) / (bounds + 1e-8))
    η = η.clip(-bounds, bounds)

    return η


class LIVAE(nn.Module):
    latent_dim: int = 20
    image_shape: Tuple[int, int, int] = (28, 28, 1)
    Z_given_X: Optional[KwArgs] = None
    Xhat_given_Z: Optional[KwArgs] = None
    Eta_given_Z: Optional[KwArgs] = None
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
        self.p_Η_given_Z_base = DenseEncoder(
            latent_dim=7, **(self.Eta_given_Z or {}).get("base", {})
        )
        self.p_Η_given_Z_bij = Bijector(**(self.Eta_given_Z or {}).get("bijector", {}))
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

        p_Η_given_z = distrax.Transformed(self.p_Η_given_Z_base(z), self.p_Η_given_Z_bij(z))

        q_Η_given_x = distrax.Transformed(self.q_Η_given_X_base(x_), self.q_Η_given_X_bij(x_))
        η = q_Η_given_x.sample(seed=η_rng)
        η = make_η_bounded(η, self.bounds_array * α)

        p_Xhat_given_z = self.p_Xhat_given_Z(z)
        xhat = p_Xhat_given_z.sample(seed=xhat_rng)

        p_X_given_xhat_η = distrax.Normal(transform_image(xhat, η), 1.0)

        return q_Z_given_x, q_Η_given_x, p_X_given_xhat_η, p_Xhat_given_z, p_Η_given_z, self.p_Z

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

        p_Η_given_z = distrax.Transformed(self.p_Η_given_Z_base(z), self.p_Η_given_Z_bij(z))
        η = p_Η_given_z.sample(seed=η_rng) if sample_η else p_Η_given_z.mean()
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


def make_approx_invariant(
    p_Z_given_X: distrax.Normal,
    x: Array,
    num_samples: int,
    rng: PRNGKey,
    bounds: Tuple[float, float, float, float, float, float, float],
    α: float = 1.0,
) -> distrax.Normal:
    """Construct an approximately invariant distribution by sampling transformations then averaging.

    Args:
        p_Z_given_X: A distribution whose parameters are a function of x.
        x: An image.
        num_samples: The number of samples to use for the approximation.
        rng: A random number generator.

    Returns:
        An approximately invariant distribution of the same type as p.
    """
    p_Η = distrax.Uniform(low=-1 * jnp.array(bounds) * α, high=jnp.array(bounds) * α)
    rngs = random.split(rng, num_samples)
    # TODO: investigate scaling of num_samples with the size of η.

    # TODO: this function is not aware of the bounds for η.
    def sample_params(x, rng):
        η = p_Η.sample(seed=rng)
        x_ = transform_image(x, -η)
        p_Z_given_x_ = p_Z_given_X(x_)
        assert type(p_Z_given_x_) == distrax.Normal
        # TODO: generalise to other distributions.
        return p_Z_given_x_.loc, p_Z_given_x_.scale

    params = jax.vmap(sample_params, in_axes=(None, 0))(x, rngs)
    params = tree_map(lambda x: jnp.mean(x, axis=0), params)

    return distrax.Normal(*params)


def calculate_livae_elbo(
    x: Array,
    q_Z_given_x: distrax.Distribution,
    q_Η_given_x: distrax.Distribution,
    p_X_given_xhat_η: distrax.Distribution,
    p_Η_given_z: distrax.Distribution,
    p_Z: distrax.Distribution,
    β: float = 1.0,
) -> Tuple[float, Mapping[str, float]]:
    ll = p_X_given_xhat_η.log_prob(x).sum()
    z_kld = q_Z_given_x.kl_divergence(p_Z).sum()

    # dist1 = q_Η_given_x
    # input_hint1 = jnp.zeros(
    #       dist1.distribution.event_shape, dtype=dist1.distribution.dtype)
    # jaxpr_bij1 = jax.make_jaxpr(dist1.bijector.forward)(input_hint1)
    # print(jaxpr_bij1)
    # print('XXXX')
    # dist2 = p_Η_given_z
    # input_hint2 = jnp.zeros(
    #       dist2.distribution.event_shape, dtype=dist2.distribution.dtype)
    # input_hint2 = jax.device_put(input_hint2)
    # jaxpr_bij2 = jax.make_jaxpr(dist2.bijector.forward)(input_hint2)
    # print(jaxpr_bij2)

    η_kld = q_Η_given_x.kl_divergence(p_Η_given_z).sum()

    elbo = ll - β * z_kld - η_kld
    # TODO: add beta term for η_kld? Use same beta?

    return -elbo, {"elbo": elbo, "ll": ll, "z_kld": z_kld, "η_kld": η_kld}


def livae_loss_fn(
    model: nn.Module, params: nn.FrozenDict, x: Array, rng: PRNGKey, β: float = 1.0, α: float = 1.0
) -> Tuple[float, Mapping[str, float]]:
    """Single example loss function for LIVAE."""
    # TODO: this loss function is a 1 sample estimate, add an option for more samples?
    rng_local = random.fold_in(rng, lax.axis_index("batch"))
    q_Z_given_x, q_Η_given_x, p_X_given_xhat_η, _, p_Η_given_z, p_Z = model.apply(
        {"params": params},
        x,
        rng_local,
        α,
    )

    loss, metrics = calculate_livae_elbo(
        x, q_Z_given_x, q_Η_given_x, p_X_given_xhat_η, p_Η_given_z, p_Z, β
    )

    return loss, metrics
