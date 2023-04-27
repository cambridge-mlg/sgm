"""Self-Supervised Invariance Learner VAE.

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
    Encoder,
    Decoder,
    DenseEncoder,
    Bijector,
    INV_SOFTPLUS_1,
    make_η_bounded,
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
        # p(Η|Xhat)
        self.p_Η_given_Xhat_base = DenseEncoder(
            latent_dim=7, **(self.Η_given_Xhat or {}).get("base", {})
        )
        self.p_Η_given_Xhat_bij = Bijector(**(self.Η_given_Xhat or {}).get("bijector", {}))
        # q(Η|X)
        self.q_Η_given_X_base = DenseEncoder(latent_dim=7, **(self.Η_given_X or {}).get("base", {}))
        self.q_Η_given_X_bij = Bijector(**(self.Η_given_X or {}).get("bijector", {}))
        self.σ = jax.nn.softplus(
            self.param(
                "σ_",
                init.constant(INV_SOFTPLUS_1),
                # ^ this value is softplus^{-1}(1), i.e., σ starts at 1.
                (),
            )
        )
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
        # q(Z|Xhat)
        self.q_Z_given_Xhat = Encoder(latent_dim=self.latent_dim, **(self.Z_given_Xhat or {}))
        # p(Xhat|Z)
        self.p_Xhat_given_Z = Decoder(image_shape=self.image_shape, **(self.Xhat_given_Z or {}))

    def __call__(self, x: Array, rng: PRNGKey) -> Tuple[distrax.Distribution, ...]:
        η_rng, η_unif_rng, η_tot_rng = random.split(rng, 3)

        Η_uniform = distrax.Uniform(low=-self.bounds_array, high=self.bounds_array)
        η_uniform = Η_uniform.sample(seed=η_unif_rng)

        x_uniform = transform_image(x, η_uniform)

        q_Η_given_x_uniform = distrax.Transformed(
            self.q_Η_given_X_base(x_uniform), self.q_Η_given_X_bij(x_uniform)
        )
        η_tot = q_Η_given_x_uniform.sample(seed=η_tot_rng)
        η_tot = make_η_bounded(η_tot, self.bounds_array)

        xhat = transform_image(x_uniform, -η_tot)
        # TODO: should this ^ be a sample from a distribution?

        p_Η_given_xhat = distrax.Transformed(
            self.p_Η_given_Xhat_base(xhat), self.p_Η_given_Xhat_bij(xhat)
        )

        x_ = jnp.sum(xhat) * 0 + x
        # TODO: this ^ is a hack to get jaxprs to match up. It isn't clear why this is necessary.
        # But without the hack the jaxpr for the p_Η_given_Z_bij is different from the jaxpr for the
        # q_Η_given_X_bij, where the difference comes from registers being on device0 vs not being
        # commited to a device. This is a problem because the take the KLD between the p_Η_given_Z
        # and the q_Η_given_x, the jaxprs must match up.

        q_Η_given_x = distrax.Transformed(self.q_Η_given_X_base(x_), self.q_Η_given_X_bij(x_))
        η = q_Η_given_x.sample(seed=η_rng)
        η = make_η_bounded(η, self.bounds_array)

        p_X_given_xhat_and_η = distrax.Normal(transform_image(xhat, η), self.σ)

        q_Z_given_xhat = self.q_Z_given_Xhat(xhat)
        z = q_Z_given_xhat.sample(seed=rng)

        p_Xhat_given_z = self.p_Xhat_given_Z(z)

        return (
            xhat,
            p_X_given_xhat_and_η,
            p_Η_given_xhat,
            q_Η_given_x,
            q_Z_given_xhat,
            p_Xhat_given_z,
            self.p_Z,
        )

    def sample(
        self,
        rng: PRNGKey,
        prototype: bool = False,
        sample_xhat: bool = False,
        sample_η: bool = False,
        sample_x: bool = False,
    ) -> Array:
        z_rng, xhat_rng, η_rng, x_rng = random.split(rng, 4)
        z = self.p_Z.sample(seed=z_rng)

        p_Xhat_given_z = self.p_Xhat_given_Z(z)
        xhat = p_Xhat_given_z.sample(seed=xhat_rng) if sample_xhat else p_Xhat_given_z.mean()
        if prototype:
            return xhat

        p_Η_given_xhat = distrax.Transformed(
            self.p_Η_given_Xhat_base(xhat), self.p_Η_given_Xhat_bij(xhat)
        )
        η = p_Η_given_xhat.sample(seed=η_rng) if sample_η else p_Η_given_xhat.mean()
        η = make_η_bounded(η, self.bounds_array)

        p_X_given_xhat_and_η = distrax.Normal(transform_image(xhat, η), self.σ)
        x = p_X_given_xhat_and_η.sample(seed=x_rng) if sample_x else p_X_given_xhat_and_η.mean()
        return x

    def reconstruct(
        self,
        x: Array,
        rng: PRNGKey,
        prototype: bool = False,
        sample_η: bool = False,
        sample_z: bool = False,
        sample_xhat: bool = False,
        sample_xrecon: bool = False,
    ) -> Array:
        η1_rng, z_rng, xhat_rng, η2_rng, xrecon_rng = random.split(rng, 5)

        q_Η_given_x = distrax.Transformed(self.q_Η_given_X_base(x), self.q_Η_given_X_bij(x))
        η1 = q_Η_given_x.sample(seed=η1_rng) if sample_η else q_Η_given_x.mean()
        η1 = make_η_bounded(η1, self.bounds_array)

        xhat = transform_image(x, -η1)

        q_Z_given_xhat = self.q_Z_given_Xhat(xhat)
        z = q_Z_given_xhat.sample(seed=z_rng) if sample_z else q_Z_given_xhat.mean()

        p_Xhat_given_z = self.p_Xhat_given_Z(z)
        xhat_recon = p_Xhat_given_z.sample(seed=xhat_rng) if sample_xhat else p_Xhat_given_z.mean()
        if prototype:
            return xhat_recon

        η2 = q_Η_given_x.sample(seed=η2_rng) if sample_η else η1
        η2 = make_η_bounded(η2, self.bounds_array)

        p_Xrecon_given_xhat_and_η = distrax.Normal(transform_image(xhat_recon, η2), self.σ)
        x_recon = (
            p_Xrecon_given_xhat_and_η.sample(seed=xrecon_rng)
            if sample_xrecon
            else p_Xrecon_given_xhat_and_η.mean()
        )
        return x_recon


def calculate_ssilvae_elbo(
    x: Array,
    xhat: Array,
    p_X_given_xhat_and_η: distrax.Distribution,
    p_Η_given_xhat: distrax.Distribution,
    q_Η_given_x: distrax.Distribution,
    q_Z_given_xhat: distrax.Distribution,
    p_Xhat_given_z: distrax.Distribution,
    p_Z: distrax.Distribution,
    β: float = 1.0,
) -> Tuple[float, Mapping[str, float]]:
    ll = p_X_given_xhat_and_η.log_prob(x).sum()
    η_kld = q_Η_given_x.kl_divergence(p_Η_given_xhat).sum()

    def entropy(dist: distrax.Distribution, n: int) -> float:
        xs = dist.sample(seed=random.PRNGKey(0), sample_shape=(n,))
        log_probs = jax.vmap(lambda x: dist.log_prob(x).sum())(xs)
        return -jnp.mean(log_probs)

    entropy_term = entropy(q_Η_given_x, 1000)

    ce_term = p_Xhat_given_z.log_prob(xhat).sum()
    # TODO: replace with multiple sample version
    z_kld = q_Z_given_xhat.kl_divergence(p_Z).sum()
    # TODO: add stop gradient tp z_kld so that learning the VAE doesn't mess with learning invariances?

    elbo = ll - η_kld + entropy_term + ce_term - β * z_kld

    return -elbo, {
        "elbo": elbo,
        "ll": ll,
        "η_kld": η_kld,
        "entropy_term": entropy_term,
        "ce_term": ce_term,
        "z_kld": z_kld,
    }


def ssilvae_loss_fn(
    model: nn.Module, params: nn.FrozenDict, x: Array, rng: PRNGKey, β: float = 1.0
) -> Tuple[float, Mapping[str, float]]:
    """Single example loss function for Contrastive Invariance Learner."""
    # TODO: this loss function is a 1 sample estimate, add an option for more samples?
    rng_local = random.fold_in(rng, lax.axis_index("batch"))
    (
        xhat,
        p_X_given_xhat_and_η,
        p_Η_given_xhat,
        q_Η_given_x,
        q_Z_given_xhat,
        p_Xhat_given_z,
        p_Z,
    ) = model.apply(
        {"params": params},
        x,
        rng_local,
    )

    loss, metrics = calculate_ssilvae_elbo(
        x,
        xhat,
        p_X_given_xhat_and_η,
        p_Η_given_xhat,
        q_Η_given_x,
        q_Z_given_xhat,
        p_Xhat_given_z,
        p_Z,
        β,
    )

    return loss, metrics


def make_ssilvae_batch_loss(model, agg=jnp.mean):
    @jax.jit
    def batch_loss(params, x_batch, mask, rng, state):
        # Broadcast loss over batch and aggregate.
        loss, metrics = jax.vmap(
            ssilvae_loss_fn, in_axes=(None, None, 0, None, None), axis_name="batch"  # type: ignore
        )(model, params, x_batch, rng, state.β)
        loss, metrics, mask = jax.tree_util.tree_map(partial(agg, axis=0), (loss, metrics, mask))
        return loss, (metrics, mask)

    return batch_loss


def make_ssilvae_reconstruction_plot(x, n_visualize, model, state, visualisation_rng):
    @partial(
        jax.jit,
        static_argnames=(
            "prototype",
            "sample_η",
            "sample_z",
            "sample_xhat",
            "sample_xrecon",
        ),
    )
    def reconstruct(x, prototype=False):
        rng = random.fold_in(visualisation_rng, jax.lax.axis_index("image"))  # type: ignore
        return model.apply(
            {"params": state.params},
            x,
            rng,
            prototype=prototype,
            sample_η=True,
            sample_z=False,
            sample_xhat=False,
            sample_xrecon=False,
            method=model.reconstruct,
        )

    x_proto = jax.vmap(reconstruct, axis_name="image", in_axes=(0, None))(x, True)  # type: ignore

    x_recon = jax.vmap(reconstruct, axis_name="image", in_axes=(0, None))(x, False)  # type: ignore

    recon_fig = plot_utils.plot_img_array(
        jnp.concatenate((x, x_proto, x_recon), axis=0),
        ncol=n_visualize,  # type: ignore
        pad_value=1,
        padding=1,
        title="Original | Prototype | Reconstruction",
    )

    return recon_fig


def make_ssilvae_sampling_plot(n_visualize, model, state, visualisation_rng):
    @partial(
        jax.jit,
        static_argnames=(
            "prototype",
            "sample_xhat",
            "sample_η",
            "sample_x",
        ),
    )
    def sample(rng, prototype=False):
        return model.apply(
            {"params": state.params},
            rng,
            prototype=prototype,
            sample_xhat=True,
            sample_η=True,
            sample_x=True,
            method=model.sample,
        )

    sampled_protos = jax.vmap(sample, in_axes=(0, None))(  # type: ignore
        jax.random.split(visualisation_rng, n_visualize), True  # type: ignore
    )

    sampled_data = jax.vmap(sample, in_axes=(0, None))(  # type: ignore
        jax.random.split(visualisation_rng, n_visualize), False  # type: ignore
    )

    sample_fig = plot_utils.plot_img_array(
        jnp.concatenate((sampled_protos, sampled_data), axis=0),
        ncol=n_visualize,  # type: ignore
        pad_value=1,
        padding=1,
        title="Sampled prototypes | Sampled data",
    )
    return sample_fig
