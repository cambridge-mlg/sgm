"""Self-Supervised Invariance Learner.

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
    DenseEncoder,
    Bijector,
    INV_SOFTPLUS_1,
    make_η_bounded,
    approximate_mode,
)
import src.utils.plotting as plot_utils

KwArgs = Mapping[str, Any]
PRNGKey = Any


class SSIL(nn.Module):
    image_shape: Tuple[int, int, int] = (28, 28, 1)
    Η_given_Xhat: Optional[KwArgs] = None
    Η_given_X: Optional[KwArgs] = None
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
        # self.p_Η_given_Xhat_bij = lambda _: distrax.Lambda(lambda x: x)
        # q(Η|X)
        self.q_Η_given_X_base = DenseEncoder(latent_dim=7, **(self.Η_given_X or {}).get("base", {}))
        self.q_Η_given_X_bij = Bijector(**(self.Η_given_X or {}).get("bijector", {}))
        # self.q_Η_given_X_bij = lambda _: distrax.Lambda(lambda x: x)
        self.σ = jax.nn.softplus(
            self.param(
                "σ_",
                init.constant(INV_SOFTPLUS_1),
                # ^ this value is softplus^{-1}(1), i.e., σ starts at 1.
                (),
            )
        )

    def __call__(self, x: Array, rng: PRNGKey, α: float = 1.0) -> Tuple[distrax.Distribution, ...]:
        η_rng, η_unif_rng, η_tot_rng = random.split(rng, 3)

        Η_uniform = distrax.Uniform(low=-α*self.bounds_array, high=α*self.bounds_array)
        η_uniform = Η_uniform.sample(seed=η_unif_rng)

        x_uniform = transform_image(x, η_uniform)

        q_Η_given_x_uniform = distrax.Transformed(
            self.q_Η_given_X_base(x_uniform), self.q_Η_given_X_bij(x_uniform)
        )
        η_tot = q_Η_given_x_uniform.sample(seed=η_tot_rng)
        η_tot = make_η_bounded(η_tot, α*self.bounds_array)

        xhat = transform_image(x_uniform, -η_tot)
        # TODO: should this ^ be a sample from a distribution? In the paper it is a distribution currently.

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
        η = make_η_bounded(η, α*self.bounds_array)

        p_X_given_xhat_and_η = distrax.Normal(transform_image(xhat, η), self.σ)

        return η, p_X_given_xhat_and_η, p_Η_given_xhat, q_Η_given_x

    def sample(
        self,
        rng: PRNGKey,
    ) -> Array:
        raise NotImplementedError

    def reconstruct(
        self,
        x: Array,
        rng: PRNGKey,
        prototype: bool = False,
        sample_η_proto: bool = False,
        sample_η_recon: bool = False,
        sample_xrecon: bool = False,
        α: float = 1.0,
    ) -> Array:
        η1_rng, η2_rng, xrecon_rng = random.split(rng, 3)

        q_Η_given_x = distrax.Transformed(self.q_Η_given_X_base(x), self.q_Η_given_X_bij(x))
        if sample_η_proto:
            η1 = q_Η_given_x.sample(seed=η1_rng)
        else:
            η1 = approximate_mode(q_Η_given_x, 100, rng=η1_rng)
        η1 = make_η_bounded(η1, α*self.bounds_array)

        xhat = transform_image(x, -η1)
        # TODO: should this ^ be a sample from a distribution?
        if prototype:
            return xhat

        p_Η_given_xhat = distrax.Transformed(
            self.p_Η_given_Xhat_base(xhat), self.p_Η_given_Xhat_bij(xhat)
        )
        if sample_η:
            η2 = p_Η_given_xhat.sample(seed=η2_rng)
        else:
            η2 = approximate_mode(p_Η_given_xhat, 100, rng=η2_rng)
        η2 = make_η_bounded(η2, α*self.bounds_array)

        p_Xrecon_given_xhat_and_η = distrax.Normal(transform_image(xhat, η2), self.σ)
        if sample_xrecon:
            xrecon = p_Xrecon_given_xhat_and_η.sample(seed=xrecon_rng)
        else:
            xrecon = p_Xrecon_given_xhat_and_η.mean()

        return xrecon


def calculate_ssil_elbo(
    x: Array,
    p_X_given_xhat_and_η: distrax.Distribution,
    p_Η_given_xhat: distrax.Distribution,
    q_Η_given_x: distrax.Distribution,
    rng: PRNGKey,
    β: float = 1.0,
) -> Tuple[float, Mapping[str, float]]:
    # dist1 = q_Η_given_x
    # input_hint1 = jnp.zeros(dist1.distribution.event_shape, dtype=dist1.distribution.dtype)
    # jaxpr_bij1 = jax.make_jaxpr(dist1.bijector.forward)(input_hint1)
    # print(jaxpr_bij1)
    # print("XXXX")
    # dist2 = p_Η_given_xhat
    # input_hint2 = jnp.zeros(dist2.distribution.event_shape, dtype=dist2.distribution.dtype)
    # input_hint2 = jax.device_put(input_hint2)
    # jaxpr_bij2 = jax.make_jaxpr(dist2.bijector.forward)(input_hint2)
    # print(jaxpr_bij2)

    ll = p_X_given_xhat_and_η.log_prob(x).sum()
    η_kld = q_Η_given_x.kl_divergence(p_Η_given_xhat).sum()

    # entropy_term = q_Η_given_x.entropy().sum()
    def entropy(dist: distrax.Distribution, n: int) -> float:
        xs = dist.sample(seed=rng, sample_shape=(n,))
        log_probs = jax.vmap(lambda x: dist.log_prob(x).sum())(xs)
        return -jnp.mean(log_probs)

    entropy_term = entropy(q_Η_given_x, 1000)

    elbo = ll - β * η_kld + entropy_term

    return -elbo, {"elbo": elbo, "ll": ll, "η_kld": η_kld, "entropy_term": entropy_term}


def ssil_loss_fn(
    model: nn.Module, params: nn.FrozenDict, x: Array, rng: PRNGKey, β: float = 1.0, α: float = 1.0,
) -> Tuple[float, Mapping[str, float]]:
    """Single example loss function for Contrastive Invariance Learner."""
    # TODO: this loss function is a 1 sample estimate, add an option for more samples?
    rng_local = random.fold_in(rng, lax.axis_index("batch"))
    rng_model, rng_loss = random.split(rng_local)
    _, p_X_given_xhat_and_η, p_Η_given_xhat, q_Η_given_x = model.apply(
        {"params": params},
        x,
        rng_model,
        α,
    )

    loss, metrics = calculate_ssil_elbo(
        x, p_X_given_xhat_and_η, p_Η_given_xhat, q_Η_given_x, rng_loss, β
    )

    return loss, metrics


def make_ssil_batch_loss(model, agg=jnp.mean):
    @jax.jit
    def batch_loss(params, x_batch, mask, rng, state):
        # Broadcast loss over batch and aggregate.
        loss, metrics = jax.vmap(
            ssil_loss_fn, in_axes=(None, None, 0, None, None, None), axis_name="batch"  # type: ignore
        )(model, params, x_batch, rng, state.β, state.α)
        loss, metrics, mask = jax.tree_util.tree_map(partial(agg, axis=0), (loss, metrics, mask))
        return loss, (metrics, mask)

    return batch_loss


def make_ssil_reconstruction_plot(x, n_visualize, model, state, visualisation_rng):
    @partial(
        jax.jit,
        static_argnames=("prototype", "sample_η_proto", "sample_η_recon"),
    )
    def reconstruct(x, prototype=False, sample_η_proto=False, sample_η_recon=False):
        rng = random.fold_in(visualisation_rng, jax.lax.axis_index("image"))  # type: ignore
        return model.apply(
            {"params": state.params},
            x,
            rng,
            prototype=prototype,
            sample_η_proto=sample_η_proto,
            sample_η_recon=sample_η_recon,
            sample_xrecon=False,
            method=model.reconstruct,
            α=state.α,
        )

    x_proto = jax.vmap(
        reconstruct, axis_name="image", in_axes=(0, None, None, None)  # type: ignore
    )(x, True, False, False)

    x_recon = jax.vmap(
        reconstruct, axis_name="image", in_axes=(0, None, None, None)  # type: ignore
    )(x, False, False, False)

    # x_recon_sample = jax.vmap(
    #     reconstruct, axis_name="image", in_axes=(0, None, None, None)  # type: ignore
    # )(x, False, False, True)

    recon_fig = plot_utils.plot_img_array(
        jnp.concatenate((x, x_proto, x_recon), axis=0),
        ncol=n_visualize,  # type: ignore
        pad_value=1,
        padding=1,
        title="Original | Prototype | Reconstruction",
    )

    return recon_fig


def make_ssil_sampling_plot(n_visualize, model, state, visualisation_rng):
    return None
