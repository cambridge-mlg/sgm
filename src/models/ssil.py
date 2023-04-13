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
    Flow,
    INV_SOFTPLUS_1,
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
    σ_min: float = 1e-2

    def setup(self):
        self.bounds_array = jnp.array(self.bounds)
        # p(Η|Xhat)
        self.p_Η_given_Xhat = Flow(
            **(self.Η_given_Xhat or {}),
            bounds_array=self.bounds_array,
            event_shape=self.bounds_array.shape,
        )
        # TODO: try a simpler flow which only applies one layer of spline to each of the base distribution dims.
        # q(Η|X)
        self.q_Η_given_X = Flow(
            **(self.Η_given_X or {}),
            bounds_array=self.bounds_array,
            event_shape=self.bounds_array.shape,
        )
        self.σ = jax.nn.softplus(
            self.param(
                "σ_",
                init.constant(INV_SOFTPLUS_1),
                # ^ this value is softplus^{-1}(1), i.e., σ starts at 1.
                (),
            )
        ).clip(min=self.σ_min)

    def __call__(
        self, x: Array, rng: PRNGKey, α: float = 1.0, train=True
    ) -> Tuple[distrax.Distribution, ...]:
        η_rng, η_unif_rng, η_tot_rng = random.split(rng, 3)

        Η_uniform = distrax.Uniform(low=-α * self.bounds_array, high=α * self.bounds_array)
        η_uniform = Η_uniform.sample(seed=η_unif_rng)

        x_uniform = transform_image(x, η_uniform)

        q_Η_given_x_uniform = self.q_Η_given_X(x_uniform, train=train)
        η_tot = q_Η_given_x_uniform.sample(seed=η_tot_rng)
        # add a small gaussian noise to η_tot
        # η_tot = η_tot + 0.05 * random.normal(η_tot_rng, η_tot.shape)

        xhat = transform_image(x_uniform, -η_tot)
        # TODO: should this ^ be a sample from a reconstructor distribution? In the paper it is a distribution currently.

        p_Η_given_xhat = self.p_Η_given_Xhat(xhat, train=train)
        # p_Η = distrax.Normal(jnp.zeros((7,)), 0.5)
        # p_Η_given_xhat = distrax.MixtureOfTwo(0.9, p_Η_given_xhat_, p_Η)
        η_ = p_Η_given_xhat.sample(seed=η_rng)

        q_Η_given_x = self.q_Η_given_X(x, train=train)
        η = q_Η_given_x.sample(seed=η_rng)
        # add a small gaussian noise to η
        # η = η + 0.05 * random.normal(η_rng, η.shape)

        p_X_given_xhat_and_η = distrax.Independent(
            distrax.Normal(transform_image(xhat, η), self.σ), reinterpreted_batch_ndims=len(x.shape)
        )
        # TODO: this requires applying transform_image twice to the same input, but we could do it as a single call to two different inputs.

        return η, η_, p_X_given_xhat_and_η, p_Η_given_xhat, q_Η_given_x

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

        q_Η_given_x = self.q_Η_given_X(x)
        # TODO: should this be a randomly transformed x?
        if sample_η_proto:
            η1 = q_Η_given_x.sample(seed=η1_rng)
        else:
            η1 = approximate_mode(q_Η_given_x, 100, rng=η1_rng)

        xhat = transform_image(x, -η1)
        # TODO: should this ^ be a sample from a distribution?
        if prototype:
            return xhat

        p_Η_given_xhat = self.p_Η_given_Xhat(xhat)
        # TODO: should we use a randomly transformed x here (and in the ELBO)? I.e., single sample monte carlo estimate of the invariant function.
        # p_Η = distrax.Normal(jnp.zeros((7,)), 0.5)
        # p_Η_given_xhat = distrax.MixtureOfTwo(0.9, p_Η_given_xhat_, p_Η)
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


def calculate_ssil_elbo(
    x: Array,
    p_X_given_xhat_and_η: distrax.Distribution,
    p_Η_given_xhat: distrax.Distribution,
    q_Η_given_x: distrax.Distribution,
    rng: PRNGKey,
    bounds=Array,
    β: float = 1.0,
) -> Tuple[float, Mapping[str, float]]:
    ll = p_X_given_xhat_and_η.log_prob(x)

    def kld(dist1: distrax.Distribution, dist2: distrax.Distribution, n: int) -> float:
        xs, log_probs1 = dist1.sample_and_log_prob(seed=rng, sample_shape=(n,))
        xs = xs.clip(min=-0.999 * bounds, max=0.999 * bounds)
        log_probs2 = jax.vmap(dist2.log_prob)(xs)
        return jnp.mean(log_probs1, axis=0) - jnp.mean(log_probs2, axis=0)

    η_kld_ = kld(q_Η_given_x, p_Η_given_xhat, 100)

    def norm(x: Array) -> float:
        return jnp.sum((x/(bounds + 1e-3))**2) / len(x)

    # TODO: should this a) average over multiple samples, and b) divide by the bounds?

    η_q_sample = q_Η_given_x.sample(seed=rng, sample_shape=())
    # q_η_ce = 0.1 * p_Η.log_prob(η_q_sample).sum()
    q_η_ce = norm(η_q_sample)

    η_p_sample = p_Η_given_xhat.sample(seed=rng, sample_shape=())
    # p_η_ce = 0.1 * p_Η.log_prob(η_p_sample).sum()
    p_η_ce = norm(η_p_sample)

    η_kld = η_kld_ + q_η_ce + p_η_ce

    def entropy(dist: distrax.Distribution, n: int) -> float:
        _, log_probs = dist.sample_and_log_prob(seed=rng, sample_shape=(n,))
        return -jnp.mean(log_probs, axis=0)

    entropy_term = entropy(q_Η_given_x, 100)
    # TODO: implement this term properly

    elbo = ll - β * η_kld + entropy_term

    return -elbo, {
        "elbo": elbo,
        "ll": ll,
        "η_kld": η_kld,
        "η_kld_": η_kld_,
        "q_η_ce": q_η_ce,
        "p_η_ce": p_η_ce,
        "entropy_term": entropy_term,
    }


def ssil_loss_fn(
    model: nn.Module,
    params: nn.FrozenDict,
    x: Array,
    rng: PRNGKey,
    β: float = 1.0,
    α: float = 1.0,
    train: bool = True,
) -> Tuple[float, Mapping[str, float]]:
    """Single example loss function for Contrastive Invariance Learner."""
    # TODO: this loss function is a 1 sample estimate, add an option for more samples?
    rng_local = random.fold_in(rng, lax.axis_index("batch"))
    rng_model, rng_loss, rng_dropout = random.split(rng_local, 3)
    _, _, p_X_given_xhat_and_η, p_Η_given_xhat, q_Η_given_x = model.apply(
        {"params": params}, x, rng_model, α, train=train, rngs={"dropout": rng_dropout}
    )

    loss, metrics = calculate_ssil_elbo(
        x,
        p_X_given_xhat_and_η,
        p_Η_given_xhat,
        q_Η_given_x,
        rng_loss,
        jnp.array(model.bounds),
        β,
    )

    return loss, metrics


def make_ssil_batch_loss(model, agg=jnp.mean, train=True):
    # @jax.jit
    def batch_loss(params, x_batch, mask, rng, state):
        # Broadcast loss over batch and aggregate.
        loss, metrics = jax.vmap(
            ssil_loss_fn, in_axes=(None, None, 0, None, None, None, None), axis_name="batch"  # type: ignore
        )(model, params, x_batch, rng, state.β, state.α, train)
        loss, metrics, mask = jax.tree_util.tree_map(partial(agg, axis=0), (loss, metrics, mask))
        return loss, (metrics, mask)

    return batch_loss


def make_ssil_reconstruction_plot(x, n_visualize, model, state, visualisation_rng):
    # @partial(
    #     jax.jit,
    #     static_argnames=("prototype", "sample_η_proto", "sample_η_recon"),
    # )
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
