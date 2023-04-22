"""Self-Supervised Invariance Learner.

A note on notation. In order to distinguish between random variables and their values, we use upper
and lower case variable names. I.e., p(Z) or `p_Z` is the distribution over the r.v. Z, and is a
function, while p(z) or `p_z` is the probability that Z=z. Similarly, p(X|Z) or `p_X_given_Z` is a
a function which returns another function p(X|z) or `p_X_given_z`, which would return the proability
that X=x|Z=z a.k.a `p_x_given_z`.
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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

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
        self, x: Array, rng: PRNGKey, α: float = 1.0, train: bool = True
    ) -> Tuple[distrax.Distribution, ...]:
        η_rng, η_unif_rng, η_tot_rng = random.split(rng, 3)

        Η_uniform = distrax.Uniform(low=-α * self.bounds_array, high=α * self.bounds_array)
        η_uniform = Η_uniform.sample(seed=η_unif_rng)

        x_uniform = transform_image(x, η_uniform)

        q_Η_given_x_uniform = self.q_Η_given_X(x_uniform, train=train)
        η_tot = q_Η_given_x_uniform.sample(seed=η_tot_rng)
        # add a small noise to η_tot
        # η_tot = η_tot + 0.03 * α * self.bounds_array * random.uniform(
        #     η_tot_rng, η_tot.shape, minval=-1, maxval=1
        # )

        xhat = transform_image(x_uniform, -η_tot)
        # TODO: should this ^ be a sample from a reconstructor distribution? In the paper it is a distribution currently.

        p_Η_given_xhat = self.p_Η_given_Xhat(xhat, train=train)
        # p_Η = distrax.Normal(jnp.zeros((7,)), 0.5)
        # p_Η_given_xhat = distrax.MixtureOfTwo(0.9, p_Η_given_xhat_, p_Η)
        η_ = p_Η_given_xhat.sample(seed=η_rng)

        q_Η_given_x = self.q_Η_given_X(x, train=train)
        η = q_Η_given_x.sample(seed=η_rng)
        # add a small noise to η
        # η = η + 0.03 * α * self.bounds_array * random.uniform(η_rng, η.shape, minval=-1, maxval=1)

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
        train: bool = True,
    ) -> Array:
        η1_rng, η2_rng, xrecon_rng = random.split(rng, 3)

        q_Η_given_x = self.q_Η_given_X(x, train=train)
        # TODO: should this be a randomly transformed x?
        if sample_η_proto:
            η1 = q_Η_given_x.sample(seed=η1_rng)
        else:
            η1 = approximate_mode(q_Η_given_x, 100, rng=η1_rng)

        xhat = transform_image(x, -η1)
        # TODO: should this ^ be a sample from a distribution?
        if prototype:
            return xhat

        p_Η_given_xhat = self.p_Η_given_Xhat(xhat, train=train)
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
    bounds: Array,
    β: float = 1.0,
    γ: float = 1.0,
    n: int = 100,
    # TODO: more? less?
    ε: float = 1e-6,
) -> Tuple[float, Mapping[str, float]]:
    ll = p_X_given_xhat_and_η.log_prob(x)

    η_qs, q_Η_log_probs = q_Η_given_x.sample_and_log_prob(seed=rng, sample_shape=(n,))
    q_H_entropy = -jnp.mean(q_Η_log_probs, axis=0)

    p_Η_log_probs = jax.vmap(p_Η_given_xhat.log_prob)(
        η_qs.clip(min=-(1 - ε) * bounds, max=(1 - ε) * bounds)
    )
    p_q_H_cross_entropy = -jnp.mean(p_Η_log_probs, axis=0)

    η_kld_ = p_q_H_cross_entropy - q_H_entropy

    def norm(η: Array) -> float:
        return jnp.sum(η**2) / len(η)

    q_η_norm = jax.vmap(norm)(η_qs).mean()

    η_ps = p_Η_given_xhat.sample(seed=rng, sample_shape=(n,))
    p_η_norm = jax.vmap(norm)(η_ps).mean()

    η_kld = η_kld_ + γ * (q_η_norm + p_η_norm)

    elbo = ll - β * η_kld + q_H_entropy
    # TODO: this entropy term should be using a distribtuion conditioned on a randomly transformed x.

    return -elbo, {
        "elbo": elbo,
        "ll": ll,
        "η_kld": η_kld,
        "η_kld_": η_kld_,
        "q_η_ce": q_η_norm,
        "p_η_ce": p_η_norm,
        "entropy_term": q_H_entropy,
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
        γ,
    )

    return loss, metrics


def make_ssil_batch_loss(model, agg=jnp.mean, train=True):
    # @jax.jit
    def batch_loss(params, x_batch, mask, rng, state):
        # Broadcast loss over batch and aggregate.
        loss, metrics = jax.vmap(
            ssil_loss_fn, in_axes=(None, None, 0, None, None, None, None, None), axis_name="batch"  # type: ignore
        )(model, params, x_batch, rng, state.α, state.β, state.γ, train)
        loss, metrics, mask = jax.tree_util.tree_map(partial(agg, axis=0), (loss, metrics, mask))
        return loss, (metrics, mask)

    return batch_loss


def make_ssil_reconstruction_plot(x, n_visualize, model, state, visualisation_rng, train=False):
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
            train=train,
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


def make_summary_plot(config, final_state, x, rng):
    fig, axs = plt.subplots(3, 6, figsize=(10, 5), dpi=200)

    axs[0, 0].imshow(x, cmap="gray")
    axs[0, 0].set_title("x")
    axs[0, 0].axis("off")

    axs[1, 0].axis("off")

    η_rng, η_p_rng, η_pp_rng, η_recon_p_rng, η_recon_rng = random.split(rng, 5)
    # η_rng = η_p_rng = η_pp_rng = η_recon_p_rng = η_recon_rng = rng

    q_Η_given_X = Flow(
        **(config.model.Η_given_X or {}),
        event_shape=(7,),
        bounds_array=jnp.array(config.model.bounds),
        train=False,
    )
    p_Η_given_Xhat = Flow(
        **(config.model.Η_given_Xhat or {}),
        event_shape=(7,),
        bounds_array=jnp.array(config.model.bounds),
        train=False,
    )

    # x hat
    q_Η_given_x = q_Η_given_X.apply({"params": final_state.params["q_Η_given_X"]}, x)
    η = approximate_mode(q_Η_given_x, 100, η_rng)
    axs[1, 1].bar(range(7), η / jnp.array(config.model.bounds), label="η", color="C0", alpha=0.7)
    axs[1, 1].set_ylim(-1, 1)
    axs[1, 1].set_title("η | x")

    xhat = transform_image(x, -η)
    axs[0, 1].imshow(xhat, cmap="gray")
    axs[0, 1].set_title("xhat")
    axs[0, 1].axis("off")

    # x recon
    p_Η_given_xhat = p_Η_given_Xhat.apply({"params": final_state.params["p_Η_given_Xhat"]}, xhat)
    η_recon = approximate_mode(p_Η_given_xhat, 100, η_recon_rng)
    axs[1, 2].bar(
        range(7), η_recon / jnp.array(config.model.bounds), label="η_recon", color="C1", alpha=0.7
    )
    axs[1, 2].sharey(axs[1, 1])
    axs[1, 2].set_title("η_recon | xhat")

    x_recon = transform_image(xhat, η_recon)
    axs[0, 2].imshow(x_recon, cmap="gray")
    axs[0, 2].set_title("x_recon")
    axs[0, 2].axis("off")

    # x'
    Η_uniform = distrax.Uniform(
        low=-jnp.array(config.model.bounds), high=jnp.array(config.model.bounds)
    )
    η_p = Η_uniform.sample(seed=η_p_rng)
    axs[1, 3].bar(
        range(7), η_p / jnp.array(config.model.bounds), label="η_p", color="C2", alpha=0.7
    )
    axs[1, 3].sharey(axs[1, 1])
    axs[1, 3].set_title("η_rng")

    x_p = transform_image(x, η_p)
    axs[0, 3].imshow(x_p, cmap="gray")
    axs[0, 3].set_title("x'")
    axs[0, 3].axis("off")

    # x' hat
    q_Η_given_x_p = q_Η_given_X.apply({"params": final_state.params["q_Η_given_X"]}, x_p)
    η_pp = approximate_mode(q_Η_given_x_p, 100, rng=η_pp_rng)
    axs[1, 4].bar(
        range(7), η_pp / jnp.array(config.model.bounds), label="η_pp", color="C3", alpha=0.7
    )
    axs[1, 4].sharey(axs[1, 1])
    axs[1, 4].set_title("η' | x'")

    xhat_p = transform_image(x_p, -η_pp)
    axs[0, 4].imshow(xhat_p, cmap="gray")
    axs[0, 4].set_title("xhat'")
    axs[0, 4].axis("off")

    # x' recon
    p_Η_given_xhat_p = p_Η_given_Xhat.apply(
        {"params": final_state.params["p_Η_given_Xhat"]}, xhat_p
    )
    η_recon_p = approximate_mode(p_Η_given_xhat_p, 100, η_recon_p_rng)
    axs[1, 5].bar(
        range(7),
        η_recon_p / jnp.array(config.model.bounds),
        label="η_recon'",
        color="C4",
        alpha=0.7,
    )
    axs[1, 5].sharey(axs[1, 1])
    axs[1, 5].set_title("η_recon' | xhat'")

    x_recon_p = transform_image(xhat_p, η_recon_p)
    axs[0, 5].imshow(x_recon_p, cmap="gray")
    axs[0, 5].set_title("x_recon'")
    axs[0, 5].axis("off")

    # make distribution histograms
    def _make_dist_plots(dist, rng, axs, color="C0", ls="-"):
        samples, _ = dist.sample_and_log_prob(seed=rng, sample_shape=(3000,))
        samples = samples / jnp.array(config.model.bounds)

        for i, ax in enumerate(axs):
            ax.hist(samples[:, i], bins=50, density=True, color=color, alpha=0.25)
            # plot a gaussian KDE over the histogram
            x = jnp.linspace(-1, 1, 1000)
            kde = gaussian_kde(samples[:, i])
            axs[i].plot(x, kde(x), ls, color=color, alpha=0.5, lw=1)

    _make_dist_plots(q_Η_given_x, η_rng, axs[2, :], color="C0", ls="-")
    _make_dist_plots(p_Η_given_xhat, η_recon_rng, axs[2, :], color="C1", ls="-")
    _make_dist_plots(q_Η_given_x_p, η_pp_rng, axs[2, :], color="C3", ls="--")
    _make_dist_plots(p_Η_given_xhat_p, η_recon_p_rng, axs[2, :], color="C4", ls="--")

    axs[2, 0].set_yscale("symlog")
    axs[2, 0].set_ylabel("log density")
    for ax in axs[2, 0:]:
        ax.sharey(axs[2, 0])
        ax.set_yscale("symlog")
        ax.set_xlim(-1, 1)

    for i, ax in enumerate(axs[2, :]):
        _titles = ["trans x", "trans y", "rot", "scale x", "scale y", "shear x"]
        ax.set_title(_titles[i])

    # add legend to axs[1, 0], by building Line2D objects
    axs[1, 0].legend(
        handles=[
            Line2D([0], [0], color="C0", lw=4, label="η"),
            Line2D([0], [0], color="C1", lw=4, label="η_recon"),
            Line2D([0], [0], color="C2", lw=4, label="η_rng"),
            Line2D([0], [0], color="C3", lw=4, label="η'"),
            Line2D([0], [0], color="C4", lw=4, label="η_recon'"),
        ]
    )

    for i, axs_ in enumerate(axs):
        if i == 0:
            continue
        for ax in axs_:
            if i != 2:
                ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    fig.tight_layout()
    fig.show()

    return fig
