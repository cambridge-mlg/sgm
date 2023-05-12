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

import tensorflow_probability.substrates.jax as tfp

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
    rng_apply, rng_dropout = random.split(rng_local)
    q_Z_given_x, p_X_given_z, p_Z = model.apply(
        {"params": params},
        x,
        rng_apply,
        train,
        rngs={'dropout': rng_dropout},
    )

    loss, metrics = calculate_vae_elbo(x, q_Z_given_x, p_X_given_z, p_Z, β)

    return loss, metrics


def create_vae_mll_estimator(
    model: nn.Module,
    params: nn.FrozenDict,
    train: bool = False,
    num_chains: int = 100,
    num_steps: int = 1000,
    step_size: float = 1e-1,
    num_leapfrog_steps: int = 2,
):
    """Create an estimator for the marginal log likelihood of X under a VAE model.
    This uses Hamiltonian Annealed Importance Sampling (HAIS) https://arxiv.org/abs/1205.1925.

    Args:
        model: The VAE model.
        params: The parameters of the VAE model.
        train: Whether to run the model in training mode.
        num_chains: The number of chains to run.
        num_steps: The number of steps to run each chain for.
        step_size: The step size to use for the HMC.
        num_leapfrog_steps: The number of leapfrog steps to use for the HMC.

    Returns:
        A function whose arguments are the data, a random number generator, and the params, and which returns
        an estimate of the marginal log likelihood of the data.
    """


    def estimate_mll(x: Array, rng: PRNGKey, params: nn.FrozenDict):
        logp_x_given_z = lambda z: model.apply(
            {"params": params}, x, z, train=train, method=model.logp_x_given_z
        )
        logp_z = lambda z: model.apply({"params": params}, z, method=model.logp_z)

        @jax.vmap
        def target_log_prob_fn(z):
            return logp_z(z) + logp_x_given_z(z)

        @jax.vmap
        def proposal_log_prob_fn(z):
            return logp_z(z)

        _, ais_weights, _ = tfp.mcmc.sample_annealed_importance_chain(
            num_steps=num_steps,
            proposal_log_prob_fn=proposal_log_prob_fn,
            target_log_prob_fn=target_log_prob_fn,
            current_state=jnp.zeros((num_chains, model.latent_dim)),
            make_kernel_fn=lambda tlp_fn: tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=tlp_fn,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps,
            ),
            seed=rng,
        )

        log_normalizer_estimate = jax.nn.logsumexp(ais_weights, axis=0) - jnp.log(num_chains)

        return log_normalizer_estimate

    return estimate_mll


# Step size | Num chains | Num steps | Num Leap Frogs | MLL        | Time
# 1e-1      | 100        | 1000      | 2              | 652.18     | 17.1s
# 1e-2      | 100        | 1000      | 2              | 477.62     |
# 2e-1      | 100        | 1000      | 2              | 633.18     |
# 5e-1      | 100        | 1000      | 2              | 572.47     |
# 1e-1      | 100        | 1000      | 8              | 657.97     | 53.4s
# 1e-1      | 100        | 1000      | 16             | 656.50     | 1m 42.5s
# 1e-1      | 300        | 1000      | 2              | 652.18     | 1m 1.3s
# 1e-1      | 100        | 3000      | 2              | 657.91     | 47.9s
# 1e-1      | 100        |  200      | 2              | 630.28     | 6.3s
# 1e-1      | 100        |  300      | 2              | 636.63     | 7.9s
# 1e-1      | 100        |  500      | 2              | 644.46     | 9.6s


def make_vae_batch_loss(model, agg=jnp.mean, train=True):
    def batch_loss(params, x_batch, mask, rng, state):
        # Broadcast loss over batch and aggregate.
        loss, metrics = jax.vmap(
            vae_loss_fn, in_axes=(None, None, 0, None, None, None), axis_name="batch"  # type: ignore
        )(model, params, x_batch, rng, state.β, train)
        loss, metrics, mask = jax.tree_util.tree_map(partial(agg, axis=0), (loss, metrics, mask))
        return loss, (metrics, mask)

    return batch_loss


def make_vae_reconstruction_plot(x, n_visualize, model, state, visualisation_rng, train=False):
    def reconstruct(x, sample_z=False, sample_xrecon=False):
        rng = random.fold_in(visualisation_rng, jax.lax.axis_index("image"))  # type: ignore
        rng_dropout, rng_apply = random.split(rng)
        return model.apply(
            {"params": state.params},
            x,
            rng_apply,
            sample_z=sample_z,
            sample_xrecon=sample_xrecon,
            train=train,
            method=model.reconstruct,
            rngs={'dropout': rng_dropout},
        )

    x_recon_modes = jax.vmap(
        reconstruct, axis_name="image", in_axes=(0, None, None)  # type: ignore
    )(x, False, False)

    recon_fig = plot_utils.plot_img_array(
        jnp.concatenate((x, x_recon_modes), axis=0),
        ncol=n_visualize,  # type: ignore
        pad_value=1,
        padding=1,
        title="Original | Reconstruction",
    )

    return recon_fig


def make_vae_sampling_plot(n_visualize, model, state, visualisation_rng, train=False):
    def sample(rng, sample_x=False):
        rng_dropout, rng_apply = random.split(rng)
        return model.apply(
            {"params": state.params},
            rng_apply,
            sample_x=sample_x,
            train=train,
            method=model.sample,
            rngs={'dropout': rng_dropout},
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
