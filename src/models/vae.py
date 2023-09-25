"""VAE implementation.

A note on notation. In order to distinguish between random variables and their values, we use upper
and lower case variable names. I.e., p(Z) or `p_Z` is the distribution over the r.v. Z, and is a
function, while p(z) or `p_z` is the probability that Z=z. Similarly, p(X|Z) or `p_X_given_Z` is a
a function which returns another function p(X|z) or `p_X_given_z`, which would return the proability
that X=x|Z=z a.k.k `p_x_given_z`.
"""

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
from functools import partial

import numpy as np
import jax
from jax import numpy as jnp
from jax import random, lax
from chex import Array, PRNGKey
from flax import linen as nn
import flax.linen.initializers as init
import distrax

import tensorflow_probability.substrates.jax as tfp

import src.utils.plotting as plot_utils
from src.utils.types import KwArgs


INV_SOFTPLUS_1 = jnp.log(jnp.exp(1) - 1.0)


def _get_num_even_divisions(x):
    """Returns the number of times x can be divided by 2."""
    i = 0
    while x % 2 == 0:
        x = x // 2
        i += 1
    return i


class Encoder(nn.Module):
    """p(z|x) = N(μ(x), σ(x)), where μ(x) and σ(x) are neural networks."""

    latent_dim: int
    conv_dims: Optional[Sequence[int]] = None
    dense_dims: Optional[Sequence[int]] = None
    act_fn: Callable = nn.relu
    norm_cls: nn.Module = nn.LayerNorm
    σ_min: float = 1e-2
    dropout_rate: float = 0.0
    input_dropout_rate: float = 0.0
    max_2strides: Optional[int] = None
    train: Optional[bool] = None

    @nn.compact
    def __call__(self, x: Array, train: Optional[bool] = None) -> distrax.Distribution:
        train = nn.merge_param("train", self.train, train)

        conv_dims = self.conv_dims if self.conv_dims is not None else [64, 128, 256]
        dense_dims = self.dense_dims if self.dense_dims is not None else [64, 32]

        x = nn.Dropout(rate=self.input_dropout_rate, deterministic=not train)(x)

        h = x
        i = -1
        if len(x.shape) > 1:
            assert x.shape[0] == x.shape[1], "Images should be square."
            num_2strides = np.minimum(
                _get_num_even_divisions(x.shape[0]), len(conv_dims)
            )
            if self.max_2strides is not None:
                num_2strides = np.minimum(num_2strides, self.max_2strides)

            for i, conv_dim in enumerate(conv_dims):
                h = nn.Conv(
                    conv_dim,
                    kernel_size=(3, 3),
                    strides=(2, 2) if i < num_2strides else (1, 1),
                    name=f"conv_{i}",
                )(h)
                h = self.norm_cls(name=f"norm_{i}")(h)
                h = self.act_fn(h)

            h = nn.Conv(3, kernel_size=(3, 3), strides=(1, 1), name=f"resize")(h)

            h = h.flatten()

        for j, dense_dim in enumerate(dense_dims):
            h = nn.Dense(dense_dim, name=f"dense_{j+i+1}")(h)
            h = self.norm_cls(name=f"norm_{j+i+1}")(h)
            h = self.act_fn(h)
            h = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(h)

        # We initialize these dense layers so that we get μ=0 and σ=1 at the start.
        μ = nn.Dense(
            self.latent_dim, kernel_init=init.zeros, bias_init=init.zeros, name="μ"
        )(h)
        σ = jax.nn.softplus(
            nn.Dense(
                self.latent_dim,
                kernel_init=init.zeros,
                bias_init=init.constant(INV_SOFTPLUS_1),
                name="σ_",
            )(h)
        )

        return distrax.Independent(
            distrax.Normal(loc=μ, scale=σ.clip(min=self.σ_min)), 1
        )


class Decoder(nn.Module):
    """p(x|z) = N(μ(z), σ), where μ(z) is a neural network."""

    image_shape: Tuple[int, int, int]
    conv_dims: Optional[Sequence[int]] = None
    dense_dims: Optional[Sequence[int]] = None
    σ_init: Callable = init.constant(INV_SOFTPLUS_1)
    act_fn: Callable = nn.relu
    norm_cls: nn.Module = nn.LayerNorm
    σ_min: float = 1e-2
    dropout_rate: float = 0.0
    input_dropout_rate: float = 0.0
    max_2strides: Optional[int] = None
    train: Optional[bool] = None

    @nn.compact
    def __call__(self, z: Array, train: Optional[bool] = None) -> distrax.Distribution:
        train = nn.merge_param("train", self.train, train)

        conv_dims = self.conv_dims if self.conv_dims is not None else [256, 128, 64]
        dense_dims = self.dense_dims if self.dense_dims is not None else [32, 64]

        if len(self.image_shape) == 3:
            assert (
                self.image_shape[0] == self.image_shape[1]
            ), "Images should be square."
        output_size = self.image_shape[0]
        num_2strides = np.minimum(_get_num_even_divisions(output_size), len(conv_dims))
        if self.max_2strides is not None:
            num_2strides = np.minimum(num_2strides, self.max_2strides)

        z = nn.Dropout(rate=self.input_dropout_rate, deterministic=not train)(z)

        j = -1
        for j, dense_dim in enumerate(dense_dims):
            z = nn.Dense(dense_dim, name=f"dense_{j}")(z)
            z = self.norm_cls(name=f"norm_{j}")(z)
            z = self.act_fn(z)
            z = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(z)

        dense_size = output_size // (2**num_2strides)
        h = nn.Dense(dense_size * dense_size * 3, name=f"resize")(z)
        h = h.reshape(dense_size, dense_size, 3)

        for i, conv_dim in enumerate(conv_dims):
            h = nn.ConvTranspose(
                conv_dim,
                kernel_size=(3, 3),
                # use stride of 2 for the last few layers
                strides=(2, 2) if i >= len(conv_dims) - num_2strides else (1, 1),
                name=f"conv_{i+j+1}",
            )(h)
            h = self.norm_cls(name=f"norm_{i+j+1}")(h)
            h = self.act_fn(h)

        μ = nn.Conv(
            self.image_shape[-1], kernel_size=(3, 3), strides=(1, 1), name=f"μ"
        )(h)
        σ = jax.nn.softplus(self.param("σ_", self.σ_init, self.image_shape))

        return distrax.Independent(
            distrax.Normal(loc=μ, scale=σ.clip(min=self.σ_min)), len(self.image_shape)
        )


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
        self.p_X_given_Z = Decoder(
            image_shape=self.image_shape, **(self.X_given_Z or {})
        )

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
        return_z: bool = False,
    ) -> Array:
        z_rng, x_rng = random.split(rng, 2)
        z = self.p_Z.sample(seed=z_rng)
        if return_z:
            return z

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

    def importance_weighted_lower_bound(
        self,
        x: Array,
        rng: PRNGKey,
        num_samples: int = 50,
        train: bool = False,
    ):
        def single_sample_w(i):
            rng_i = random.fold_in(rng, i)
            q_Z_given_x = self.q_Z_given_X(x, train=train)
            z = q_Z_given_x.sample(seed=rng_i, sample_shape=())
            logq_z_given_x = q_Z_given_x.log_prob(z)

            p_X_given_z = self.p_X_given_Z(z, train=train)
            logp_x_given_z = p_X_given_z.log_prob(x)

            logp_z = self.p_Z.log_prob(z)

            return logp_x_given_z + logp_z - logq_z_given_x

        log_ws = jax.vmap(single_sample_w)(jnp.arange(num_samples))

        return jax.nn.logsumexp(log_ws, axis=0) - jnp.log(num_samples)


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
    **kwargs,
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
        rngs={"dropout": rng_dropout},
    )

    loss, metrics = calculate_vae_elbo(x, q_Z_given_x, p_X_given_z, p_Z, β)

    return loss, metrics


def create_vae_hais_mll_estimator(
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

    def estimate_mll(x: Array, rng: PRNGKey):
        logp_x_given_z = lambda z: model.apply(
            {"params": params}, x, z, train=train, method=model.logp_x_given_z
        )
        logp_z = lambda z: model.apply({"params": params}, z, method=model.logp_z)
        logq_z_given_x = lambda z: model.apply(
            {"params": params}, x, z, train=train, method=model.logq_z_given_x
        )

        @jax.vmap
        def sample_z(rng):
            return model.apply(
                {"params": params}, rng, train=train, return_z=True, method=model.sample
            )

        zs = sample_z(random.split(rng, num_chains))

        @jax.vmap
        def target_log_prob_fn(z):
            return logp_z(z) + logp_x_given_z(z)

        @jax.vmap
        def proposal_log_prob_fn(z):
            # return logp_z(z)
            return logq_z_given_x(z)

        _, ais_weights, _ = tfp.mcmc.sample_annealed_importance_chain(
            num_steps=num_steps,
            proposal_log_prob_fn=proposal_log_prob_fn,
            target_log_prob_fn=target_log_prob_fn,
            current_state=zs,
            make_kernel_fn=lambda tlp_fn: tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=tlp_fn,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps,
            ),
            seed=rng,
        )

        log_normalizer_estimate = jax.nn.logsumexp(ais_weights, axis=0) - jnp.log(
            num_chains
        )

        return log_normalizer_estimate

    return estimate_mll


def make_vae_batch_loss(
    model,
    agg: Callable = jnp.mean,
    train: bool = True,
    run_iwlb: bool = False,
    iwlb_kwargs: Optional[Mapping[str, Any]] = None,
):
    def batch_loss(params, x_batch, mask, rng, state):
        # Broadcast loss over batch...
        loss, metrics = jax.vmap(
            vae_loss_fn, in_axes=(None, None, 0, None, None, None), axis_name="batch"  # type: ignore
        )(model, params, x_batch, rng, state.β, train)

        if run_iwlb:
            iwlb_kwargs_ = {} if iwlb_kwargs is None else iwlb_kwargs

            @jax.vmap
            def _iwlb(x):
                return model.apply(
                    {"params": params},
                    x,
                    rng,
                    **iwlb_kwargs_,
                    method=model.importance_weighted_lower_bound,
                )

            metrics["iwlb"] = _iwlb(x_batch)

        # ... and aggregate.
        loss, metrics, mask = jax.tree_util.tree_map(
            partial(agg, axis=0), (loss, metrics, mask)
        )
        return loss, (metrics, mask)

    return batch_loss


def make_vae_reconstruction_plot(
    x, n_visualize, model, state, visualisation_rng, train=False
):
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
            rngs={"dropout": rng_dropout},
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
            rngs={"dropout": rng_dropout},
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
