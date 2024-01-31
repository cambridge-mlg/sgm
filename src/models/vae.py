"""VAE implementation.

A note on notation. In order to distinguish between random variables and their values, we use upper
and lower case variable names. I.e., p(Z) or `p_Z` is the distribution over the r.v. Z, and is a
function, while p(z) or `p_z` is the probability that Z=z. Similarly, p(X|Z) or `p_X_given_Z` is a
a function which returns another function p(X|z) or `p_X_given_z`, which would return the proability
that X=x|Z=z a.k.a `p_x_given_z`.
"""

from functools import partial
from typing import Callable, Optional, Sequence, Tuple

import ciclo
import distrax
import flax
import flax.linen.initializers as init
import jax
import numpy as np
import optax
from chex import Array, PRNGKey
from clu import metrics, parameter_overview
from flax import linen as nn
from flax.training import train_state
from jax import lax
from jax import numpy as jnp
from jax import random

import src.utils.plotting as plot_utils
from src.models.mlp import MLP
from src.models.utils import clipped_adamw
from src.utils.training import get_learning_rate
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
    conv_dims: Sequence[int]
    dense_dims: Sequence[int]
    act_fn: Callable = nn.gelu
    norm_cls: nn.Module = nn.LayerNorm
    σ_min: float = 1e-2
    dropout_rate: float = 0.0
    max_2strides: Optional[int] = None
    train: Optional[bool] = None

    @nn.compact
    def __call__(self, x: Array, train: Optional[bool] = None) -> distrax.Distribution:
        train = nn.merge_param("train", self.train, train)

        h = x
        if len(x.shape) > 1:
            assert x.shape[0] == x.shape[1], "Images should be square."
            num_2strides = np.minimum(
                _get_num_even_divisions(x.shape[0]), len(self.conv_dims)
            )
            if self.max_2strides is not None:
                num_2strides = np.minimum(num_2strides, self.max_2strides)

            for i, conv_dim in enumerate(self.conv_dims):
                h = nn.Conv(
                    conv_dim,
                    kernel_size=(3, 3),
                    strides=(2, 2) if i < num_2strides else (1, 1),
                )(h)
                h = self.act_fn(h)
                h = self.norm_cls()(h)

            h = nn.Conv(3, kernel_size=(3, 3), strides=(1, 1), name=f"resize")(h)

            h = h.flatten()

        h = MLP(
            self.dense_dims,
            act_fn=self.act_fn,
            norm_cls=self.norm_cls,
            dropout_rate=self.dropout_rate,
        )(h, train)

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
    conv_dims: Sequence[int]
    dense_dims: Sequence[int]
    σ_init: Callable = init.constant(INV_SOFTPLUS_1)
    act_fn: Callable = nn.gelu
    norm_cls: nn.Module = nn.LayerNorm
    σ_min: float = 1e-2
    dropout_rate: float = 0.0
    max_2strides: Optional[int] = None
    train: Optional[bool] = None

    @nn.compact
    def __call__(self, z: Array, train: Optional[bool] = None) -> distrax.Distribution:
        train = nn.merge_param("train", self.train, train)

        if len(self.image_shape) == 3:
            assert (
                self.image_shape[0] == self.image_shape[1]
            ), "Images should be square."
        output_size = self.image_shape[0]
        num_2strides = np.minimum(
            _get_num_even_divisions(output_size), len(self.conv_dims)
        )
        if self.max_2strides is not None:
            num_2strides = np.minimum(num_2strides, self.max_2strides)

        z = MLP(
            self.dense_dims,
            act_fn=self.act_fn,
            norm_cls=self.norm_cls,
            dropout_rate=self.dropout_rate,
        )(z, train)

        dense_size = output_size // (2**num_2strides)
        h = nn.Dense(dense_size * dense_size * 3, name=f"resize")(z)
        h = h.reshape(dense_size, dense_size, 3)

        for i, conv_dim in enumerate(self.conv_dims):
            h = nn.ConvTranspose(
                conv_dim,
                kernel_size=(3, 3),
                # use stride of 2 for the last few layers
                strides=(2, 2) if i >= len(self.conv_dims) - num_2strides else (1, 1),
            )(h)
            h = self.act_fn(h)
            h = self.norm_cls()(h)

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
    conv_dims: Optional[Sequence[int]] = None
    dense_dims: Optional[Sequence[int]] = None
    Z_given_X: Optional[KwArgs] = None
    X_given_Z: Optional[KwArgs] = None
    σ_min: float = 1e-2

    def setup(self):
        conv_dims = self.conv_dims if self.conv_dims is not None else [256, 128, 64]
        dense_dims = self.dense_dims if self.dense_dims is not None else [32, 64]

        # p(Z)
        self.p_Z = distrax.Independent(
            distrax.Normal(
                loc=self.param("prior_μ", init.zeros, (self.latent_dim,)),
                scale=jax.nn.softplus(
                    self.param(
                        "prior_σ_",
                        init.constant(INV_SOFTPLUS_1),
                        (self.latent_dim,),
                    )
                ).clip(min=self.σ_min),
            ),
            reinterpreted_batch_ndims=1,
        )
        # q(Z|X)
        self.q_Z_given_X = Encoder(
            latent_dim=self.latent_dim,
            conv_dims=conv_dims,
            dense_dims=dense_dims,
            **(self.Z_given_X or {}),
        )
        # p(X|Z)
        self.p_X_given_Z = Decoder(
            image_shape=self.image_shape,
            conv_dims=tuple(reversed(conv_dims)),
            dense_dims=tuple(reversed(dense_dims)),
            **(self.X_given_Z or {}),
        )

    def __call__(
        self, x: Array, train: bool = True
    ) -> Tuple[distrax.Distribution, ...]:
        q_Z_given_x = self.q_Z_given_X(x, train=train)
        z = q_Z_given_x.sample(seed=self.make_rng("sample"))

        p_X_given_z = self.p_X_given_Z(z, train=train)

        return q_Z_given_x, p_X_given_z, self.p_Z

    # def inference(self, x: Array, train: bool = True) -> distrax.Distribution:
    #     return self.q_Z_given_X(x, train=train)

    # def generative(self, z: Array, train: bool = True) -> distrax.Distribution:
    #     return self.p_X_given_Z(z, train=train)

    # def prior(self) -> distrax.Distribution:
    #     return self.p_Z

    def sample(
        self,
        sample_x: bool = False,
        train: bool = True,
        return_z: bool = False,
    ) -> Array:
        z = self.p_Z.sample(seed=self.make_rng("sample"))
        if return_z:
            return z

        p_X_given_z = self.p_X_given_Z(z, train=train)
        if sample_x:
            x = p_X_given_z.sample(seed=self.make_rng("sample"))
        else:
            x = p_X_given_z.mode()

        return x

    def reconstruct(
        self,
        x: Array,
        sample_z: bool = False,
        sample_xrecon: bool = False,
        train: bool = True,
    ) -> Array:
        q_Z_given_x = self.q_Z_given_X(x, train=train)
        if sample_z:
            z = q_Z_given_x.sample(seed=self.make_rng("sample"))
        else:
            z = q_Z_given_x.mode()

        p_X_given_z = self.p_X_given_Z(z, train=train)
        if sample_xrecon:
            x_recon = p_X_given_z.sample(seed=self.make_rng("sample"))
        else:
            x_recon = p_X_given_z.mode()

        return x_recon

    def elbo(
        self,
        x: Array,
        train: bool = False,
        β: float = 1.0,
    ) -> float:
        q_Z_given_x, p_X_given_z, p_Z = self(x, train=train)

        ll = p_X_given_z.log_prob(x) / x.shape[-1]
        kld = q_Z_given_x.kl_divergence(p_Z)

        return ll - β * kld, ll, kld

    def importance_weighted_lower_bound(
        self,
        x: Array,
        num_samples: int = 50,
        train: bool = False,
    ) -> float:
        def single_sample_log_w(i):
            q_Z_given_x = self.q_Z_given_X(x, train=train)
            z = q_Z_given_x.sample(
                seed=random.fold_in(self.make_rng("sample"), i), sample_shape=()
            )
            # NOTE: although make_rng returns a different rng each time it is called,
            # because of how vmap interacts (I think) with stateful objects, make_rng
            # will only be called once, thus we fold in the sample index.
            logq_z_given_x = q_Z_given_x.log_prob(z)

            p_X_given_z = self.p_X_given_Z(z, train=train)
            logp_x_given_z = p_X_given_z.log_prob(x)

            logp_z = self.p_Z.log_prob(z)

            return logp_x_given_z + logp_z - logq_z_given_x

        log_ws = jax.vmap(single_sample_log_w)(jnp.arange(num_samples))

        return jax.nn.logsumexp(log_ws, axis=0) - jnp.log(num_samples)


def make_vae_train_and_eval(model, config):
    def loss_fn(
        x,
        params,
        state,
        step_rng,
        train,
    ):
        rng_local = random.fold_in(step_rng, lax.axis_index("batch"))
        sample_rng, dropout_rng = random.split(rng_local)
        # ^ the rng for dropout isn't used by the VAE at the moment, but is used
        # by the AugVAe which also uses this function for creating train and
        # eval steps. This is a small refused bequest, but oh well!

        elbo, ll, kld = model.apply(
            {"params": params},
            x,
            train,
            β=state.β,
            method=model.elbo,
            rngs={"sample": sample_rng, "dropout": dropout_rng},
        )

        if not train and config.get("run_iwlb", False):
            iwlb = model.apply(
                {"params": params},
                x,
                num_samples=config.get("iwlb_num_samples", 50),
                train=train,
                method=model.importance_weighted_lower_bound,
                rngs={"sample": sample_rng, "dropout": dropout_rng},
            )
        else:
            iwlb = jnp.nan

        # add the reconstruction MSE as a metric
        x_recon = model.apply(
            {"params": params},
            x,
            sample_z=False,
            sample_xrecon=False,
            train=train,
            method=model.reconstruct,
            rngs={"sample": sample_rng, "dropout": dropout_rng},
        )
        x_mse = jnp.mean((x - x_recon) ** 2)

        return -elbo, {
            "loss": -elbo,
            "elbo": elbo,
            "ll": ll,
            "kld": kld,
            "iwlb": iwlb,
            "x_mse": x_mse,
        }

    @jax.jit
    def train_step(state, batch):
        step_rng = random.fold_in(state.rng, state.step)

        def batch_loss_fn(params):
            losses, metrics = jax.vmap(
                loss_fn,
                in_axes=(0, None, None, None, None),
                axis_name="batch",
            )(batch["image"][0], params, state, step_rng, True)

            avg_loss = losses.mean(axis=0)

            return avg_loss, metrics

        (_, metrics), grads = jax.value_and_grad(batch_loss_fn, has_aux=True)(
            state.params
        )
        state = state.apply_gradients(grads=grads)

        metrics = state.metrics.update(**metrics)
        logs = ciclo.logs()
        logs.add_stateful_metrics(**metrics.compute())
        logs.add_entry("schedules", "β", state.β)
        logs.add_entry(
            "schedules",
            "lr",
            get_learning_rate(state.opt_state),
        )

        return logs, state.replace(metrics=metrics)

    @jax.jit
    def eval_step(state, batch):
        step_rng = random.fold_in(state.rng, state.step)

        def batch_loss_fn(params):
            _, metrics = jax.vmap(
                loss_fn,
                in_axes=(0, None, None, None, None),
                axis_name="batch",
            )(batch["image"][0], params, state, step_rng, False)

            return metrics

        metrics = batch_loss_fn(state.params)

        metrics = state.metrics.update(**metrics, mask=batch["mask"][0])
        logs = ciclo.logs()
        logs.add_stateful_metrics(**metrics.compute())

        return logs, state.replace(metrics=metrics)

    return train_step, eval_step


def create_vae_optimizer(config):
    return optax.inject_hyperparams(clipped_adamw)(
        optax.warmup_cosine_decay_schedule(
            config.lr * config.init_lr_mult,
            config.lr,
            config.steps * config.warmup_steps_pct,
            config.steps,
            config.lr * config.final_lr_mult,
        ),
        config.get("clip_norm", 2.0),
        config.get("weight_decay", 1e-4),
    )


@flax.struct.dataclass
class VaeMetrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    elbo: metrics.Average.from_output("elbo")
    ll: metrics.Average.from_output("ll")
    kld: metrics.Average.from_output("kld")
    iwlb: metrics.Average.from_output("iwlb")
    x_mse: metrics.Average.from_output("x_mse")

    def update(self, **kwargs) -> "VaeMetrics":
        updates = self.single_from_model_output(**kwargs)
        return self.merge(updates)


class VaeTrainState(train_state.TrainState):
    metrics: VaeMetrics
    rng: PRNGKey
    β: float
    β_schedule: optax.Schedule = flax.struct.field(pytree_node=False)

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            β=self.β_schedule(self.step),
            **kwargs,
        )

    @classmethod
    def create(
        cls,
        *,
        apply_fn,
        params,
        tx,
        β_schedule,
        **kwargs,
    ):
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            β_schedule=β_schedule,
            β=β_schedule(0),
            **kwargs,
        )


def create_vae_state(model, config, rng, input_shape):
    state_rng, init_rng = random.split(rng)
    variables = model.init(
        {"params": init_rng, "sample": init_rng},
        jnp.empty(input_shape),
        train=False,
    )

    parameter_overview.log_parameter_overview(variables)

    params = flax.core.freeze(variables["params"])
    del variables

    opt = create_vae_optimizer(config)

    return VaeTrainState.create(
        apply_fn=VAE.apply,
        params=params,
        tx=opt,
        metrics=VaeMetrics.empty(),
        rng=state_rng,
        β_schedule=optax.cosine_decay_schedule(
            config.β_schedule_init_value,
            config.steps,
            config.β_schedule_final_value / config.β_schedule_init_value,
        ),
    )


def make_vae_plotting_fns(config, model, x):
    n_visualize = config.get("n_visualize", 20)

    x = x[:n_visualize]

    def reconstruction_plot(state):
        @partial(jax.jit, static_argnums=(1, 2))
        def reconstruct(x, sample_z=False, sample_xrecon=False):
            rng = random.fold_in(state.rng, jax.lax.axis_index("image"))
            rng_dropout, rng_sample = random.split(rng)
            return model.apply(
                {"params": state.params},
                x,
                sample_z=sample_z,
                sample_xrecon=sample_xrecon,
                train=False,
                method=model.reconstruct,
                rngs={"dropout": rng_dropout, "sample": rng_sample},
            )

        x_recon_modes = jax.vmap(
            reconstruct, axis_name="image", in_axes=(0, None, None)
        )(x, False, False)

        recon_fig = plot_utils.plot_img_array(
            jnp.concatenate((x, x_recon_modes), axis=0),
            ncol=n_visualize,
            pad_value=1,
            padding=1,
            title="Original | Reconstruction",
        )

        logs = ciclo.logs()
        logs.add_entry("images", "recon", recon_fig)

        return logs, state

    def sampling_plot(state):
        @partial(jax.jit, static_argnums=(1,))
        def sample(rng, sample_x=False):
            rng_dropout, rng_sample = random.split(rng)
            return model.apply(
                {"params": state.params},
                sample_x=sample_x,
                train=False,
                method=model.sample,
                rngs={"dropout": rng_dropout, "sample": rng_sample},
            )

        sampled_data = jax.vmap(sample, in_axes=(0, None))(
            jax.random.split(state.rng, n_visualize), True
        )

        sample_fig = plot_utils.plot_img_array(
            sampled_data,
            ncol=n_visualize,
            pad_value=1,
            padding=1,
            title="Sampled data",
        )

        logs = ciclo.logs()
        logs.add_entry("images", "sample", sample_fig)

        return logs, state

    return reconstruction_plot, sampling_plot
