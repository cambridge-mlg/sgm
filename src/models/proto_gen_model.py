from typing import Callable, Optional, Sequence

import ciclo
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
from chex import Array, PRNGKey
from clu import metrics
from flax import traverse_util
from flax.linen import initializers as init
from flax.training import train_state
from jax import lax

from src.transformations import transform_image
from src.utils.types import KwArgs


class TransformationInferenceNet(nn.Module):
    event_shape: Sequence[int]
    hidden_dims: Sequence[int]
    bounds_array: Array
    offset_array: Array
    σ_init: Callable = init.constant(jnp.log(jnp.exp(0.01) - 1.0))

    @nn.compact
    def __call__(self, x, train: bool = False):
        h = x

        # apply average pooling
        h = nn.avg_pool(h, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        h = h.flatten()

        for hidden_dim in self.hidden_dims:
            h = nn.Dense(hidden_dim)(h)
            h = nn.relu(h)

        output_dim = np.prod(self.event_shape)
        μ = nn.Dense(output_dim)(h)

        σ = jax.nn.softplus(self.param("σ_", self.σ_init, self.event_shape))

        base = distrax.Independent(
            distrax.Normal(loc=μ, scale=σ), len(self.event_shape)
        )

        bijector = distrax.Chain(
            [
                distrax.Block(
                    distrax.ScalarAffine(
                        shift=self.offset_array, scale=self.bounds_array
                    ),
                    len(self.event_shape),
                ),
                distrax.Block(distrax.Tanh(), len(self.event_shape)),
            ]
        )

        return distrax.Transformed(base, bijector)


class Conditioner(nn.Module):
    """A neural network that predicts the parameters of a flow given an input."""

    event_shape: Sequence[int]
    num_bijector_params: int
    hidden_dims: Sequence[int]
    train: Optional[bool] = None

    @nn.compact
    def __call__(self, x: Array, train: Optional[bool] = None) -> Array:
        train = nn.merge_param("train", self.train, train)

        h = x.flatten()

        for hidden_dim in self.hidden_dims:
            h = nn.Dense(hidden_dim)(h)
            h = nn.relu(h)

        # We initialize this dense layer to zero so that the flow is initialized to the identity function.
        y = nn.Dense(
            np.prod(self.event_shape) * self.num_bijector_params,
            kernel_init=init.zeros,
            bias_init=init.zeros,
        )(h)
        y = y.reshape(tuple(self.event_shape) + (self.num_bijector_params,))

        return y


class TransformationGenerativeNet(nn.Module):
    event_shape: Sequence[int]
    hidden_dims: Sequence[int]
    bounds_array: Array
    offset_array: Array
    num_flows: int
    num_bins: int
    conditioner: Optional[KwArgs] = None
    ε: float = 1e-6

    @nn.compact
    def __call__(self, x_hat, train: bool = False):
        h = x_hat.flatten()

        # shared feature extractor
        for hidden_dim in self.hidden_dims:
            h = nn.Dense(hidden_dim)(h)
            h = nn.relu(h)

        # base distribution
        base_hidden = nn.Dense(hidden_dim // 2)(h)
        base_hidden = nn.relu(base_hidden)
        base_hidden = nn.Dense(hidden_dim // 2)(base_hidden)
        base_hidden = nn.relu(base_hidden)

        output_dim = np.prod(self.event_shape)
        μ = nn.Dense(output_dim)(base_hidden)
        σ = jax.nn.softplus(nn.Dense(output_dim, name="σ_")(base_hidden))
        base = distrax.Independent(
            distrax.Normal(loc=μ, scale=σ), len(self.event_shape)
        )

        # bijector
        # Number of parameters for the rational-quadratic spline:
        # - `num_bins` bin widths
        # - `num_bins` bin heights
        # - `num_bins + 1` knot slopes
        # for a total of `3 * num_bins + 1` parameters.
        num_bijector_params = 3 * self.num_bins + 1

        layers = []
        for _ in range(self.num_flows):
            params = Conditioner(
                event_shape=self.event_shape,
                num_bijector_params=num_bijector_params,
                train=train,
                **(self.conditioner or {}),
            )(h)
            layer = distrax.Block(
                distrax.RationalQuadraticSpline(params, range_min=-3.0, range_max=3.0),
                len(self.event_shape),
            )
            layers.append(layer)

        bijector = distrax.Chain(
            [
                distrax.Block(
                    distrax.ScalarAffine(
                        shift=self.offset_array, scale=self.bounds_array + self.ε
                    ),
                    len(self.event_shape),
                ),
                distrax.Block(distrax.Tanh(), len(self.event_shape)),
                # We invert the flow so that the `forward` method is called with `log_prob`.
                distrax.Inverse(distrax.Chain(layers)),
            ]
        )

        return distrax.Transformed(base, bijector)


class PrototypicalGenerativeModel(nn.Module):
    bounds: Sequence[int]
    offset: Optional[Sequence[int]] = None
    inference: Optional[KwArgs] = None
    generative: Optional[KwArgs] = None

    def setup(self) -> None:
        self.bounds_array = jnp.array(self.bounds)
        self.offset_array = (
            jnp.array(self.offset)
            if self.offset is not None
            else jnp.zeros_like(self.bounds_array)
        )
        self.event_shape = self.bounds_array.shape
        self.inference_net = TransformationInferenceNet(
            event_shape=self.event_shape,
            bounds_array=self.bounds_array,
            offset_array=self.offset_array,
            **(self.inference or {}),
        )
        self.generative_net = TransformationGenerativeNet(
            event_shape=self.event_shape,
            bounds_array=self.bounds_array,
            offset_array=self.offset_array,
            **(self.generative or {}),
        )

    def __call__(self, x, train: bool = False):
        q_H_x = self.inference_net(x, train=train)
        η = q_H_x.sample(seed=self.make_rng("sample"))

        x_hat = transform_image(x, -η)

        p_H_x_hat = self.generative_net(x_hat, train=train)
        log_p_η_x_hat = p_H_x_hat.log_prob(η)

        return η, log_p_η_x_hat

    def generative_net_ll(self, x_hat, η, train: bool = False):
        p_H_x_hat = self.generative_net(x_hat, train=train)
        return p_H_x_hat.log_prob(η)


def make_pgm_train_and_eval(config, model):
    def loss_fn(
        x,
        params,
        state,
        step_rng,
        train,
    ):
        """
        The self-supervised loss for the generative network can be summarised with the following diagram

                x ------- -η_x -----> x_hat
                |                       |
                |                       v
            η_rand                    mse
                |                       ∧
                ∨                       |
            x_rand --- -η_x_rand ---> x_hat'.

        However, implementing this directly requires doing 3 affine transformations, which adds 'blur' to the image.
        So instead we note that the diagram above is equivalent to

                x --------> mse <------- x'
                |                        ∧
                |                        |
            η_rand                     η_x
                |                        |
                v                        |
            x_rand --- -η_x_rand ---> x_hat'.

        Finally, this computation can be simplified to

                x --------> mse <-------- x'
                |                         ∧
                └ η_rand - η_x_rand + η_x ┘

        which contains only a single transformation.

        """
        rng_local = random.fold_in(step_rng, lax.axis_index("batch"))

        def per_sample_loss(rng):
            (
                rng_sample1,
                rng_sample2,
                rng_η_rand,
            ) = random.split(rng, 3)

            η_x, _ = model.apply(
                {"params": params}, x, train, rngs={"sample": rng_sample1}
            )

            Η_rand = distrax.Uniform(
                low=-jnp.array(config.model.bounds) + jnp.array(config.model.offset),
                high=jnp.array(config.model.bounds) + jnp.array(config.model.offset),
            )
            η_rand = Η_rand.sample(seed=rng_η_rand, sample_shape=())

            x_rand = transform_image(x, η_rand)
            η_x_rand, _ = model.apply(
                {"params": params}, x_rand, train, rngs={"sample": rng_sample2}
            )

            x_mse = optax.squared_error(
                x, transform_image(x, η_rand - η_x_rand + η_x)
            ).mean()

            difficulty = optax.squared_error(x, x_rand).mean()

            x_hat = jax.lax.stop_gradient(transform_image(x, η_rand - η_x_rand))

            # Use the gradient of the density w.r.t η to regularises log_p_η_x_hat,
            # so that you'd get a smooth density from the generative model.
            def get_log_p_η_x_hat(η_x):
                return model.apply(
                    {"params": params},
                    x_hat,
                    η_x,
                    train=train,
                    method=model.generative_net_ll,
                )

            log_p_η_x_hat, η_grad = jax.value_and_grad(get_log_p_η_x_hat)(
                jax.lax.stop_gradient(η_x)
            )

            return x_mse, difficulty, log_p_η_x_hat, jnp.abs(η_grad).mean()

        rngs = random.split(rng_local, config.n_samples)
        x_mse, difficulty, log_p_η_x_hat, η_grad = jax.vmap(per_sample_loss)(rngs)

        # (maybe) do a weighted average based on the difficulty of the sample
        weights = (
            difficulty / difficulty.sum()
            if train and config.difficulty_weighted_inf_loss
            else jnp.ones((config.n_samples,)) / config.n_samples
        )
        x_mse = (x_mse * weights).sum(axis=0)

        # Regularise p(η|x_hat) densities against small pertubations on x_hat,
        # by minimizing the difference between desitities for slighty different
        # x_hat's, that result from an imperfect inference net.
        pairwise_diffs = jax.vmap(
            jax.vmap(lambda x, y: x - y, in_axes=(0, None)), in_axes=(None, 0)
        )(log_p_η_x_hat, log_p_η_x_hat)
        mae = jnp.abs(pairwise_diffs).mean()

        log_p_η_x_hat, η_grad = jax.tree_map(
            lambda x: x.mean(axis=0), (log_p_η_x_hat, η_grad)
        )

        loss = (
            x_mse
            - log_p_η_x_hat * state.λ
            + mae * (1 - state.α)
            + η_grad * (1 - state.β)
        )

        return loss, {
            "loss": loss,
            "x_mse": x_mse,
            "log_p_η_x_hat": log_p_η_x_hat,
            "mae": mae,
            "η_grad": η_grad,
        }

    # TODO: figure out how to make this work with jax.pmap
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
        logs.add_entry("schedules", "λ", state.λ)
        logs.add_entry("schedules", "α", state.α)
        logs.add_entry("schedules", "β", state.β)
        logs.add_entry(
            "schedules",
            "lr_inf",
            state.opt_state.inner_states["inference"][0].hyperparams["learning_rate"],
        )
        logs.add_entry(
            "schedules",
            "lr_gen",
            state.opt_state.inner_states["generative"][0].hyperparams["learning_rate"],
        )
        logs.add_entry(
            "schedules",
            "lr_σ",
            state.opt_state.inner_states["σ"][0].hyperparams["learning_rate"],
        )
        logs.add_entry("gradients", "grad_norm", optax.global_norm(grads))

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


def create_pgm_optimizer(params, config):
    def clipped_adamw(learning_rate, norm):
        return optax.MultiSteps(
            optax.chain(
                optax.clip_by_global_norm(norm),
                optax.adamw(learning_rate=learning_rate),
            ),
            1,
        )

    partition_optimizers = {
        "inference": optax.inject_hyperparams(clipped_adamw)(
            optax.join_schedules(
                [
                    optax.warmup_cosine_decay_schedule(
                        config.inf_init_lr,
                        config.inf_init_lr * config.inf_peak_lr_mult,
                        config.inf_warmup_steps,
                        config.inf_steps,
                        config.inf_init_lr * config.inf_final_lr_mult,
                    ),
                    optax.constant_schedule(0.0),
                ],
                [
                    config.inf_steps,
                ],
            ),
            2.0,
        ),
        "σ": optax.inject_hyperparams(optax.adam)(
            optax.join_schedules(
                [
                    optax.warmup_cosine_decay_schedule(
                        config.σ_lr,
                        config.σ_lr * 3,
                        config.inf_warmup_steps,
                        config.inf_steps,
                        config.σ_lr / 3,
                    ),
                    optax.constant_schedule(0.0),
                ],
                [
                    config.inf_steps,
                ],
            )
        ),
        "generative": optax.inject_hyperparams(clipped_adamw)(
            optax.join_schedules(
                [
                    optax.constant_schedule(0.0),
                    optax.warmup_cosine_decay_schedule(
                        config.gen_init_lr,
                        config.gen_init_lr * config.gen_peak_lr_mult,
                        config.gen_warmup_steps,
                        config.gen_steps,
                        config.gen_init_lr * config.gen_final_lr_mult,
                    ),
                ],
                [
                    config.inf_steps,
                ],
            ),
            2.0,
        ),
    }

    def get_partition(path, value):
        if "generative_net" in path:
            return "generative"

        if "inference_net" in path:
            if "σ_" in path:
                return "σ"

            return "inference"

    param_partitions = flax.core.freeze(
        traverse_util.path_aware_map(get_partition, params)
    )
    return optax.multi_transform(partition_optimizers, param_partitions)


@flax.struct.dataclass
class PgmMetrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    x_mse: metrics.Average.from_output("x_mse")
    log_p_η_x_hat: metrics.Average.from_output("log_p_η_x_hat")
    mae: metrics.Average.from_output("mae")
    η_grad: metrics.Average.from_output("η_grad")

    def update(self, **kwargs) -> "PgmMetrics":
        updates = self.single_from_model_output(**kwargs)
        return self.merge(updates)


class PgmTrainState(train_state.TrainState):
    metrics: PgmMetrics
    rng: PRNGKey
    λ: float
    λ_schedule: optax.Schedule = flax.struct.field(pytree_node=False)
    α: float
    α_schedule: optax.Schedule = flax.struct.field(pytree_node=False)
    β: float
    β_schedule: optax.Schedule = flax.struct.field(pytree_node=False)

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            λ=self.λ_schedule(self.step),
            α=self.α_schedule(self.step),
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
        λ_schedule,
        α_schedule,
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
            λ_schedule=λ_schedule,
            λ=λ_schedule(0),
            α_schedule=α_schedule,
            α=α_schedule(0),
            β_schedule=β_schedule,
            β=β_schedule(0),
            **kwargs,
        )


def create_pgm_state(params, rng, config):
    opt = create_pgm_optimizer(params, config)

    return PgmTrainState.create(
        apply_fn=PrototypicalGenerativeModel.apply,
        params=params,
        tx=opt,
        metrics=PgmMetrics.empty(),
        rng=rng,
        λ_schedule=optax.join_schedules(
            [
                optax.constant_schedule(0.0),
                optax.constant_schedule(1.0),
            ],
            [
                config.inf_steps,
            ],
        ),
        α_schedule=optax.join_schedules(
            [
                optax.constant_schedule(1.0),
                optax.cosine_decay_schedule(
                    1.0,
                    int(config.gen_steps * config.α_schedule_pct),
                    config.α_schedule_final_value,
                ),
            ],
            [
                config.inf_steps,
            ],
        ),
        β_schedule=optax.join_schedules(
            [
                optax.constant_schedule(1.0),
                optax.cosine_decay_schedule(
                    1.0,
                    int(config.gen_steps * config.β_schedule_pct),
                    config.β_schedule_final_value,
                ),
            ],
            [
                config.inf_steps,
            ],
        ),
    )
