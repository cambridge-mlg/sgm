from typing import Callable, Optional, Sequence, Union

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from chex import Array
import flax
import flax.linen as nn
from flax.linen import initializers as init
from flax import traverse_util
import distrax
import optax
import ciclo

from src.transformations import transform_image
from src.models.utils import clipped_adamw
from src.utils.types import KwArgs
from src.models.proto_model import AugmentGenerativeMetrics, TrainStateWithMetrics


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
    hidden_dims: Sequence[int]
    num_flows: int
    num_bins: int
    bounds: Sequence[int]
    offset: Optional[Sequence[int]] = None
    conditioner: Optional[KwArgs] = None
    ε: float = 1e-6

    def setup(self) -> None:
        self.bounds_array = jnp.array(self.bounds)
        self.offset_array = (
            jnp.array(self.offset)
            if self.offset is not None
            else jnp.zeros_like(self.bounds_array)
        )
        self.event_shape = self.bounds_array.shape

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
                # distrax.Block(distrax.Tanh(), len(self.event_shape)),
                # We invert the flow so that the `forward` method is called with `log_prob`.
                distrax.Inverse(distrax.Chain(layers)),
            ]
        )

        return distrax.Transformed(base, bijector)


def make_augment_generative_train_and_eval(
    config,
    model: TransformationGenerativeNet,
    canon_function: Callable[[Array, Array], Array],
):
    def loss_fn(
        x,
        params,
        state,
        step_rng,
        train,
    ):
        rng_local = random.fold_in(step_rng, lax.axis_index("batch"))

        η_x = canon_function(x, rng_local)

        x_hat = transform_image(x, -η_x, order=config.interpolation_order)

        p_Η_x_hat = model.apply({"params": params}, x_hat, train=train)
        log_p_η_x_hat = p_Η_x_hat.log_prob(η_x)

        # --- TODO: smoothness loss? Bruno: Need to think about it, cause the l2 of grad. doesn't necessarily
        # make sense

        loss = -log_p_η_x_hat

        return loss, {
            "loss": loss,
            "log_p_η_x_hat": log_p_η_x_hat,
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
        logs.add_entry(
            "schedules",
            "lr_gen",
            state.opt_state.inner_states["generative"][0].hyperparams["learning_rate"],
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


def create_augment_generative_optimizer(params, config):
    partition_optimizers = {
        "generative": optax.inject_hyperparams(clipped_adamw)(
            optax.warmup_cosine_decay_schedule(
                config.gen_init_lr_mult * config.gen_lr,
                config.gen_lr,
                config.gen_warmup_steps,
                config.gen_steps,
                config.init_lr * config.gen_final_lr_mult,
            ),
            2.0,
        ),
    }

    def get_partition(path, value):
        return "generative"

    param_partitions = flax.core.freeze(
        traverse_util.path_aware_map(get_partition, params)
    )
    return optax.multi_transform(partition_optimizers, param_partitions)


def create_augment_generative_state(params, rng, config):
    opt = create_augment_generative_optimizer(params, config)
    return TrainStateWithMetrics.create(
        apply_fn=TransformationGenerativeNet.apply,
        params=params,
        tx=opt,
        metrics=AugmentGenerativeMetrics.empty(),
        rng=rng,
    )
