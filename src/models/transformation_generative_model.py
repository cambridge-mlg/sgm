"""Transformation generative model implementation.

A note on notation. In order to distinguish between random variables and their values, we use upper
and lower case variable names. I.e., p(Z) or `p_Z` is the distribution over the r.v. Z, and is a
function, while p(z) or `p_z` is the probability that Z=z. Similarly, p(X|Z) or `p_X_given_Z` is a
a function which returns another function p(X|z) or `p_X_given_z`, which would return the proability
that X=x|Z=z a.k.a `p_x_given_z`.
"""

import functools
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
from clu import metrics, parameter_overview
from flax.linen import initializers as init
from flax.training import train_state
from jax import lax
from ml_collections import config_dict

from src.models.mlp import MLP
from src.models.utils import clipped_adamw
from src.transformations.transforms import AffineTransformWithoutShear, Transform
from src.utils.types import KwArgs


class Conditioner(nn.Module):
    """A neural network that predicts the parameters of a flow given an input."""

    event_shape: Sequence[int]
    num_bijector_params: int
    hidden_dims: Sequence[int]
    train: Optional[bool] = None
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: Array, h: Array, train: Optional[bool] = None) -> Array:
        train = nn.merge_param("train", self.train, train)

        h = jnp.concatenate((x.flatten(), h.flatten()), axis=0)

        h = MLP(self.hidden_dims, dropout_rate=self.dropout_rate)(h, train=train)

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
    bounds: Optional[Sequence[float]] = None
    offset: Optional[Sequence[float]] = None
    conditioner: Optional[KwArgs] = None
    ε: float = 1e-6
    squash_to_bounds: bool = False
    dropout_rate: float = 0.0
    transform: Transform = AffineTransformWithoutShear

    def setup(self) -> None:
        self.bounds_array = (
            jnp.array(self.bounds)
            if self.bounds is not None
            else jnp.ones_like(self.offset_array)
        )
        self.offset_array = (
            jnp.array(self.offset)
            if self.offset is not None
            else jnp.zeros_like(self.bounds_array)
        )
        self.event_shape = self.bounds_array.shape

    @nn.compact
    def __call__(self, x_hat, train: bool = False, η: Optional[Array] = None):
        h = x_hat.flatten()

        # shared feature extractor
        h = MLP(self.hidden_dims, dropout_rate=self.dropout_rate)(h, train=train)

        # base distribution
        base_hidden = MLP((self.hidden_dims[-1] // 2,) * 2)(h, train=train)

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
        mask = jnp.arange(0, np.prod(self.event_shape)) % 2
        mask = jnp.reshape(mask, self.event_shape)
        mask = mask.astype(bool)

        def bijector_fn(params: Array):
            return distrax.RationalQuadraticSpline(
                params, range_min=-3.0, range_max=3.0
            )

        for _ in range(self.num_flows):
            conditioner = Conditioner(
                event_shape=self.event_shape,
                num_bijector_params=num_bijector_params,
                train=train,
                **(self.conditioner or {}),
            )

            layer = distrax.MaskedCoupling(
                mask=mask,
                bijector=bijector_fn,
                conditioner=functools.partial(conditioner, h=h),
            )

            layers.append(layer)
            mask = ~mask

        bijectors = [
            distrax.Block(
                distrax.ScalarAffine(
                    shift=self.offset_array, scale=self.bounds_array + self.ε
                ),
                len(self.event_shape),
            )
        ]

        if self.squash_to_bounds:
            bijectors.append(distrax.Block(distrax.Tanh(), len(self.event_shape)))

        bijectors.append(distrax.Inverse(distrax.Chain(layers)))

        bijector = distrax.Chain(bijectors)

        transformed = distrax.Transformed(base, bijector)

        if η is not None:
            return transformed, transformed.log_prob(η)
        else:
            return transformed


def make_transformation_generative_train_and_eval(
    model: TransformationGenerativeNet,
    config: config_dict.ConfigDict,
    prototype_function: Callable[[Array, Array], Array],
):
    def loss_fn(
        x,
        params,
        state,
        step_rng,
        train,
    ):
        rng_local = random.fold_in(step_rng, lax.axis_index("batch"))

        def get_xhat_on_random_augmentation(x, rng):
            """
            Rather than obtaining the prototype of `x` directly by passing it into the prototype
            function, augment it randomly and then pass it through to get the prototype.

            This is useful for training the generative model to be robust to imperfections in the
            inference net.
            """
            sample_rng, prototype_fn_rng = random.split(rng)
            Η_rand = distrax.Uniform(
                low=-jnp.array(config.augment_bounds)
                + jnp.array(config.augment_offset),
                high=jnp.array(config.augment_bounds)
                + jnp.array(config.augment_offset),
            )
            η_rand = Η_rand.sample(seed=sample_rng, sample_shape=())
            η_rand_transform = model.transform(η_rand)
            x_rand = η_rand_transform.apply(x)

            η_rand_proto = prototype_function(x_rand, prototype_fn_rng)

            η_rand_proto_transform = model.transform(η_rand_proto)
            η_rand_proto_inv_transform = η_rand_proto_transform.inverse()

            composed_transform = η_rand_proto_inv_transform << η_rand_transform

            return composed_transform.apply(x, order=config.interpolation_order)

        def per_sample_loss_fn(rng):
            η_rng, x_hat_rng, dropout_rng = random.split(rng, 3)

            η_x = prototype_function(x, η_rng)

            x_hat = get_xhat_on_random_augmentation(x, x_hat_rng)

            p_Η_x_hat = model.apply(
                {"params": params}, x_hat, train=train, rngs={"dropout": dropout_rng}
            )
            log_p_η_x_hat = p_Η_x_hat.log_prob(η_x)
            return log_p_η_x_hat

        log_p_η_x_hat = jax.vmap(per_sample_loss_fn)(
            random.split(rng_local, config.n_samples)
        )  # (n_samples,)

        # Regularise p(η|x_hat) densities against small pertubations on x_hat,
        # by minimizing the difference between desitities for slighty different
        # x_hat's, that result from an imperfect inference net.
        pairwise_diffs = jax.vmap(
            jax.vmap(lambda x, y: x - y, in_axes=(0, None)), in_axes=(None, 0)
        )(log_p_η_x_hat, log_p_η_x_hat)
        mae = jnp.abs(pairwise_diffs).mean()

        # --- TODO: smoothness loss? Need to think about it, cause the l2 of grad. doesn't necessarily make sense

        loss = -log_p_η_x_hat.mean() + config.mae_loss_mult * mae

        return loss, {
            "loss": loss,
            "mae": mae,
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
            state.opt_state.hyperparams["learning_rate"],
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


@flax.struct.dataclass
class TransformationGenerativeMetrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    mae: metrics.Average.from_output("mae")
    log_p_η_x_hat: metrics.Average.from_output("log_p_η_x_hat")

    def update(self, **kwargs) -> "TransformationGenerativeMetrics":
        updates = self.single_from_model_output(**kwargs)
        return self.merge(updates)


class TransformationGenerativeTrainState(train_state.TrainState):
    metrics: TransformationGenerativeMetrics
    rng: PRNGKey

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


def create_transformation_generative_optimizer(params, config):
    return optax.inject_hyperparams(clipped_adamw)(
        optax.warmup_cosine_decay_schedule(
            config.init_lr_mult * config.lr,
            config.lr,
            config.steps * config.warmup_steps_pct,
            config.steps,
            config.lr * config.final_lr_mult,
        ),
        config.get("clip_norm", 2.0),
        config.get("weight_decay", 1e-4),
    )


def create_transformation_generative_state(model, config, rng, input_shape):
    state_rng, init_rng = random.split(rng)
    variables = model.init(
        {"params": init_rng, "sample": init_rng},
        jnp.empty(input_shape),
        η=jnp.empty((len(config.augment_bounds),)),
        train=False,
    )

    parameter_overview.log_parameter_overview(variables)

    params = flax.core.freeze(variables["params"])
    del variables

    opt = create_transformation_generative_optimizer(params, config)
    return TransformationGenerativeTrainState.create(
        apply_fn=TransformationGenerativeNet.apply,
        params=params,
        tx=opt,
        metrics=TransformationGenerativeMetrics.empty(),
        rng=state_rng,
    )
