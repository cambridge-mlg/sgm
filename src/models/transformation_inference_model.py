"""Transformation inference model implementation.

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
from chex import PRNGKey
from clu import metrics, parameter_overview
from flax import traverse_util
from flax.linen import initializers as init
from flax.training import train_state
from jax import lax
from jax.scipy import linalg
from ml_collections import config_dict

from src.models.utils import clipped_adamw, huber_loss
from src.transformations.affine import (
    gen_affine_matrix_no_shear,
    transform_image_with_affine_matrix,
)
from src.utils.blur import gaussian_filter2d


class TransformationInferenceNet(nn.Module):
    hidden_dims: Sequence[int]
    bounds: Optional[Sequence[float]] = None
    offset: Optional[Sequence[float]] = None
    σ_init: float = np.log(np.exp(0.01) - 1.0)  # softplus(σ_init) = 0.01
    squash_to_bounds: bool = False
    use_layernorm: bool = True

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
    def __call__(self, x, train: bool = False) -> distrax.Transformed:
        h = x.flatten()

        for hidden_dim in self.hidden_dims:
            h = nn.Dense(hidden_dim)(h)
            h = nn.gelu(h)
            if self.use_layernorm:
                h = nn.LayerNorm()(h)

        output_dim = np.prod(self.event_shape)
        μ = nn.Dense(
            output_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )(h)

        σ = jax.nn.softplus(
            self.param("σ_", init.constant(self.σ_init), self.event_shape)
        )

        base = distrax.Independent(
            distrax.Normal(loc=μ, scale=σ), len(self.event_shape)
        )

        bijectors = [
            distrax.Block(
                distrax.ScalarAffine(shift=self.offset_array, scale=self.bounds_array),
                len(self.event_shape),
            )
        ]

        if self.squash_to_bounds:
            bijectors.append(distrax.Block(distrax.Tanh(), len(self.event_shape)))

        bijector = distrax.Chain(bijectors)

        return distrax.Transformed(base, bijector)


def make_transformation_inference_train_and_eval(
    model: TransformationInferenceNet, config: config_dict.ConfigDict
):
    transform_image_fn = jax.jit(
        functools.partial(
            transform_image_with_affine_matrix,
            order=config.interpolation_order,
            fill_value=-1.0,
            fill_mode="constant",
        )
    )
    # Currently `gen_affine_augment` and `gen_affine_model` are the same, but it might make sense for them
    # to be different
    gen_affine_augment = gen_affine_matrix_no_shear
    gen_affine_model = gen_affine_matrix_no_shear

    def invertibility_loss_fn(x_, affine_mat, affine_mat_inv):
        transformed_x = transform_image_fn(
            x_,
            affine_mat,
        )
        untransformed_x = transform_image_fn(
            transformed_x,
            affine_mat_inv,
        )
        mse = optax.squared_error(
            untransformed_x,
            x_,
        )
        return mse.mean()

    def loss_fn(
        x,  # [height, width, num_channels]
        params,
        state,
        step_rng,
        train,
    ):
        rng_local = random.fold_in(step_rng, lax.axis_index("batch"))

        if train and config.blur_σ_init > 0.0:
            # Possibly blur the sample (lax.cond useful for defining schedules that go to 0.):
            x = jax.lax.cond(
                state.blur_sigma
                > 1e-5,  # A small value to avoid numerical issues with a very narrow kernel
                lambda im: gaussian_filter2d(
                    im,
                    sigma=state.blur_sigma,
                    filter_shape=(config.blur_filter_size,) * 2,
                ),
                lambda im: im,
                x,
            )

        def per_sample_loss(rng):
            """
            The self-supervised loss for the generative network can be summarised with the following diagram:
                      ┌───────┐            ┌─────────────┐
                ┌────►│n_rand1├──x_rand1──►│n_x_rand1.inv├───x_hat1──┐
                │     └───────┘            └─────────────┘           │
                │                                                    ▼
                x                                                   mse
                │                                                    ▲
                │     ┌───────┐            ┌─────────────┐           │
                └────►│n_rand2├──x_rand2──►│n_x_rand2.inv├───x_hat2──┘
                      └───────┘            └─────────────┘
            However, implementing this directly requires doing 4 affine transformations, which adds 'blur' to the image.
            Instead, we note the equivalence between computing the MSE loss between two different reconstruction of
            x_hat and the MSE loss between x and the re-reconstruction of x as shown below.
                      ┌───────┐             ┌─────────────┐
                x ───►│n_rand1├───x_rand1──►│n_x_rand1.inv├───┐
                │     └───────┘             └─────────────┘   │
                ▼                                             │
               mse                                          x_hat
                ▲                                             │
                │   ┌───────────┐             ┌─────────┐     │
                x'──┤n_rand2.inv│◄───x_rand2──┤n_x_rand2│◄────┘
                    └───────────┘             └─────────┘
            Finally, we note that the diagram above is equivalent to (assuming invertible group actions):
                x──────────────────────────────┐
                │                              │
                │                              ▼
                ▼     ┌─────────────────────────────────────────────────┐
               mse    │n_rand2.inv ∘ n_x_rand2 ∘ n_x_rand1.inv ∘ n_rand1│
                ▲     └────────────────────────┬────────────────────────┘
                │                              │
                x'─────────────────────────────┘
            Where ∘ denotes composition of group actions.
            """
            (
                rng_sample1,
                rng_sample2,
                rng_η_rand1,
                rng_η_rand2,
            ) = random.split(rng, 4)

            Η_rand = distrax.Uniform(
                low=-jnp.array(config.augment_bounds)
                + jnp.array(config.augment_offset),
                high=jnp.array(config.augment_bounds)
                + jnp.array(config.augment_offset),
            )
            η_rand1 = Η_rand.sample(seed=rng_η_rand1, sample_shape=())
            η_rand2 = Η_rand.sample(seed=rng_η_rand2, sample_shape=())

            η_rand1_aff_mat = gen_affine_augment(η_rand1)
            η_rand2_aff_mat = gen_affine_augment(η_rand2)
            η_rand2_inv_aff_mat = linalg.inv(η_rand2_aff_mat)

            x_rand1 = transform_image_fn(x, η_rand1_aff_mat)
            x_rand2 = transform_image_fn(x, η_rand2_aff_mat)

            q_H_x_rand1 = model.apply({"params": params}, x_rand1, train)
            q_H_x_rand2 = model.apply({"params": params}, x_rand2, train)

            η_x_rand1 = q_H_x_rand1.sample(seed=rng_sample1)
            η_x_rand2 = q_H_x_rand2.sample(seed=rng_sample2)

            η_x_rand1_aff_mat = gen_affine_model(η_x_rand1)
            η_x_rand1_inv_aff_mat = linalg.inv(η_x_rand1_aff_mat)
            η_x_rand2_aff_mat = gen_affine_model(η_x_rand2)

            x_mse = optax.squared_error(
                x,
                transform_image_fn(
                    x,
                    η_rand1_aff_mat
                    @ η_x_rand1_inv_aff_mat
                    @ η_x_rand2_aff_mat
                    @ η_rand2_inv_aff_mat,
                ),
            ).mean()

            # This loss regularises for the applied sequence of transformations to be identity, but directly in η space.
            # However, it is possible for it to be non-0 while the MSE is perfect due to self-symmetric objects
            # (multiple orbit stabilizers). Ultimately, we care about low MSE. The loss is chosen to be more-or-less
            # linear in the difference (hence the Huber loss) because we expect the error distribution to be multimodal,
            # and we want the model to settle on one of the modes. The square loss would make the model settle inbetween
            # the modes (at the mean of the modes).
            η_recon_loss = huber_loss(
                # Measure how close the transformation is to identity
                (
                    η_rand1_aff_mat
                    @ η_x_rand1_inv_aff_mat
                    @ η_x_rand2_aff_mat
                    @ η_rand2_inv_aff_mat
                )[:2, :].ravel(),
                jnp.eye(3, dtype=η_rand2_aff_mat.dtype)[:2, :].ravel(),
                slope=1,
                radius=1e-2,  # Choose a relatively small delta - want the loss to be mostly linear.
            ).mean()

            invertibility_loss = invertibility_loss_fn(
                x_rand1,
                affine_mat=η_x_rand1_inv_aff_mat,
                affine_mat_inv=η_x_rand1_aff_mat,
            )

            return x_mse, η_recon_loss, invertibility_loss

        rngs = random.split(rng_local, config.n_samples)
        x_mse, η_recon_loss, invertibility_loss = jax.vmap(per_sample_loss)(rngs)
        x_mse, η_recon_loss, invertibility_loss = jax.tree_map(
            lambda x: x.mean(), (x_mse, η_recon_loss, invertibility_loss)
        )

        loss = (
            config.x_mse_loss_mult * x_mse
            + config.η_loss_mult * η_recon_loss
            + config.invertibility_loss_mult * invertibility_loss
        )

        return loss, {
            "loss": loss,
            "x_mse": x_mse,
            "η_recon_loss": η_recon_loss,
            "invertibility_loss": invertibility_loss,
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
        logs.add_entry(
            "schedules",
            "lr_inf",
            state.opt_state.inner_states["inference"][0].hyperparams["learning_rate"],
        )
        logs.add_entry(
            "schedules",
            "blur_sigma",
            state.blur_sigma,
        )
        logs.add_entry(
            "schedules",
            "lr_σ",
            state.opt_state.inner_states["σ"][0].hyperparams["learning_rate"],
        )
        logs.add_entry("parameters", "σ", jax.nn.softplus(state.params["σ_"]).mean())
        logs.add_entry("gradients", "grad_norm", optax.global_norm(grads))

        return logs, state.replace(metrics=metrics)

    @jax.jit
    def eval_step(state, batch):
        step_rng = random.fold_in(state.rng, state.step)

        def mse_same_label_examples(xs, labels, mask, rng):
            """
            Compute MSE between a pair of images transformed "into each other" through
            the prototype model, where the pair of images is chosen to have the same label
            (and hence, for some datasets, likely a similar prototype).

            The "transforming into each other" happens by first transforming the original image
            into the prototype, and then transforming the prototype into the paired image
            by using (inverse of) the canonicalization transform of the paired image.
            """
            sample_rng, pair_match_rng = random.split(rng, 2)

            # Create a mask that excludes self-matching images
            num_images = xs.shape[0]
            self_match_mask = jnp.eye(
                num_images, dtype=bool
            )  # [batch_size, batch_size]

            # Create a mask that matches labels between original and transformed images
            label_match_mask = jnp.equal(
                jnp.expand_dims(labels, axis=-1), labels
            )  # [batch_size, batch_size]
            mask_mask = jnp.expand_dims(mask, axis=0)  # [1, batch_size]

            match_mask = jnp.logical_and(label_match_mask, ~self_match_mask)
            match_mask = jnp.logical_and(match_mask, mask_mask)

            # Randomly select an image from the transformed images that matches the label of the original image
            @jax.jit
            def get_matching_index(is_matching, rng) -> tuple:
                num_idxs = is_matching.shape[0]
                any_matching = is_matching.any()
                return jax.lax.cond(
                    any_matching,
                    lambda: (
                        random.choice(rng, num_idxs, p=is_matching / is_matching.sum()),
                        True,
                    ),
                    lambda: (0, False),
                )

            pair_match_rngs = random.split(pair_match_rng, num_images)
            paired_idxs, has_matching = jax.vmap(get_matching_index, in_axes=(0, 0))(
                match_mask, pair_match_rngs
            )
            paired_xs = jax.vmap(lambda idx, xs_: xs_[idx], in_axes=(0, None))(
                paired_idxs, xs
            )

            # Now transform the original image using the eta from the transformed image
            def per_example_paired_image_mse(x1, x2, rng):
                rng_sample1, rng_sample2 = random.split(rng, 2)
                p_η_x1 = model.apply({"params": state.params}, x1, False)
                η1 = p_η_x1.sample(seed=rng_sample1)
                p_η_x2 = model.apply({"params": state.params}, x2, False)
                η2 = p_η_x2.sample(seed=rng_sample2)

                η1_aff_mat = gen_affine_model(η1)
                η1_inv_aff_mat = linalg.inv(η1_aff_mat)
                η2_aff_mat = gen_affine_model(η2)
                η2_inv_aff_mat = linalg.inv(η2_aff_mat)

                x1_hat = transform_image_fn(x1, η1_inv_aff_mat)
                x2_recon = transform_image_fn(x1_hat, η2_aff_mat)
                return optax.squared_error(x2, x2_recon).mean()

            # Return MSE of 0 if there is no label match
            return (
                jax.vmap(per_example_paired_image_mse, in_axes=(0, 0, 0))(
                    xs, paired_xs, random.split(sample_rng, num_images)
                ),
                has_matching,
            )

        label_paired_image_mse, has_matching = mse_same_label_examples(
            batch["image"][0], batch["label"][0], batch["mask"][0], step_rng
        )
        mean_label_paired_image_mse = (
            label_paired_image_mse * has_matching
        ).sum() / has_matching.sum()

        # Loss function metrics
        _, metrics = jax.vmap(
            loss_fn,
            in_axes=(0, None, None, None, None),
            axis_name="batch",
        )(batch["image"][0], state.params, state, step_rng, False)

        metrics = state.metrics.update(**metrics, mask=batch["mask"][0])
        logs = ciclo.logs()
        logs.add_stateful_metrics(**metrics.compute())
        logs.add_metric("label_paired_image_mse", mean_label_paired_image_mse)

        return logs, state.replace(metrics=metrics)

    return train_step, eval_step


def create_transformation_inference_optimizer(params, config):
    partition_optimizers = {
        "inference": optax.inject_hyperparams(clipped_adamw)(
            optax.warmup_cosine_decay_schedule(
                config.init_lr_mult * config.lr,
                config.lr,
                config.steps * config.warmup_steps_pct,
                config.steps,
                config.lr * config.final_lr_mult,
            ),
            config.get("clip_norm", 2.0),
            # Optax WD default: 0.0001 https://optax.readthedocs.io/en/latest/api.html#optax.adamw
            config.get("weight_decay", 0.0001),
        ),
        "σ": optax.inject_hyperparams(optax.adam)(
            optax.warmup_cosine_decay_schedule(
                config.σ_lr,
                config.σ_lr * 3,
                config.steps * config.warmup_steps_pct,
                config.steps,
                config.σ_lr / 3,
            ),
        ),
    }

    def get_partition(path, value):
        if "σ_" in path:
            return "σ"
        else:
            return "inference"

    param_partitions = flax.core.freeze(
        traverse_util.path_aware_map(get_partition, params)
    )
    return optax.multi_transform(partition_optimizers, param_partitions)


@flax.struct.dataclass
class TransformationInferenceMetrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    x_mse: metrics.Average.from_output("x_mse")
    η_recon_loss: metrics.Average.from_output("η_recon_loss")
    invertibility_loss: metrics.Average.from_output("invertibility_loss")

    def update(self, **kwargs) -> "TransformationInferenceMetrics":
        updates = self.single_from_model_output(**kwargs)
        return self.merge(updates)


class TransformationInferenceTrainState(train_state.TrainState):
    metrics: TransformationInferenceMetrics
    blur_sigma: float
    blur_sigma_schedule: optax.Schedule = flax.struct.field(pytree_node=False)
    rng: PRNGKey

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            blur_sigma=self.blur_sigma_schedule(self.step),
            **kwargs,
        )

    @classmethod
    def create(
        cls,
        *,
        apply_fn,
        params,
        tx,
        blur_sigma_schedule,
        **kwargs,
    ):
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            blur_sigma_schedule=blur_sigma_schedule,
            blur_sigma=blur_sigma_schedule(0),
            **kwargs,
        )


def create_transformation_inference_state(model, config, rng, input_shape):
    state_rng, init_rng = random.split(rng, 2)
    variables = model.init(
        {"params": init_rng, "sample": init_rng},
        jnp.empty(input_shape),
        train=False,
    )

    parameter_overview.log_parameter_overview(variables)

    params = flax.core.freeze(variables["params"])
    del variables

    opt = create_transformation_inference_optimizer(params, config)
    return TransformationInferenceTrainState.create(
        apply_fn=TransformationInferenceNet.apply,
        params=params,
        tx=opt,
        metrics=TransformationInferenceMetrics.empty(),
        blur_sigma_schedule=optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=config.blur_σ_init,
                    end_value=0.0,
                    transition_steps=config.steps * config.blur_end_pct,
                ),
                optax.constant_schedule(0.0),
            ],
            boundaries=[config.steps * config.blur_end_pct],
        ),
        rng=state_rng,
    )
