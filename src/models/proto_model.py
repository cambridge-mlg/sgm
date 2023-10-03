import functools
from typing import Callable, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.scipy import linalg
from chex import PRNGKey
import flax
import flax.linen as nn
from flax.linen import initializers as init
from flax.training import train_state
from flax import traverse_util
import distrax
from clu import metrics
import optax
import ciclo
from src.models.convnext import ConvNeXt, get_convnext_constructor

from src.transformations import transform_image
from src.models.utils import clipped_adamw, huber_loss
from src.utils.types import KwArgs
from src.transformations.affine import transform_image_with_affine_matrix
from src.transformations.affine import gen_affine_matrix_no_shear as gen_affine


class TransformationInferenceNet(nn.Module):
    hidden_dims: Sequence[int]
    bounds: Sequence[int]
    offset: Optional[Sequence[int]] = None
    σ_init: Callable = init.constant(jnp.log(jnp.exp(0.01) - 1.0))
    squash_to_bounds: bool = False
    model_type: str = "mlp" # "mlp" or "convnext"
    convnext_type: str = "tiny"
    convnext_depths: Optional[Sequence[int]] = None
    convnext_dims: Optional[Sequence[int]] = None
    # default factory kwargs for convnext
    convnext_kwargs: Optional[KwArgs] = None

    def setup(self) -> None:
        self.bounds_array = jnp.array(self.bounds)
        self.offset_array = (
            jnp.array(self.offset)
            if self.offset is not None
            else jnp.zeros_like(self.bounds_array)
        )
        self.event_shape = self.bounds_array.shape

    @nn.compact
    def __call__(self, x, train: bool = False):
        match self.model_type:
            case "mlp":
                h = x.flatten()

                for hidden_dim in self.hidden_dims:
                    h = nn.Dense(hidden_dim)(h)
                    h = nn.gelu(h)
                    h = nn.LayerNorm()(h)

            case "convnext":
                constructor_kwargs = dict(num_outputs=1024, in_channels=1, init_downsample=1, **(self.convnext_kwargs or {}))
                match self.convnext_type:
                    case "custom":
                        assert self.convnext_depths is not None
                        assert self.convnext_dims is not None
                        convnext = ConvNeXt(dims=self.convnext_dims, depths=self.convnext_depths, **constructor_kwargs)
                    case _:
                        convnext = get_convnext_constructor(self.convnext_type)(**constructor_kwargs)
                h = convnext(x)
                h = nn.LayerNorm()(h)  # no layer norm might actually help with scaling the right way?
            case _:
                raise ValueError(f"Unknown model type {self.model_type}")

        output_dim = np.prod(self.event_shape)
        μ = nn.Dense(output_dim, kernel_init=nn.initializers.zeros_init(), bias_init=nn.initializers.zeros_init())(h)
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
            ]
            + ([
                distrax.Block(distrax.Tanh(), len(self.event_shape)),
            ] if self.squash_to_bounds else [])
        )

        return distrax.Transformed(base, bijector)


def make_canonicalizer_train_and_eval(config, model: TransformationInferenceNet):
    transform_image_fn = jax.jit(
        functools.partial(transform_image_with_affine_matrix, order=config.interpolation_order, fill_value=-1., fill_mode="constant")
    )

    def invertibility_loss_fn(x_, η_affine_mat, η_inv_affine_mat):
        transformed_x = transform_image_fn(
            x_,
            η_affine_mat,
        )
        untransformed_x = transform_image_fn(
            transformed_x,
            # linalg.inv(η_affine_mat),
            η_inv_affine_mat,
        )
        # mse = (transformed_x - untransformed_x)**2
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

        def nonsymmetrised_per_sample_loss(rng):
            """TODO: Old loss: delete? Keep under a flag for comparison?"""
            """
            The self-supervised loss for the generative network can be summarised with the following diagram

                    x ------- -η_x -----> x_hat
                    ∧                       |
                    |                       v
                    x                    mse
                    |                       ∧
                    ∨                       |
                x_rand --- -η_x_rand ---> x_hat'.

                    x ------- -η_x -----> x_hat
                    ∧                       |
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

            (
                rng_sample1,
                rng_sample2,
                rng_η_rand1,
            ) = random.split(rng, 3)

            Η_rand = distrax.Uniform(
                # Separate model bounds and augment bounds
                low=-jnp.array(config.augment_bounds) * state.augment_bounds_mult
                + jnp.array(config.augment_offset),
                high=jnp.array(config.augment_bounds) * state.augment_bounds_mult
                + jnp.array(config.augment_offset),
            )
            η_rand1 = Η_rand.sample(seed=rng_η_rand1, sample_shape=())

            x_rand1 = transform_image(x, η_rand1, order=config.interpolation_order)
            q_H_x_rand1 = model.apply({"params": params}, x_rand1, train)
            q_H_x = model.apply({"params": params}, x, train)

            η_x_rand1 = q_H_x_rand1.sample(seed=rng_sample1)
            η_x = q_H_x.sample(seed=rng_sample2)
            # Get the affine matrices
            η_rand1_aff_mat = gen_affine(η_rand1)
            η_x_rand1_aff_mat = gen_affine(η_x_rand1)
            η_x_rand1_inv_aff_mat = linalg.inv(η_x_rand1_aff_mat)  # Inv. faster than matrix exponential
            η_x_aff_mat = gen_affine(η_x)
            η_x_inv_aff_mat = linalg.inv(η_x_aff_mat)

            x_mse = optax.squared_error(
                x,
                transform_image_fn(
                    x,
                    η_x_aff_mat @ η_x_rand1_inv_aff_mat @ η_rand1_aff_mat,
                ),
            ).mean()

            difficulty = optax.squared_error(x_rand1, x).mean()

            # This loss is a useful proxy and can provide a helpful learning signal.
            # However, it is possible for it to be non-0 while the MSE is perfect due to 
            # self-symmetric objects (multiple orbit stabilizers). Ultimately, we care about
            # low MSE.
            # The loss is chosen to be more-or-less linear in the difference (hence the Huber loss)
            # because we expect the error distribution to be multimodal, and we want the model
            # to settle on one of the modes. The square loss would make the model settle inbetween
            # the modes (at the mean of the modes)
            η_recon_loss = huber_loss(
                # Measure how close the transformation is to identity
                (η_x_aff_mat @ η_x_rand1_inv_aff_mat @ η_rand1_aff_mat)[:2, :].ravel(),
                jnp.eye(3, dtype=η_rand1_aff_mat.dtype)[:2, :].ravel(),
                slope=1,
                radius=1e-2,  # Choose a relatively small delta - want the loss to be mostly linear
            ).mean()

            invertibility_loss = invertibility_loss_fn(x_rand1, η_affine_mat=η_x_rand1_inv_aff_mat, η_inv_affine_mat=η_x_rand1_aff_mat)

            return x_mse, η_recon_loss, invertibility_loss, difficulty

        def symmetrised_per_sample_loss(rng):
            """
            The self-supervised loss for the generative network can be summarised with the following diagram:

                      ┌───────┐            ┌────────────┐
                ┌─────┤n_rand1├──►x_rand1──┤(-n_x_rand1)├───┐
                │     └───────┘            └────────────┘   │
                │                                           ▼
                x                                         x_hat
                │                                           ▲
                │     ┌───────┐            ┌────────────┐   │
                └─────┤n_rand2├──►x_rand2──┤(-n_x_rand2)├───┘
                      └───────┘            └────────────┘

            Instead of computing the MSE loss between two different reconstruction of x_hat, we instead
            compute the MSE loss between x and the re-reconstruction of x as shown below:


                      ┌───────┐            ┌────────────┐   
                x─────┤n_rand1├──►x_rand1──┤(-n_x_rand1)├───┐
                │     └───────┘            └────────────┘   │
                ▼                                           ▼                                                                                      
               mse                                        x_hat
                ▲                                           │
                │   ┌──────────┐             ┌─────────┐    │
                x'◄─┤(-n_rand2)├──x_rand2◄───┤n_x_rand2├────┘
                    └──────────┘             └─────────┘    

            However, implementing this directly requires doing 3 affine transformations, which adds 'blur' to the image.
            So instead we note that the diagram above is equivalent to (assuming invertible group actions):


                x'◄───────────────────────────┐
                │                             │
                ▼     ┌───────────────────────┴───────────────────────┐
               mse    │(-n_rand2) @ n_x_rand2 @ (-n_x_rand1) @ n_rand1│
                ▲     └───────────────────────┬───────────────────────┘
                │                             │
                x─────────────────────────────┘

            Where @ denotes composition of group actions. In all of the above, we assume we have group action representations
            for which negation yields the inverse.
            """
            (
                rng_sample1,
                rng_sample2,
                rng_η_rand1,
                rng_η_rand2,
            ) = random.split(rng, 4)

            Η_rand = distrax.Uniform(
                # Separate model bounds and augment bounds
                low=-jnp.array(config.augment_bounds) * state.augment_bounds_mult
                + jnp.array(config.augment_offset),
                high=jnp.array(config.augment_bounds) * state.augment_bounds_mult
                + jnp.array(config.augment_offset),
            )
            η_rand1 = Η_rand.sample(seed=rng_η_rand1, sample_shape=())
            η_rand2 = Η_rand.sample(seed=rng_η_rand2, sample_shape=())

            # Get the affine matrices
            η_rand1_aff_mat = gen_affine(η_rand1)
            η_rand1_inv_aff_mat = linalg.inv(η_rand1_aff_mat)
            η_rand2_aff_mat = gen_affine(η_rand2)
            η_rand2_inv_aff_mat = linalg.inv(η_rand2_aff_mat)

            x_rand1 = transform_image_fn(x, η_rand1_aff_mat)
            x_rand2 = transform_image_fn(x, η_rand2_aff_mat)

            q_H_x_rand1 = model.apply({"params": params}, x_rand1, train)
            q_H_x_rand2 = model.apply({"params": params}, x_rand2, train)

            η_x_rand1 = q_H_x_rand1.sample(seed=rng_sample1)
            η_x_rand2 = q_H_x_rand2.sample(seed=rng_sample2)

            # Get the affine matrices
            η_x_rand1_aff_mat = gen_affine(η_x_rand1)
            η_x_rand1_inv_aff_mat = linalg.inv(η_x_rand1_aff_mat)  # Inv. faster than matrix exponential
            η_x_rand2_aff_mat = gen_affine(η_x_rand2)
            η_x_rand2_inv_aff_mat = linalg.inv(η_x_rand2_aff_mat)

            # Consider transforming x twice (first into latent space, and then untransforming) as a
            # way to regularise for invertibility
            x_mse = optax.squared_error(
                x, 
                transform_image_fn(
                    x,
                    η_rand2_inv_aff_mat @ η_x_rand2_aff_mat @ η_x_rand1_inv_aff_mat @ η_rand1_aff_mat,
                )
            ).mean()

            difficulty = optax.squared_error(x_rand1, x_rand2).mean()

            # This loss is a useful proxy and can provide a helpful learning signal.
            # However, it is possible for it to be non-0 while the MSE is perfect due to 
            # self-symmetric objects (multiple orbit stabilizers). Ultimately, we care about
            # low MSE.
            # The loss is chosen to be more-or-less linear in the difference (hence the Huber loss)
            # because we expect the error distribution to be multimodal, and we want the model
            # to settle on one of the modes. The square loss would make the model settle inbetween
            # the modes (at the mean of the modes)
            η_recon_loss = huber_loss(
                # Measure how close the transformation is to identity
                (η_rand2_inv_aff_mat @ η_x_rand2_aff_mat @ η_x_rand1_inv_aff_mat @ η_rand1_aff_mat)[:2, :].ravel(),
                jnp.eye(3, dtype=η_rand2_aff_mat.dtype)[:2, :].ravel(),
                slope=1,
                radius=1e-2,  # Choose a relatively small delta - want the loss to be mostly linear
            ).mean()

            def invertibility_loss_fn(x_, η_affine_mat, η_inv_affine_mat):
                transformed_x = transform_image_fn(
                    x_,
                    η_affine_mat,
                )
                untransformed_x = transform_image_fn(
                    transformed_x,
                    # linalg.inv(η_affine_mat),
                    η_inv_affine_mat,
                )
                # mse = (transformed_x - untransformed_x)**2
                mse = optax.squared_error(
                    untransformed_x,
                    x_,
                )
                return mse.mean()

            invertibility_loss = invertibility_loss_fn(x_rand1, η_affine_mat=η_x_rand1_inv_aff_mat, η_inv_affine_mat=η_x_rand1_aff_mat)

            return x_mse, η_recon_loss, invertibility_loss, difficulty

        per_sample_loss = symmetrised_per_sample_loss if config.symmetrised_samples_in_loss else nonsymmetrised_per_sample_loss

        rngs = random.split(rng_local, config.n_samples)
        x_mse, η_recon_loss, invertibility_loss, difficulty = jax.vmap(per_sample_loss)(rngs)

        # (maybe) do a weighted average based on the difficulty of the sample
        weights = (
            difficulty / difficulty.sum()
            if train and config.difficulty_weighted_inf_loss
            else jnp.ones((config.n_samples,)) / config.n_samples
        )
        x_mse = (x_mse * weights).sum()
        η_recon_loss = η_recon_loss.mean()

        invertibility_loss = invertibility_loss.mean()

        loss = config.x_mse_loss_mult * x_mse + state.η_loss_mult * η_recon_loss + config.invertibility_loss_mult * invertibility_loss

        return loss, {
            "loss": loss,
            "x_mse": x_mse,
            "η_recon_loss": η_recon_loss,
            "invertibility_loss": invertibility_loss,
        }


    @jax.jit
    def train_step(state, batch):
        # --- Update the model ---
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
        # --- Log the metrics
        logs = ciclo.logs()
        logs.add_stateful_metrics(**metrics.compute())
        logs.add_entry(
            "schedules",
            "lr_inf",
            state.opt_state.inner_states["inference"][0].hyperparams["learning_rate"],
        )
        logs.add_entry(
            "schedules",
            "lr_σ",
            state.opt_state.inner_states["σ"][0].hyperparams["learning_rate"],
        )
        logs.add_entry(
            "schedules",
            "η_loss_mult",
            state.η_loss_mult,
        )
        logs.add_entry(
            "schedules",
            "augment_bounds_mult",
            state.augment_bounds_mult,
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
            the canonicalization model, where the pair of images is chosen to have the same label
            (and hence, for some datasets, likely a similar canonicalized form).

            The "transforming into each other" happens by first transforming the original image
            into the canonical form, and then transforming the canonical form into the paired image
            by using (inverse of) the canonicalization transform of the paired image.
            """
            sample_rng, pair_match_rng = random.split(rng, 2)

            # Create a mask that excludes self-matching images
            num_images = xs.shape[0]
            self_match_mask = jnp.eye(num_images, dtype=bool)  # [batch_size, batch_size]

            # Create a mask that matches labels between original and transformed images
            label_match_mask = jnp.equal(jnp.expand_dims(labels, axis=-1), labels)  # [batch_size, batch_size]
            mask_mask  = jnp.expand_dims(mask, axis=0)  # [1, batch_size]

            match_mask = jnp.logical_and(label_match_mask, ~self_match_mask)
            match_mask = jnp.logical_and(match_mask, mask_mask)

            # Randomly select an image from the transformed images that matches the label of the original image
            @jax.jit
            def get_matching_index(is_matching, rng) -> tuple:
                num_idxs = is_matching.shape[0]
                any_matching = is_matching.any()
                return jax.lax.cond(
                    any_matching,
                    lambda: (random.choice(rng, num_idxs, p=is_matching/is_matching.sum()), True),
                    lambda: (0, False),
                )

            pair_match_rngs = random.split(pair_match_rng, num_images)
            paired_idxs, has_matching = jax.vmap(get_matching_index, in_axes=(0, 0))(match_mask, pair_match_rngs)
            paired_xs = jax.vmap(lambda idx, xs_: xs_[idx], in_axes=(0, None))(paired_idxs, xs)

            # Now transform the original image using the eta from the transformed image
            def per_example_paired_image_mse(x1, x2, rng):
                rng_sample1, rng_sample2 = random.split(rng, 2)
                p_η_x1 = model.apply({"params": state.params}, x1, False)
                η1 = p_η_x1.sample(seed=rng_sample1)
                p_η_x2 = model.apply({"params": state.params}, x2, False)
                η2 = p_η_x2.sample(seed=rng_sample2)
                x1_hat = transform_image(x1, -η1, order=config.interpolation_order)
                x2_recon = transform_image(x1_hat, η2, order=config.interpolation_order)
                return optax.squared_error(x2, x2_recon).mean()
            
            # Return MSE of 0 if there is no label match
            return jax.vmap(per_example_paired_image_mse, in_axes=(0, 0, 0))(xs, paired_xs, random.split(sample_rng, num_images)), has_matching

        label_paired_image_mse, has_matching = mse_same_label_examples(batch["image"][0], batch["label"][0], batch["mask"][0], step_rng)
        mean_label_paired_image_mse = (label_paired_image_mse * has_matching).sum() / has_matching.sum()

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


def create_canonicalizer_optimizer(params, config):
    partition_optimizers = {
        "inference": optax.inject_hyperparams(clipped_adamw)(
            optax.warmup_cosine_decay_schedule(
                config.inf_init_lr_mult * config.inf_lr,
                config.inf_lr,
                config.inf_warmup_steps,
                config.inf_steps,
                config.inf_lr * config.inf_final_lr_mult,
            ),
            2.0,
            # Optax WD default: 0.0001 https://optax.readthedocs.io/en/latest/api.html#optax.adamw
            config.get("inf_weight_decay", 0.0001),  
        ),
        "σ": optax.inject_hyperparams(optax.adam)(
            optax.warmup_cosine_decay_schedule(
                config.σ_lr,
                config.σ_lr * 3,
                config.inf_warmup_steps,
                config.inf_steps,
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
class CanonincalizerMetrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    x_mse: metrics.Average.from_output("x_mse")
    η_recon_loss: metrics.Average.from_output("η_recon_loss")
    invertibility_loss: metrics.Average.from_output("invertibility_loss")

    def update(self, **kwargs) -> "CanonincalizerMetrics":
        updates = self.single_from_model_output(**kwargs)
        return self.merge(updates)


@flax.struct.dataclass
class AugmentGenerativeMetrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    p_η_x_hat: metrics.Average.from_output("p_η_x_hat")

    def update(self, **kwargs) -> "AugmentGenerativeMetrics":
        updates = self.single_from_model_output(**kwargs)
        return self.merge(updates)


class TrainStateWithMetrics(train_state.TrainState):
    metrics: CanonincalizerMetrics | AugmentGenerativeMetrics
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

class CanonicalizerTrainState(TrainStateWithMetrics):
    metrics: CanonincalizerMetrics
    augment_bounds_mult: float
    augment_bounds_mult_schedule: optax.Schedule = flax.struct.field(pytree_node=False)
    η_loss_mult: float
    η_loss_mult_schedule: optax.Schedule = flax.struct.field(pytree_node=False)

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            augment_bounds_mult=self.augment_bounds_mult_schedule(self.step),
            η_loss_mult=self.η_loss_mult_schedule(self.step),
            **kwargs,
        )
    
    @classmethod
    def create(cls, *, apply_fn, params, tx, augment_bounds_mult_schedule, η_loss_mult_schedule, **kwargs):
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            augment_bounds_mult_schedule=augment_bounds_mult_schedule,
            augment_bounds_mult=augment_bounds_mult_schedule(0),
            η_loss_mult_schedule=η_loss_mult_schedule,
            η_loss_mult=η_loss_mult_schedule(0),
            **kwargs,
        )


def create_canonicalizer_state(params, rng, config):
    opt = create_canonicalizer_optimizer(params, config)
    return CanonicalizerTrainState.create(
        apply_fn=TransformationInferenceNet.apply,
        params=params,
        tx=opt,
        metrics=CanonincalizerMetrics.empty(),
        augment_bounds_mult_schedule=optax.join_schedules(
            [
                optax.linear_schedule(init_value=0.0, end_value=1.0, transition_steps=config.inf_steps *config.augment_warmup_end),
                optax.constant_schedule(1.0),
            ],
            boundaries=[config.inf_steps * config.augment_warmup_end],
        ),
        η_loss_mult_schedule=optax.join_schedules(
            [
                optax.constant_schedule(config.η_loss_mult_peak),
                optax.linear_schedule(init_value=config.η_loss_mult_peak, end_value=0.0, transition_steps=(config.η_loss_decay_end - config.η_loss_decay_start) * config.inf_steps),
                optax.constant_schedule(0.0),
            ],
            boundaries=[config.η_loss_decay_start * config.inf_steps, config.η_loss_decay_end * config.inf_steps],
        ),
        rng=rng,
    )