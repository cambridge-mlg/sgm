from typing import Any, Mapping, Optional, Tuple, Union
from functools import partial

import wandb
from tqdm.auto import trange
import numpy as np
import tensorflow as tf

import jax
from jax import random
from jax import numpy as jnp

import flax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
import flax.linen as nn

import optax

from chex import Array

from ml_collections import config_dict
from clu import parameter_overview
from clu import preprocess_spec
from absl import logging

import src.utils.input as input_utils
import src.utils.preprocess as preprocess_utils
import src.utils.plotting as plot_utils
import src.models as models


PRNGKey = Any
ScalarOrSchedule = Union[float, optax.Schedule]


def _write_note(note: str):
    if jax.process_index() == 0:
        logging.info(note)


def _tree_concatenate(list_of_trees):
    """Convert a list of trees of identical structure into a single tree of lists."""
    return jax.tree_map(lambda *xs: jnp.array(list(xs)), *list_of_trees)


class TrainState(train_state.TrainState):
    """A Flax TrainState which also tracks model state (e.g. BatchNorm running averages) and schedules β/α."""

    model_state: FrozenDict
    β: float
    β_val_or_schedule: ScalarOrSchedule = flax.struct.field(pytree_node=False)
    α: float
    α_val_or_schedule: ScalarOrSchedule = flax.struct.field(pytree_node=False)

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            β=_get_value_for_step(self.step, self.β_val_or_schedule),
            α=_get_value_for_step(self.step, self.α_val_or_schedule),
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, β_val_or_schedule, α_val_or_schedule, **kwargs):
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            β_val_or_schedule=β_val_or_schedule,
            β=_get_value_for_step(0, β_val_or_schedule),
            α_val_or_schedule=α_val_or_schedule,
            α=_get_value_for_step(0, α_val_or_schedule),
            **kwargs,
        )


# Helper which helps us deal with the fact that we can either specify a fixed β/α
# or a schedule for adjusting β/α. This is a pattern similar to the one used by
# optax for dealing with LRs either being specified as a constant or schedule.
def _get_value_for_step(step, val_or_schedule):
    if callable(val_or_schedule):
        return val_or_schedule(step)
    else:
        return val_or_schedule


def setup_model(
    config: config_dict.ConfigDict,
    rng: PRNGKey,
    train_ds: tf.data.Dataset,
) -> Tuple[nn.Module, TrainState]:
    """Helper which returns the model object and the corresponding initialised train state for a given config."""
    _write_note("Initializing model...")
    _write_note(f"config.model_name = {config.model_name}")
    _write_note(f"config.model = {config.model}")

    input_size = tuple(train_ds.element_spec["image"].shape[2:])
    _write_note(f"input_size = {input_size}")

    model_cls = getattr(models, config.model_name)  # type: ignore
    model = model_cls(image_shape=input_size, **config.model.to_dict())  # type: ignore

    _write_note("Initializing model...")

    @partial(jax.jit, backend="cpu")
    def init(rng: PRNGKey) -> nn.FrozenDict:
        dummy_input = jnp.zeros(input_size, jnp.float32)

        fwd_rng, init_rng = jax.random.split(rng)

        variables = model.init(init_rng, dummy_input, fwd_rng)
        return variables

    rng, rng_init = jax.random.split(rng)
    variables_cpu = init(rng_init)

    if jax.process_index() == 0:
        # num_params = sum(p.size for p in jax.tree_flatten(params_cpu)[0])
        parameter_overview.log_parameter_overview(variables_cpu)
        # writer.write_scalars(step=0, scalars={'num_params': num_params})

    model_state, params = variables_cpu.pop("params")
    del variables_cpu

    if config.get("lr_schedule_name", None):
        schedule = getattr(optax, config.lr_schedule_name)  # type: ignore
        lr = schedule(init_value=config.learning_rate, **config.lr_schedule.to_dict())  # type: ignore
    else:
        lr = config.learning_rate

    optim = getattr(optax, config.optim_name)  # type: ignore
    optim = optax.inject_hyperparams(optim)
    # This ^ allows us to access the lr as opt_state.hyperparams['learning_rate'].

    if config.get("β_schedule_name", None):
        schedule = getattr(optax, config.β_schedule_name)  # type: ignore
        β = schedule(init_value=config.β, **config.β_schedule.to_dict())  # type: ignore
    else:
        β = config.β

    if config.get("α_schedule_name", None):
        schedule = getattr(optax, config.α_schedule_name)  # type: ignore
        α = schedule(init_value=config.α, **config.α_schedule.to_dict())  # type: ignore
    else:
        α = config.α

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optim(learning_rate=lr, **config.optim.to_dict()),  # type: ignore
        model_state=model_state,
        β_val_or_schedule=β,
        α_val_or_schedule=α,
    )

    return model, state


def get_dataset_splits(
    config: config_dict.ConfigDict, rng: PRNGKey, local_batch_size: int, local_batch_size_eval: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset]]:
    rng, train_ds_rng = jax.random.split(rng)
    train_ds_rng = jax.random.fold_in(train_ds_rng, jax.process_index())

    _write_note("Initializing train dataset...")
    train_ds = input_utils.get_data(
        dataset=config.dataset,
        split=config.train_split,  # type: ignore
        rng=train_ds_rng,
        process_batch_size=local_batch_size,
        preprocess_fn=preprocess_spec.parse(
            spec=config.pp_train, available_ops=preprocess_utils.all_ops()  # type: ignore
        ),
        shuffle_buffer_size=config.shuffle_buffer_size,  # type: ignore
        repeat_after_batching=config.get("repeat_after_batching", True),  # type: ignore
        prefetch_size=config.get("prefetch_to_host", 2),  # type: ignore
        drop_remainder=False,
        data_dir=config.get("data_dir"),  # type: ignore
    )

    _write_note("Initializing val dataset...")
    rng, val_ds_rng = jax.random.split(rng)
    val_ds_rng = jax.random.fold_in(val_ds_rng, jax.process_index())

    nval_img = input_utils.get_num_examples(
        config.dataset,
        split=config.val_split,  # type: ignore
        process_batch_size=local_batch_size_eval,
        drop_remainder=False,
        data_dir=config.get("data_dir"),  # type: ignore
    )
    val_steps = int(np.ceil(nval_img / local_batch_size_eval))
    logging.info(
        f"Running validation for {val_steps} steps for {config.dataset}, {config.val_split}"
    )

    val_ds = input_utils.get_data(
        dataset=config.dataset,
        split=config.val_split,  # type: ignore
        rng=val_ds_rng,
        process_batch_size=local_batch_size_eval,
        preprocess_fn=preprocess_spec.parse(
            spec=config.pp_eval, available_ops=preprocess_utils.all_ops()  # type: ignore
        ),
        cache=config.get("val_cache", "batched"),  # type: ignore
        num_epochs=1,
        repeat_after_batching=True,
        shuffle=False,
        prefetch_size=config.get("prefetch_to_host", 2),  # type: ignore
        drop_remainder=False,
        data_dir=config.get("data_dir"),  # type: ignore
    )

    rng, test_ds_rng = jax.random.split(rng)
    test_ds_rng = jax.random.fold_in(test_ds_rng, jax.process_index())

    test_ds = None
    if config.get("test_split", None):
        _write_note("Initializing test dataset...")
        ntest_img = input_utils.get_num_examples(
            config.dataset,
            split=config.test_split,  # type: ignore
            process_batch_size=local_batch_size_eval,
            drop_remainder=False,
            data_dir=config.get("data_dir"),  # type: ignore
        )
        test_steps = int(np.ceil(ntest_img / local_batch_size_eval))
        logging.info(
            f"Running test for {test_steps} steps for {config.dataset}, {config.test_split}"
        )

        test_ds = input_utils.get_data(
            dataset=config.dataset,
            split=config.test_split,  # type: ignore
            rng=test_ds_rng,
            process_batch_size=local_batch_size_eval,
            preprocess_fn=preprocess_spec.parse(
                spec=config.pp_eval, available_ops=preprocess_utils.all_ops()  # type: ignore
            ),
            cache=config.get("test_cache", "batched"),  # type: ignore
            num_epochs=1,
            repeat_after_batching=True,
            shuffle=False,
            prefetch_size=config.get("prefetch_to_host", 2),  # type: ignore
            drop_remainder=False,
            data_dir=config.get("data_dir"),  # type: ignore
        )

    return train_ds, val_ds, test_ds


def train_loop(
    config: config_dict.ConfigDict,
    model: nn.Module,
    state: TrainState,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: Optional[tf.data.Dataset] = None,
    wandb_kwargs: Optional[Mapping] = None,
) -> TrainState:
    """Runs the training loop!"""
    wandb_kwargs = {
        "project": "learning-invariances",
        "entity": "invariance-learners",
        "notes": "",
        "config": config.to_dict(),
    } | (wandb_kwargs or {})
    # ^ whatever the user specifies takes priority.

    with wandb.init(**wandb_kwargs) as run:  # type: ignore
        seed = config.get("seed", 0)
        rng = jax.random.PRNGKey(seed)  # type: ignore
        rng, visualisation_rng = jax.random.split(rng)
        tf.random.set_seed(seed)

        _write_note("Setting up datasets...")

        batch_size = config.batch_size
        batch_size_eval = config.get("batch_size_eval", batch_size)
        if batch_size % jax.device_count() != 0 or batch_size_eval % jax.device_count() != 0:  # type: ignore
            raise ValueError(
                f"Batch sizes ({batch_size} and {batch_size_eval}) must "
                f"be divisible by device number ({jax.device_count()})"
            )

        local_batch_size = batch_size // jax.process_count()  # type: ignore
        local_batch_size_eval = batch_size_eval // jax.process_count()  # type: ignore
        _write_note(
            "Global batch size %d on %d hosts results in %d local batch size. "
            "With %d devices per host (%d devices total), that's a %d per-device "
            "batch size."
            % (
                batch_size,
                jax.process_count(),
                local_batch_size,
                jax.local_device_count(),
                jax.device_count(),
                local_batch_size // jax.local_device_count(),
            )
        )

        train_ds, val_ds, test_ds = get_dataset_splits(
            config, rng, local_batch_size, local_batch_size_eval
        )

        ntrain_img = input_utils.get_num_examples(
            config.dataset,
            split=config.train_split,  # type: ignore
            process_batch_size=local_batch_size,
            data_dir=config.get("data_dir"),  # type: ignore
            drop_remainder=False,
        )
        steps_per_epoch = ntrain_img // batch_size  # type: ignore

        if config.get("num_epochs"):
            assert not config.get("total_steps"), "Set either num_epochs or total_steps"
            total_steps = int(config.num_epochs * steps_per_epoch)
        else:
            total_steps = config.total_steps

        _write_note(f"Total train data points: {ntrain_img}")
        _write_note(
            f"Running for {total_steps} steps, that means {total_steps * batch_size / ntrain_img}"  # type: ignore
            f" epochs and {steps_per_epoch} steps per epoch."
        )

        @partial(jax.pmap, axis_name="device", in_axes=(None, 0, None), out_axes=None)  # type: ignore
        def update_fn(state, x_batch, rng):
            rng_local = jax.random.fold_in(rng, jax.lax.axis_index("device"))  # type: ignore

            @jax.jit
            def batch_loss(params, x_batch, rng, β, α):
                # Broadcast loss over batch and aggregate.
                loss, metrics = jax.vmap(
                    models.livae_loss_fn, in_axes=(None, None, 0, None, None, None), axis_name="batch"  # type: ignore
                )(model, params, x_batch, rng, β, α)
                loss, metrics = jax.tree_util.tree_map(partial(jnp.mean, axis=0), (loss, metrics))
                # TODO: replace this tree_map with a pmean call.
                return loss, metrics

            grad_fn = jax.value_and_grad(batch_loss, has_aux=True)
            (loss, metrics), grad = grad_fn(state.params, x_batch, rng_local, state.β, state.α)
            grad, loss, metrics = jax.lax.pmean((grad, loss, metrics), axis_name="device")

            # Update the state.
            new_state = state.apply_gradients(grads=grad)

            return new_state, loss, metrics

        @partial(jax.pmap, axis_name="device", in_axes=(None, 0, 0, None), out_axes=None)  # type: ignore
        def eval_fn(state, x_batch, mask, rng):
            rng_local = jax.random.fold_in(rng, jax.lax.axis_index("device"))  # type: ignore

            @jax.jit
            def batch_loss(params, x_batch, mask, rng, β, α):
                # Broadcast loss over batch and aggregate.
                loss, metrics = jax.vmap(
                    models.livae_loss_fn, in_axes=(None, None, 0, None, None, None), axis_name="batch"  # type: ignore
                )(model, params, x_batch, rng, β, α)
                loss, metrics, mask = jax.tree_util.tree_map(
                    partial(jnp.sum, axis=0), (loss, metrics, mask)
                )
                # TODO: replace this tree_map with a psum call.
                return loss, metrics, mask

            loss, metrics, mask = batch_loss(
                state.params, x_batch, mask, rng_local, state.β, state.α
            )
            loss, metrics, n_examples = jax.lax.psum((loss, metrics, mask), axis_name="device")

            return loss, metrics, n_examples

        train_iter = input_utils.start_input_pipeline(train_ds, config.get("prefetch_to_device", 1))

        _write_note("Starting training loop...")

        best_val_loss = jnp.inf

        for step in (steps := trange(total_steps)):  # type: ignore
            train_batch = next(train_iter)
            rng, train_rng, val_rng, test_rng = jax.random.split(rng, 4)
            state, loss, metrics = update_fn(state, train_batch["image"], train_rng)

            if step % config.get("log_every", 1) == 0:  # type: ignore
                steps.set_postfix_str(f"Loss: {loss:.4f}")
                learning_rate = state.opt_state.hyperparams["learning_rate"]
                run.log(
                    {"train/loss": loss, "β": state.β, "α": state.α, "learing_rate": learning_rate},
                    step=step,
                )
                for k, v in metrics.items():
                    run.log({f"train/{k}": v}, step=step)

            if step % config.get("eval_every", 1000) == 0:  # type: ignore
                val_iter = input_utils.start_input_pipeline(
                    val_ds, config.get("prefetch_to_device", 1)
                )
                batch_losses = []
                batch_metrics = []
                n_val = 0
                val_batch_0 = None
                for i, val_batch in enumerate(val_iter):
                    loss, metrics, n_examples = eval_fn(
                        state, val_batch["image"], val_batch["mask"], val_rng
                    )

                    batch_losses.append(loss)
                    batch_metrics.append(metrics)
                    n_val += n_examples

                    if i == 0:
                        val_batch_0 = val_batch

                batch_metrics = _tree_concatenate(batch_metrics)
                val_loss, val_metrics = jax.tree_util.tree_map(
                    lambda x: jnp.sum(x) / n_val, (jnp.array(batch_losses), batch_metrics)  # type: ignore
                )

                run.log({"val/loss": val_loss}, step=step)
                for k, v in val_metrics.items():
                    run.log({f"val/{k}": v}, step=step)

                # do some reconstruction visualizations
                n_visualize = config.get("n_visualize", 24)

                def make_reconstruction_plot(x):
                    @partial(
                        jax.jit,
                        static_argnames=("prototype", "sample_z", "sample_xhat", "sample_η"),
                    )
                    def reconstruct(
                        x, prototype=False, sample_z=False, sample_xhat=False, sample_η=False
                    ):
                        rng = random.fold_in(visualisation_rng, jax.lax.axis_index("image"))  # type: ignore
                        return model.apply(
                            {"params": state.params},
                            x,
                            rng,
                            prototype=prototype,
                            sample_z=sample_z,
                            sample_xhat=sample_xhat,
                            sample_η=sample_η,
                            α=state.α,
                            method=model.reconstruct,
                        )

                    x_proto_modes = jax.vmap(
                        reconstruct, axis_name="image", in_axes=(0, None, None, None, None)  # type: ignore
                    )(x, True, False, False, True)

                    x_recon_modes = jax.vmap(
                        reconstruct, axis_name="image", in_axes=(0, None, None, None, None)  # type: ignore
                    )(x, False, False, False, True)

                    recon_fig = plot_utils.plot_img_array(
                        jnp.concatenate((x, x_proto_modes, x_recon_modes), axis=0),
                        ncol=n_visualize,  # type: ignore
                        pad_value=1,
                        padding=1,
                        title="Original | Prototype | Reconstruction",
                    )

                    return recon_fig

                val_x = val_batch_0["image"][0, :n_visualize]  # type: ignore
                val_recon_fig = make_reconstruction_plot(val_x)

                # do some sampling visualizations
                def make_sampling_plot():
                    @partial(jax.jit, static_argnames=("prototype", "sample_xhat", "sample_η"))
                    def sample(rng, prototype=False, sample_xhat=False, sample_η=False):
                        return model.apply(
                            {"params": state.params},
                            rng,
                            prototype=prototype,
                            sample_xhat=sample_xhat,
                            sample_η=sample_η,
                            method=model.sample,
                            α=state.α,
                        )

                    sampled_protos = jax.vmap(sample, in_axes=(0, None, None, None))(  # type: ignore
                        jax.random.split(visualisation_rng, n_visualize), True, True, True  # type: ignore
                    )

                    sampled_data = jax.vmap(sample, in_axes=(0, None, None, None))(  # type: ignore
                        jax.random.split(visualisation_rng, n_visualize), False, True, True  # type: ignore
                    )

                    sample_fig = plot_utils.plot_img_array(
                        jnp.concatenate((sampled_protos, sampled_data), axis=0),
                        ncol=n_visualize,  # type: ignore
                        pad_value=1,
                        padding=1,
                        title="Sampled prototypes | Sampled data",
                    )
                    return sample_fig

                sample_fig = make_sampling_plot()

                run.log({"val_reconstructions": wandb.Image(val_recon_fig)}, step=step)
                run.log({"prior_samples": wandb.Image(sample_fig)}, step=step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    run.summary["best_val_loss"] = best_val_loss
                    best_val_metrics = val_metrics
                    for k, v in best_val_metrics.items():
                        run.summary[f"best_val_{k}"] = v
                    best_val_step = step
                    run.summary["best_val_step"] = best_val_step
                    run.summary["best_prior_samples"] = wandb.Image(sample_fig)

                    if test_ds:
                        test_iter = input_utils.start_input_pipeline(
                            test_ds, config.get("prefetch_to_device", 1)
                        )

                        batch_losses = []
                        batch_metrics = []
                        n_test = 0
                        for test_batch in test_iter:
                            loss, metrics, n_examples = eval_fn(
                                state, test_batch["image"], test_batch["mask"], test_rng
                            )

                            batch_losses.append(loss)
                            batch_metrics.append(metrics)
                            n_test += n_examples

                        batch_metrics = _tree_concatenate(batch_metrics)
                        test_loss, test_metrics = jax.tree_util.tree_map(
                            lambda x: jnp.sum(x) / n_test, (jnp.array(batch_losses), batch_metrics)  # type: ignore
                        )

                        run.summary["test_loss"] = test_loss
                        for k, v in test_metrics.items():
                            run.summary[f"test_{k}"] = v

                        test_x = test_batch["image"][:n_visualize]  # type: ignore
                        test_recon_fig = make_reconstruction_plot(test_x)

                        run.summary["test_reconstructions"] = wandb.Image(test_recon_fig)

        _write_note("Training finished.")

    return state
