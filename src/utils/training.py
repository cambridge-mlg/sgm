from typing import Any, Callable, Mapping, Optional, Tuple, Union
from functools import partial

import wandb
from tqdm.auto import trange
import numpy as np
import tensorflow as tf

import jax
from jax import random
from jax import numpy as jnp

import flax
from flax.core.frozen_dict import FrozenDict, freeze
from flax import traverse_util
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
    """A Flax TrainState which also tracks model state (e.g. BatchNorm running averages) and schedules α, β, γ."""

    model_state: FrozenDict
    α: float
    α_val_or_schedule: ScalarOrSchedule = flax.struct.field(pytree_node=False)
    β: float
    β_val_or_schedule: ScalarOrSchedule = flax.struct.field(pytree_node=False)
    γ: float
    γ_val_or_schedule: ScalarOrSchedule = flax.struct.field(pytree_node=False)

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            α=_get_value_for_step(self.step, self.α_val_or_schedule),
            β=_get_value_for_step(self.step, self.β_val_or_schedule),
            γ=_get_value_for_step(self.step, self.γ_val_or_schedule),
            **kwargs,
        )

    @classmethod
    def create(
        cls,
        *,
        apply_fn,
        params,
        tx,
        α_val_or_schedule,
        β_val_or_schedule,
        γ_val_or_schedule,
        **kwargs,
    ):
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            α_val_or_schedule=α_val_or_schedule,
            α=_get_value_for_step(0, α_val_or_schedule),
            β_val_or_schedule=β_val_or_schedule,
            β=_get_value_for_step(0, β_val_or_schedule),
            γ_val_or_schedule=γ_val_or_schedule,
            γ=_get_value_for_step(0, γ_val_or_schedule),
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


def _make_split_schedule(split_step: Optional[int]):
    def _split_schedule(schedule, schedule_half, const_value=0.0):
        if schedule_half == "first":
            return optax.join_schedules(
                schedules=[
                    schedule,
                    optax.constant_schedule(const_value),
                ],
                boundaries=[split_step],
            )
        elif schedule_half == "second":
            return optax.join_schedules(
                schedules=[
                    optax.constant_schedule(const_value),
                    schedule,
                ],
                boundaries=[split_step],
            )
        return schedule

    if split_step is not None:
        return _split_schedule
    else:
        return lambda schedule, *args, **kwargs: schedule


# the empty node is a struct.dataclass to be compatible with JAX.
@traverse_util.struct.dataclass
class _EmptyNode:
    pass

empty_node = _EmptyNode()


def _path_aware_map(
    f: Callable[[Tuple[str, ...], Any], Any], nested_dict: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Mapping[str, Any]]:
    """A map function that operates over nested dictionary structures while taking
    the path to each leaf into account.

    Example::

      >>> import jax.numpy as jnp
      >>> from flax import traverse_util
      ...
      >>> params = {'a': {'x': 10, 'y': 3}, 'b': {'x': 20}}
      >>> f = lambda path, x: x + 5 if 'x' in path else -x
      >>> traverse_util.path_aware_map(f, params)
      {'a': {'x': 15, 'y': -3}, 'b': {'x': 25}}

    Args:
      f: A callable that takes in ``(path, value)`` arguments and maps them
        to a new value. Here ``path`` is a tuple of strings.
      nested_dict: A nested dictionary structure.

    Returns:
      A new nested dictionary structure with the mapped values.
    """
    flat = traverse_util.flatten_dict(nested_dict, keep_empty_nodes=True)
    return traverse_util.unflatten_dict(
        {k: f(k, v) if v is not empty_node else v for k, v in flat.items()}
    )


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

        variables = model.init(init_rng, dummy_input, fwd_rng, train=False)
        return variables

    rng, rng_init = jax.random.split(rng)
    variables_cpu = init(rng_init)

    if jax.process_index() == 0:
        # num_params = sum(p.size for p in jax.tree_flatten(params_cpu)[0])
        parameter_overview.log_parameter_overview(variables_cpu)
        # writer.write_scalars(step=0, scalars={'num_params': num_params})

    model_state, params = variables_cpu.pop("params")
    del variables_cpu

    reset_step = config.get("reset_step", None)
    maybe_split_schedule = _make_split_schedule(reset_step)
    # check if config.optim_name is a single string, and if so create a single optimizer
    if isinstance(config.optim_name, str):
        optim = getattr(optax, config.optim_name)  # type: ignore
        optim = optax.inject_hyperparams(optim)
        # This ^ allows us to access the lr as opt_state.hyperparams['learning_rate'].

        if config.get("lr_schedule_name", None):
            schedule = getattr(optax, config.lr_schedule_name)  # type: ignore
            lr = schedule(init_value=config.learning_rate, **config.lr_schedule.to_dict())  # type: ignore
        else:
            lr = config.learning_rate

        tx = optim(learning_rate=lr, **config.optim.to_dict())
    # otherwise we assume it is a tuple of strings and create a multi optimizer
    else:
        assert len(config.partition_names) == 2
        assert len(config.optim_name) == 2
        assert len(config.optim) == 2
        assert len(config.learning_rate) == 2
        schedule_names = config.get("lr_schedule_name", ("constant_schedule", "constant_schedule"))
        assert len(schedule_names) == 2
        lr_schedules = config.get(
            "lr_schedule", (config_dict.ConfigDict(), config_dict.ConfigDict())
        )
        assert len(lr_schedules) == 2
        schedule_halfs = config.get("lr_schedule_halfs", ("first", "second"))
        assert len(schedule_halfs) == 2

        partition_optimizers = {}
        for (
            parition_name,
            learning_rate,
            schedule_name,
            lr_schedule,
            schedule_half,
            optim_name,
            optim_cfg,
        ) in zip(
            config.partition_names,
            config.learning_rate,
            schedule_names,
            lr_schedules,
            schedule_halfs,
            config.optim_name,
            config.optim,
        ):
            optim = getattr(optax, optim_name)  # type: ignore
            optim = optax.inject_hyperparams(optim)

            schedule = getattr(optax, schedule_name)
            lr = schedule(init_value=learning_rate, **lr_schedule.to_dict())
            lr = maybe_split_schedule(lr, schedule_half, 0.0)
            partition_optimizers[parition_name] = optim(learning_rate=lr, **optim_cfg.to_dict())

        parition_fn = config.partition_fn
        param_partitions = freeze(_path_aware_map(parition_fn, params))

        tx = optax.multi_transform(
            partition_optimizers,
            param_partitions,
        )

    if config.get("α_schedule_name", None):
        schedule = getattr(optax, config.α_schedule_name)  # type: ignore
        α = schedule(init_value=config.α, **config.α_schedule.to_dict())  # type: ignore
        α = maybe_split_schedule(α, config.get("α_schedule_half", None), 1.0)
    else:
        α = config.α

    if config.get("β_schedule_name", None):
        schedule = getattr(optax, config.β_schedule_name)  # type: ignore
        β = schedule(init_value=config.β, **config.β_schedule.to_dict())  # type: ignore
        β = maybe_split_schedule(β, config.get("β_schedule_half", None), 1.0)
    else:
        β = config.β

    if config.get("γ_schedule_name", None):
        schedule = getattr(optax, config.γ_schedule_name)  # type: ignore
        γ = schedule(init_value=config.γ, **config.γ_schedule.to_dict())  # type: ignore
        γ = maybe_split_schedule(γ, config.get("γ_schedule_half", None), 1.0)
    else:
        γ = config.γ

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,  # type: ignore
        model_state=model_state,
        α_val_or_schedule=α,
        β_val_or_schedule=β,
        γ_val_or_schedule=γ,
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
        "project": "neurips2023",
        "entity": "invariance-learners",
        "notes": "",
        "config": config.to_dict(),
    } | (wandb_kwargs or {})
    # ^ whatever the user specifies takes priority.

    with wandb.init(**wandb_kwargs) as run:  # type: ignore
        seed = config.get("seed", 0)
        rng = jax.random.PRNGKey(seed)  # type: ignore
        rng, visualisation_rng, summary_rng = jax.random.split(rng, 3)
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

        make_batch_loss = getattr(models, "make_" + config.model_name.lower() + "_batch_loss")

        @partial(jax.pmap, axis_name="device", in_axes=(None, 0, 0, None), out_axes=None)  # type: ignore
        def update_fn(state, x_batch, mask, rng):
            rng_local = jax.random.fold_in(rng, jax.lax.axis_index("device"))  # type: ignore

            batch_loss = make_batch_loss(model, jnp.mean, True)

            grad_fn = jax.value_and_grad(batch_loss, has_aux=True)
            (loss, (metrics, _)), grad = grad_fn(state.params, x_batch, mask, rng_local, state)
            grad, loss, metrics = jax.lax.pmean((grad, loss, metrics), axis_name="device")

            # Update the state.
            new_state = state.apply_gradients(grads=grad)

            return new_state, loss, metrics

        @partial(jax.pmap, axis_name="device", in_axes=(None, 0, 0, None), out_axes=None)  # type: ignore
        def eval_fn(state, x_batch, mask, rng):
            rng_local = jax.random.fold_in(rng, jax.lax.axis_index("device"))  # type: ignore

            batch_loss = make_batch_loss(model, jnp.sum, False)

            loss, (metrics, mask) = batch_loss(state.params, x_batch, mask, rng_local, state)
            loss, metrics, n_examples = jax.lax.psum((loss, metrics, mask), axis_name="device")

            return loss, metrics, n_examples

        train_iter = input_utils.start_input_pipeline(train_ds, config.get("prefetch_to_device", 1))

        _write_note("Starting training loop...")

        best_val_loss = jnp.inf
        best_state = None
        val_loss = jnp.nan
        val_ll = jnp.nan

        for step in (steps := trange(total_steps)):  # type: ignore
            train_batch = next(train_iter)
            rng, train_rng, val_rng, test_rng = jax.random.split(rng, 4)
            state, loss, metrics = update_fn(
                state, train_batch["image"], train_batch["mask"], train_rng
            )

            if step % config.get("log_every", 1) == 0:  # type: ignore
                if not isinstance(state.opt_state, optax.MultiTransformState):
                    learning_rate = state.opt_state.hyperparams["learning_rate"]
                    extra_lr_logs = {}
                else:
                    lr1 = state.opt_state.inner_states[
                        config.partition_names[0]
                    ].inner_state.hyperparams["learning_rate"]
                    lr2 = state.opt_state.inner_states[
                        config.partition_names[1]
                    ].inner_state.hyperparams["learning_rate"]
                    learning_rate = lr1 + lr2
                    extra_lr_logs = {"lr1": lr1, "lr2": lr2}

                make_σ = lambda σ_: jax.nn.softplus(σ_).clip(min=model.σ_min).mean()
                if config.model_name == "SSIL":
                    σ_logs = {"σ": make_σ(state.params["σ_"])}
                elif config.model_name == "VAE":
                    σ_logs = {"σ_vae": make_σ(state.params["p_X_given_Z"]["σ_"])}
                elif config.model_name == "SSILVAE":
                    σ_ = state.params["σ_"]
                    σ_vae_ = state.params["p_Xhat_given_Z"]["σ_"]
                    σ_logs = {"σ": make_σ(σ_), "σ_vae": make_σ(σ_vae_)}
                else:
                    σ_logs = {}
                run.log(
                    {
                        "train/loss": loss,
                        "α": state.α,
                        "β": state.β,
                        "γ": state.γ,
                        "learing_rate": learning_rate,
                        **extra_lr_logs,
                        **σ_logs,
                    },
                    step=step,
                )
                for k, v in metrics.items():
                    run.log({f"train/{k}": v}, step=step)
                steps.set_postfix_str(
                    (
                        f"Trn Loss {loss:.4g},\t Trn LL {metrics['ll']:.4g},\t "
                        f"Val Loss {val_loss:.4g},\t Val LL {val_ll:.4g}"
                    )
                )

            val_batch_0 = None
            if step % config.get("eval_every", 1000) == 0:  # type: ignore
                val_iter = input_utils.start_input_pipeline(
                    val_ds, config.get("prefetch_to_device", 1)
                )
                batch_losses = []
                batch_metrics = []
                n_val = 0
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
                val_ll = val_metrics["ll"]

                # do some reconstruction visualizations
                val_labels = val_batch_0["label"][0]
                class_labels = jnp.arange(10)

                # get the first two labels for each class
                def get_first_two_idxs_per_class(all_labels, class_labels):
                    def _get_first_two_idxs(label, labels):
                        idxs = jnp.where(labels == label, size=2, fill_value=-1)[0]
                        return idxs

                    # get the first two labels for each class (0-9)
                    idxs = jax.vmap(_get_first_two_idxs, in_axes=(0, None))(
                        class_labels, all_labels
                    )
                    idxs = jnp.concatenate(idxs)

                    # replace -1 indices with the last n indices
                    replace_idxs = jnp.where(idxs == -1)[0]
                    n = len(replace_idxs)
                    last_n_idxs = jnp.arange(len(all_labels) - n, len(all_labels))
                    idxs = idxs.at[replace_idxs].set(last_n_idxs)

                    return idxs

                idxs = get_first_two_idxs_per_class(val_labels, class_labels)

                # n_visualize = config.get("n_visualize", 24)
                n_visualize = len(idxs)

                val_x = val_batch_0["image"][0, idxs]
                make_reconstruction_plot = getattr(
                    models, "make_" + config.model_name.lower() + "_reconstruction_plot"
                )
                val_recon_fig = make_reconstruction_plot(
                    val_x, n_visualize, model, state, visualisation_rng
                )

                # do some sampling visualizations
                make_sampling_plot = getattr(
                    models, "make_" + config.model_name.lower() + "_sampling_plot"
                )
                sample_fig = make_sampling_plot(n_visualize, model, state, visualisation_rng)

                run.log({"val_reconstructions": wandb.Image(val_recon_fig)}, step=step)
                if sample_fig:
                    run.log({"prior_samples": wandb.Image(sample_fig)}, step=step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    run.summary["best_val_loss"] = best_val_loss
                    best_val_metrics = val_metrics
                    for k, v in best_val_metrics.items():
                        run.summary[f"best_val_{k}"] = v
                    best_val_step = step
                    run.summary["best_val_step"] = best_val_step
                    if sample_fig:
                        run.summary["best_prior_samples"] = wandb.Image(sample_fig)

                    best_state = state

        _write_note("Training finished.")

        if best_state is None:
            best_state = state

        test_batch_0 = None
        if test_ds:
            test_iter = input_utils.start_input_pipeline(
                test_ds, config.get("prefetch_to_device", 1)
            )

            batch_losses = []
            batch_metrics = []
            n_test = 0
            for i, test_batch in enumerate(test_iter):
                loss, metrics, n_examples = eval_fn(
                    best_state, test_batch["image"], test_batch["mask"], test_rng
                )

                batch_losses.append(loss)
                batch_metrics.append(metrics)
                n_test += n_examples

                if i == 0:
                    test_batch_0 = test_batch

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

        summary_batch = test_batch_0 or val_batch_0

        if summary_batch:
            make_summary_plot = getattr(
                models, "make_" + config.model_name.lower() + "_summary_plot"
            )
            for i, x in enumerate(summary_batch["image"][0, list(range(5)) + [9, 10]]):
                tmp_rng, summary_rng = jax.random.split(summary_rng)
                summary_fig = make_summary_plot(config, best_state, x, tmp_rng)
                if summary_fig:
                    run.summary[f"summary_fig_{i}"] = wandb.Image(summary_fig)

    return best_state, state
