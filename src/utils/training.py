
from typing import Callable, Mapping, Optional, Tuple, Union
from functools import partial

import wandb
from tqdm.auto import trange
import jax
from jax import random
from jax import numpy as jnp
from jax.tree_util import tree_map
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from flax import struct
import flax.linen as nn
import optax
from chex import Array
from ml_collections import config_dict
from clu import parameter_overview

from src.utils.plotting import plot_img_array
from src.data import NumpyLoader
import src.models as models


PRNGKey = jnp.ndarray
ScalarOrSchedule = Union[float, optax.Schedule]


class TrainState(train_state.TrainState):
    """A Flax TrainState which also tracks model (e.g. BN) state and schedules β."""
    model_state: FrozenDict
    β: float
    β_val_or_schedule: ScalarOrSchedule = struct.field(pytree_node=False)

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            β=_get_β_for_step(self.step, self.β_val_or_schedule),
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, β_val_or_schedule, **kwargs):
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            β_val_or_schedule=β_val_or_schedule,
            β=_get_β_for_step(0, β_val_or_schedule),
            **kwargs,
        )


def _get_β_for_step(step, β_val_or_schedule):
    if callable(β_val_or_schedule):
        return β_val_or_schedule(step)
    else:
        return β_val_or_schedule


def setup_training(
    config: config_dict.ConfigDict,
    rng: PRNGKey,
    init_data: Array,
) -> Tuple[nn.Module, TrainState]:
    model_cls = getattr(models, config.model_name)
    model = model_cls(**config.model.to_dict())

    init_rng, rng = random.split(rng)
    variables = model.init(init_rng, init_data, rng)

    print(parameter_overview.get_parameter_overview(variables))

    model_state, params = variables.pop('params')
    del variables

    if config.get('lr_schedule_name', None):
        schedule = getattr(optax, config.lr_schedule_name)
        lr = schedule(
            init_value=config.learning_rate,
            **config.lr_schedule.to_dict()
        )
    else:
        lr = config.learning_rate

    optim = getattr(optax, config.optim_name)
    optim = optax.inject_hyperparams(optim)
    # This ^ allows us to access the lr as opt_state.hyperparams['learning_rate'].

    if config.get('β_schedule_name', None):
        schedule = getattr(optax, config.β_schedule_name)
        β = schedule(
            init_value=config.β,
            **config.β_schedule.to_dict()
        )
    else:
        β = config.β

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optim(learning_rate=lr, **config.optim.to_dict()),
        model_state=model_state,
        β_val_or_schedule=β,
    )

    return model, state


def train_loop(
    model: nn.Module,
    state: TrainState,
    config: config_dict.ConfigDict,
    rng: PRNGKey,
    make_loss_fn: Callable,
    make_eval_fn: Callable,
    train_loader: NumpyLoader,
    val_loader: NumpyLoader,
    test_loader: Optional[NumpyLoader] = None,
    wandb_kwargs: Optional[Mapping] = None,
) -> TrainState:
    wandb_kwargs = {
        'project': 'learning-invariances',
        'entity': 'invariance-learners',
        'notes': '',
        # 'mode': 'disabled',
        'config': config.to_dict()
    } | (wandb_kwargs or {})
    # ^ here wandb_kwargs (i.e. whatever the user specifies) takes priority.

    with wandb.init(**wandb_kwargs) as run:
        z_rng, rng = random.split(rng)
        zs = random.normal(z_rng, (16, config.model.latent_dim))

        @jax.jit
        def train_step(state, x_batch, rng):
            loss_fn = make_loss_fn(model, x_batch, train=True)
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

            (loss, (model_state, metrics)), grads = grad_fn(
                state.params, state.model_state, rng, state.β,
            )

            return state.apply_gradients(grads=grads, model_state=model_state), loss, metrics

        @partial(jax.jit, static_argnames='metrics_only')
        def eval_step(state, x_batch, rng, metrics_only=False):
            eval_fn = make_eval_fn(model, x_batch, zs, config.model.decoder.image_shape)

            metrics, recon_comparison, sampled_images = eval_fn(
                state.params, state.model_state, rng, state.β,
            )

            if metrics_only:
                return metrics
            else:
                return metrics, recon_comparison, sampled_images


        train_losses = []
        val_losses = []
        epochs = trange(1, config.epochs + 1)
        for epoch in epochs:
            batch_losses = []
            batch_metrics = []
            for (x_batch, _) in train_loader:
                rng, batch_rng = random.split(rng)
                state, loss, metrics = train_step(state, x_batch, batch_rng)
                batch_losses.append(loss)
                batch_metrics.append(metrics)

            train_metrics = tree_map(lambda x: jnp.mean(x), tree_transpose(batch_metrics))
            train_losses.append(-train_metrics['elbo'])

            batch_metrics = []
            for i, (x_batch, _) in enumerate(val_loader):
                rng, eval_rng = random.split(rng)
                if i==0:
                    metrics, recon_comparison, sampled_images = eval_step(state, x_batch, eval_rng)
                else:
                    metrics = eval_step(state, x_batch, eval_rng, metrics_only=True)
                batch_metrics.append(metrics)

            val_metrics = tree_map(lambda x: jnp.mean(x), tree_transpose(batch_metrics))
            val_losses.append(-val_metrics['elbo'])

            learning_rate = state.opt_state.hyperparams['learning_rate']
            metrics_str = (f'train loss: {train_losses[-1]:7.5f}, val_loss: {val_losses[-1]:7.5f}' +
                           f', β: {state.β:3.1f}, lr: {learning_rate:7.5f}')
            epochs.set_postfix_str(metrics_str)
            print(f'epoch: {epoch:3} - {metrics_str}')

            recon_plot_title = "Reconstructions – Top: original; Mid: mode; Bot: sample"
            recon_plot = plot_img_array(recon_comparison, title=recon_plot_title)
            samples_plot_title = "Prior Samples – Top: mode; Bot: sample"
            samples_plot = plot_img_array(sampled_images, title=samples_plot_title)
            metrics = {
                'epoch': epoch,
                'train/loss': train_losses[-1],
                **{'train/' + key: val for key, val in train_metrics.items()},
                'val/loss': val_losses[-1],
                'val_reconstructions': recon_plot,
                **{'val/' + key: val for key, val in val_metrics.items()},
                'prior_samples': samples_plot,
                'β': state.β,
                'learning_rate': learning_rate,
            }
            run.log(metrics)

            rng, test_rng = random.split(rng)
            if val_losses[-1] <= min(val_losses):
                print("Best val_loss")
                # TODO add model saving.

                run.summary['best_epoch'] = epoch
                run.summary['best_val_loss'] = val_losses[-1]

                if test_loader is not None:
                    batch_metrics = []
                    for i, (x_batch, _) in enumerate(test_loader):
                        eval_rng, test_rng = random.split(test_rng)
                        if i==0:
                            metrics, recon_comparison, sampled_images = eval_step(state, x_batch, eval_rng)
                        else:
                            metrics = eval_step(state, x_batch, eval_rng, metrics_only=True)
                        batch_metrics.append(metrics)

                    test_metrics = tree_map(lambda x: jnp.mean(x), tree_transpose(batch_metrics))

                    run.summary['test/loss'] = -test_metrics['elbo']
                    for key, val in test_metrics.items():
                        run.summary['test/' + key] = val
                    run.summary['test_reconstructions'] = plot_img_array(
                        recon_comparison, title=recon_plot_title)
                    run.summary['best_prior_samples'] = plot_img_array(
                        sampled_images, title=samples_plot_title)

    return state



def tree_transpose(list_of_trees):
    """Convert a list of trees of identical structure into a single tree of lists."""
    return jax.tree_map(lambda *xs: jnp.array(list(xs)), *list_of_trees)
