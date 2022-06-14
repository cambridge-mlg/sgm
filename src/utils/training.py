
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
from src.utils.jax import tree_concatenate
from src.data import NumpyLoader
import src.models as models


PRNGKey = jnp.ndarray
ScalarOrSchedule = Union[float, optax.Schedule]


class TrainState(train_state.TrainState):
    """A Flax TrainState which also tracks model state (e.g. BatchNorm running averages) and schedules β."""
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


# Helper which helps us deal with the fact that we can either specify a fixed β
# or a schedule for adjusting β. This is a pattern similar to the one used by
# optax for dealing with LRs either being specified as a constant or schedule.
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
    """Helper which returns the model object and the corresponding initialised train state for a given config.

    Notes:
        config is a `config_dict.ConfigDict` containing the following entries:
        * config.model_name
        * config.model (a config_dict that contains everything required by the model constructor)
        * config.learning_rate
        * config.optim_name (the name of the optax optimizer)
        * config.optim (a possibly empty config dict containing everything for the scheduler constructor,
        except the learning_rate)
        * config.lr_schedule_name (optional, the name of the optax scheduler to use for lr)
        * config.lr_schedule (optional, a config dict containing everything for the scheduler constructor,
        except the initial value, which is specified by config.learning_rate)
        * config.β
        * config.β_schedule_name (optional, the name of the optax scheduler to use for β)
        * config.β_schedule (optional, a config dict containing everything for the scheduler constructor,
        except the initial value, which is specified by config.β)

        init_data is an Array of the same shape as a *single* training example.
    """
    model_cls = getattr(models, config.model_name)
    model = model_cls(**config.model.to_dict())

    init_rng, rng = random.split(rng)
    variables = model.init(init_rng, init_data, rng)

    print(parameter_overview.get_parameter_overview(variables))
    # ^ This is really nice for summarising Jax models!

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
    """Runs the training loop!

    Notes:
        The `state` arg is a `src.utils.training.TrainState`, which tracks model_state and schedules β.

        The `config` arg is a `config_dict.ConfigDict`. It must contain the following entries:
        * config.model.latent_dim
        * config.model.decoder.image_shape
        * config.epochs

        `make_loss_fn` is a callable with three arguments (model, x_batch, train), that returns the loss
        function for training. A closure or `functools.partial` can be used to deal with additional
        arguments. The returned loss function should take four arguments
        (params, model_state, rng, β) and return (loss, (new_model_state, metrics)).

        `make_eval_fn` is a callable with four arguments (model, x_batch, zs, image_shape), that returns
        a function for evaluating the model. The returned evaluation function should take four
        arguments (params, model_state, rng, β) and return (metrics, recon_data, sample_data, recon_title, sample_title).
        TODO: describe the format for reco_comaprison and sample_data, recon_title, sample_title.

        If `test_loader` is supplied, on epochs for which the validation loss is the new best, the test set
        will be evaluated using `make_eval_fn` and the results will be added to the W&B summary.

        `wandb_kwargs` are a dictionary for overriding the kwargs passed to `wandb.init`. The only kwargs
        specified by default are `project='learning-invariances'`, `entity='invariance-learners'`, and
        `config=config.to_dict`. For example, specifying `wandb_kwargs={'mode': 'disabled'}` will stop W&B syncing.
    """
    wandb_kwargs = {
        'project': 'learning-invariances',
        'entity': 'invariance-learners',
        'notes': '',
        # 'mode': 'disabled',
        'config': config.to_dict()
    } | (wandb_kwargs or {})
    # ^ here wandb_kwargs (i.e. whatever the user specifies) takes priority.

    with wandb.init(**wandb_kwargs) as run:
        zs_rng, rng = random.split(rng)

        @jax.jit
        def train_step(state, x_batch, rng):
            loss_fn = make_loss_fn(model, x_batch, train=True, aggregation='sum')
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

            (_, (model_state, metrics)), grads = grad_fn(
                state.params, state.model_state, rng, state.β,
            )

            return state.apply_gradients(grads=grads, model_state=model_state), metrics

        @partial(jax.jit, static_argnames='metrics_only')
        def eval_step(state, x_batch, rng, metrics_only=False):
            eval_fn = make_eval_fn(model, x_batch, config.model.decoder.image_shape, aggregation='sum', zs_rng=zs_rng)

            metrics, recon_data, sample_data = eval_fn(
                state.params, state.model_state, rng, state.β,
            )

            if metrics_only:
                return metrics
            else:
                return metrics, recon_data, sample_data


        train_losses = []
        val_losses = []
        epochs = trange(1, config.epochs + 1)
        for epoch in epochs:
            batch_metrics = []
            for (x_batch, _) in train_loader:
                rng, batch_rng = random.split(rng)
                state, metrics = train_step(state, x_batch, batch_rng)
                batch_metrics.append(metrics)

            train_metrics = tree_map(
                lambda x: jnp.sum(x) / len(train_loader.dataset),
                tree_concatenate(batch_metrics)
            )
            train_losses.append(-train_metrics['elbo'])

            batch_metrics = []
            for i, (x_batch, _) in enumerate(val_loader):
                rng, eval_rng = random.split(rng)
                if i==0:
                    metrics, recon_data, sample_data = eval_step(state, x_batch, eval_rng)
                else:
                    metrics = eval_step(state, x_batch, eval_rng, metrics_only=True)
                batch_metrics.append(metrics)

            val_metrics = tree_map(
                lambda x: jnp.sum(x) / len(val_loader.dataset),
                tree_concatenate(batch_metrics)
            )
            val_losses.append(-val_metrics['elbo'])

            learning_rate = state.opt_state.hyperparams['learning_rate']
            metrics_str = (f'train loss: {train_losses[-1]:7.5f}, val_loss: {val_losses[-1]:7.5f}' +
                           f', β: {state.β:3.1f}, lr: {learning_rate:7.5f}')
            epochs.set_postfix_str(metrics_str)
            print(f'epoch: {epoch:3} - {metrics_str}')

            recon_plot = plot_img_array(recon_data, title=model.recon_title)
            samples_plot = plot_img_array(sample_data, title=model.sample_title)
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
                # TODO: add model saving.

                run.summary['best_epoch'] = epoch
                run.summary['best_val_loss'] = val_losses[-1]

                if test_loader is not None:
                    batch_metrics = []
                    for i, (x_batch, _) in enumerate(test_loader):
                        eval_rng, test_rng = random.split(test_rng)
                        if i==0:
                            metrics, recon_data, sample_data = eval_step(state, x_batch, eval_rng)
                        else:
                            metrics = eval_step(state, x_batch, eval_rng, metrics_only=True)
                        batch_metrics.append(metrics)

                    test_metrics = tree_map(
                        lambda x: jnp.sum(x) / len(test_loader.dataset),
                        tree_concatenate(batch_metrics)
                    )

                    run.summary['test/loss'] = -test_metrics['elbo']
                    for key, val in test_metrics.items():
                        run.summary['test/' + key] = val
                    run.summary['test_reconstructions'] = plot_img_array(recon_data, title=model.recon_title)
                    run.summary['best_prior_samples'] = plot_img_array(sample_data, title=model.sample_title)

    return state
