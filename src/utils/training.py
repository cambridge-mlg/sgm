
from typing import Callable, Mapping, Optional
from functools import partial

import numpy as np
import wandb
from tqdm.auto import trange
import jax
from jax import random
from jax import numpy as jnp
from jax.tree_util import tree_map
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
import flax.linen as nn
from ml_collections import config_dict

from src.utils.plotting import plot_img_array
from src.data import NumpyLoader


PRNGKey = jnp.ndarray


class TrainState(train_state.TrainState):
    """A Flax TrainState which also tracks model state such as BN state."""
    model_state: FrozenDict


def train_loop(
    model: nn.Module,
    state: TrainState,
    config: config_dict,
    rng: PRNGKey,
    make_loss_fn: Callable,
    make_eval_fn: Callable,
    train_loader: NumpyLoader,
    valid_loader: NumpyLoader,
    test_loader: Optional[NumpyLoader] = None,
    wandb_kwargs: Optional[Mapping] = None,
) -> None:
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

            (loss, (model_state, metrics)), grads = grad_fn(state.params, state.model_state, rng)

            return state.apply_gradients(grads=grads, model_state=model_state), loss, metrics

        @partial(jax.jit, static_argnames='metrics_only')
        def eval_step(state, x_batch, rng, metrics_only=False):
            eval_fn = make_eval_fn(model, x_batch, zs, config.model.decoder.image_shape)

            metrics, recon_comparison, sampled_images = eval_fn(
                state.params, state.model_state, rng,
            )

            if metrics_only:
                return metrics
            else:
                return metrics, recon_comparison, sampled_images


        train_losses = []
        valid_losses = []
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
            for i, (x_batch, _) in enumerate(valid_loader):
                rng, eval_rng = random.split(rng)
                if i==0:
                    metrics, recon_comparison, sampled_images = eval_step(state, x_batch, eval_rng)
                else:
                    metrics = eval_step(state, x_batch, eval_rng, metrics_only=True)
                batch_metrics.append(metrics)

            valid_metrics = tree_map(lambda x: jnp.mean(x), tree_transpose(batch_metrics))
            valid_losses.append(-valid_metrics['elbo'])

            losses_str = f'train loss: {train_losses[-1]:8.4f}, valid_loss: {valid_losses[-1]:8.4f}'
            epochs.set_postfix_str(losses_str)
            print(f'epoch: {epoch:3} - {losses_str}')

            recon_plot_title = "Reconstructions – Top: original; Mid: mode; Bot: sample"
            recon_plot = plot_img_array(recon_comparison, title=recon_plot_title)
            samples_plot_title = "Prior Samples – Top: mode; Bot: sample"
            samples_plot = plot_img_array(sampled_images, title=samples_plot_title)
            metrics = {
                'epoch': epoch,
                'train/loss': train_losses[-1],
                **{'train/' + key: val for key, val in train_metrics.items()},
                'valid/loss': valid_losses[-1],
                'valid_reconstructions': recon_plot,
                **{'valid/' + key: val for key, val in valid_metrics.items()},
                'prior_samples': samples_plot,
            }
            run.log(metrics)

            rng, test_rng = random.split(rng)
            if valid_losses[-1] <= min(valid_losses):
                print("Best valid_loss")
                # TODO add model saving.

                run.summary['best_epoch'] = epoch
                run.summary['best_valid_loss'] = valid_losses[-1]

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



def tree_transpose(list_of_trees):
    """Convert a list of trees of identical structure into a single tree of lists."""
    return jax.tree_map(lambda *xs: jnp.array(list(xs)), *list_of_trees)
