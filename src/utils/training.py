
from typing import Callable, Mapping, Optional
import numpy as np
import wandb
from tqdm.auto import tqdm
import jax
from jax import random
from jax import numpy as jnp
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
    wandb_kwargs: Optional[Mapping] = None,
) -> None:
    wandb_kwargs = {
        'project': 'learning-invariances',
        'entity': 'invariance-learners',
        'notes': '',
        # 'mode': 'disabled',
        'config': config
    } | (wandb_kwargs or {})
    # ^ here wandb_kwargs (i.e. whatever the user specifies) takes priority.

    with wandb.init(**wandb_kwargs) as run:
        z_rng, rng = random.split(rng)
        zs = random.normal(z_rng, (16, config['model']['latent_dim']))

        @jax.jit
        def train_step(state, x_batch, rng):
            loss_fn = make_loss_fn(model, x_batch, train=True)
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

            (loss, (model_state,)), grads = grad_fn(state.params, state.model_state, rng)

            return state.apply_gradients(grads=grads, model_state=model_state), loss


        valid_x, _ = next(iter(valid_loader))
        eval_fn = make_eval_fn(model, valid_x, zs)


        train_losses = []
        valid_losses = []
        for epoch in tqdm(range(config['epochs'])):
            batch_losses = []

            for (x_batch, _) in train_loader:
                rng, batch_rng = random.split(rng)
                state, loss = train_step(state, x_batch, batch_rng)
                batch_losses.append(loss)

            train_losses.append(np.mean(batch_losses))

            rng, eval_rng = random.split(rng)
            valid_metrics, recon_comparison, sampled_images = eval_fn(
                state.params, state.model_state, eval_rng
            )
            valid_losses.append(-valid_metrics['elbo'])

            print(f'epoch: {epoch:3} - train loss: {train_losses[-1]:8.4f}, valid_loss: {valid_losses[-1]:8.4f}')
            recon_plot = plot_img_array(recon_comparison)
            samples_plot = plot_img_array(sampled_images)

            metrics = {
                'epoch': epoch,
                'train/train_loss': train_losses[-1],
                'valid/valid_loss': valid_losses[-1],
                'reconstructions': recon_plot,
                'prior_samples': samples_plot,
            }
            run.log({**metrics})
