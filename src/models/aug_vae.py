"""VAE implementation.

A note on notation. In order to distinguish between random variables and their values, we use upper
and lower case variable names. I.e., p(Z) or `p_Z` is the distribution over the r.v. Z, and is a
function, while p(z) or `p_z` is the probability that Z=z. Similarly, p(X|Z) or `p_X_given_Z` is a
a function which returns another function p(X|z) or `p_X_given_z`, which would return the proability
that X=x|Z=z a.k.a `p_x_given_z`.
"""

from typing import Optional, Tuple

import distrax
import flax
import numpy as np
import optax
from chex import Array
from flax import linen as nn
from flax import traverse_util
from jax import numpy as jnp

from src.models.proto_gen_model import PrototypicalGenerativeModel
from src.models.vae import VAE
from src.models.vae import VaeMetrics as AugVaeMetrics
from src.models.vae import VaeTrainState as AugVaeTrainState
from src.models.vae import make_vae_plotting_fns as make_aug_vae_plotting_fns
from src.models.vae import make_vae_train_and_eval as make_aug_vae_train_and_eval
from src.utils.types import KwArgs

INV_SOFTPLUS_1 = jnp.log(jnp.exp(1) - 1.0)


class AUG_VAE(nn.Module):
    vae: Optional[KwArgs] = None
    pgm: Optional[KwArgs] = None

    def setup(self):
        self.vae_model = VAE(**(self.vae or {}))
        self.proto_gen_model = PrototypicalGenerativeModel(**(self.pgm or {}))

    def __call__(
        self, x: Array, train: bool = True
    ) -> Tuple[distrax.Distribution, ...]:
        # Note: if using this function to initialize the model, if train=False,
        # the proto_gen_model will not be initialized. This is probablt not an
        # issue, since the idea is to pre-train the pgm.
        if train:
            x = self.proto_gen_model.resample(x, train=train)

        return self.vae_model(x, train=train)

    def sample(
        self,
        sample_x: bool = False,
        train: bool = True,
        return_z: bool = False,
    ) -> Array:
        return self.vae_model.sample(sample_x, train, return_z)

    def reconstruct(
        self,
        x: Array,
        sample_z: bool = False,
        sample_xrecon: bool = False,
        train: bool = True,
    ) -> Array:
        if train:
            x = self.proto_gen_model.resample(x, train=train)

        return self.vae_model.reconstruct(x, sample_z, sample_xrecon, train)

    def importance_weighted_lower_bound(
        self,
        x: Array,
        num_samples: int = 50,
        train: bool = False,
    ) -> float:
        if train:
            x = self.proto_gen_model.resample(x, train=train)

        return self.vae_model.importance_weighted_lower_bound(x, num_samples, train)


def create_aug_vae_optimizer(params, config):
    partition_optimizers = {
        "pgm": optax.identity(),
        "vae": optax.inject_hyperparams(optax.adam)(
            optax.warmup_cosine_decay_schedule(
                config.init_lr,
                config.init_lr * config.peak_lr_mult,
                config.warmup_steps,
                config.steps,
                config.init_lr * config.final_lr_mult,
            )
        ),
    }

    def get_partition(path, value):
        if "vae_model" in path:
            return "vae"

        if "proto_gen_model" in path:
            return "pgm"

    param_partitions = flax.core.freeze(
        traverse_util.path_aware_map(get_partition, params)
    )
    return optax.multi_transform(partition_optimizers, param_partitions)


def create_aug_vae_state(params, rng, config):
    opt = create_aug_vae_optimizer(params, config)

    return AugVaeTrainState.create(
        apply_fn=AUG_VAE.apply,
        params=params,
        tx=opt,
        metrics=AugVaeMetrics.empty(),
        rng=rng,
        β_schedule=optax.cosine_decay_schedule(
            config.β_schedule_init_value,
            config.steps,
            config.β_schedule_final_value / config.β_schedule_init_value,
        ),
    )
