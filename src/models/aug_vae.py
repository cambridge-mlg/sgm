"""Augmented VAE implementation.

A note on notation. In order to distinguish between random variables and their values, we use upper
and lower case variable names. I.e., p(Z) or `p_Z` is the distribution over the r.v. Z, and is a
function, while p(z) or `p_z` is the probability that Z=z. Similarly, p(X|Z) or `p_X_given_Z` is a
a function which returns another function p(X|z) or `p_X_given_z`, which would return the proability
that X=x|Z=z a.k.a `p_x_given_z`.
"""

from typing import Optional, Sequence, Tuple

import distrax
import flax
import jax
import numpy as np
import optax
from chex import Array
from clu import parameter_overview
from flax import linen as nn
from flax import traverse_util
from jax import numpy as jnp
from jax import random

from src.models.transformation_generative_model import TransformationGenerativeNet
from src.models.transformation_inference_model import TransformationInferenceNet
from src.models.utils import clipped_adamw
from src.models.vae import VAE
from src.models.vae import VaeMetrics as AugVaeMetrics
from src.models.vae import VaeTrainState as AugVaeTrainState
from src.models.vae import make_vae_plotting_fns as make_aug_vae_plotting_fns
from src.models.vae import make_vae_train_and_eval as make_aug_vae_train_and_eval
from src.transformations.transforms import Transform
from src.utils.types import KwArgs


class AUG_VAE(nn.Module):
    transform: Transform
    transform_kwargs: Optional[KwArgs] = None
    vae: Optional[KwArgs] = None
    inference: Optional[KwArgs] = None
    generative: Optional[KwArgs] = None
    bounds: Optional[Sequence[float]] = None
    offset: Optional[Sequence[float]] = None

    def setup(self):
        self.vae_model = VAE(**(self.vae or {}))
        self.inference_model = TransformationInferenceNet(
            bounds=self.bounds, offset=self.offset, **(self.inference or {})
        )
        self.generative_model = TransformationGenerativeNet(
            bounds=self.bounds, offset=self.offset, **(self.generative or {})
        )

    def __call__(
        self, x: Array, train: bool = True
    ) -> Tuple[distrax.Distribution, ...]:
        # Note: if using this function to initialize the model, if train=False,
        # the inf and gen models will not be initialized. This is probably not an
        # issue, since the idea is to pre-train them.
        if train:
            x = self.resample(x, train=train)

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
            x = self.resample(x, train=train)

        return self.vae_model.reconstruct(x, sample_z, sample_xrecon, train)

    def elbo(
        self,
        x: Array,
        train: bool = False,
        β: float = 1.0,
    ) -> float:
        if train:
            x = self.resample(x, train=train)

        return self.vae_model.elbo(x, train, β)

    def importance_weighted_lower_bound(
        self,
        x: Array,
        num_samples: int = 50,
        train: bool = False,
    ) -> float:
        if train:
            x = self.resample(x, train=train)

        return self.vae_model.importance_weighted_lower_bound(x, num_samples, train)

    def resample(self, x: Array, train: bool = True) -> Array:
        q_H_given_x = self.inference_model(x, train=train)
        η = q_H_given_x.sample(seed=self.make_rng("sample"))
        η_transform = self.transform(η)
        η_transform_inv = η_transform.inverse()
        x_hat = η_transform_inv.apply(x, **(self.transform_kwargs or {}))

        p_H_given_x_hat = self.generative_model(x_hat, train=train)
        η_new = p_H_given_x_hat.sample(seed=self.make_rng("sample"))
        η_new_transform = self.transform(η_new)

        new_x = η_transform_inv.compose(η_new_transform).apply(
            x, **(self.transform_kwargs or {})
        )

        return new_x


def create_aug_vae_optimizer(params, config):
    partition_optimizers = {
        "nop": optax.set_to_zero(),
        "vae": optax.inject_hyperparams(clipped_adamw)(
            optax.warmup_cosine_decay_schedule(
                config.lr * config.init_lr_mult,
                config.lr,
                config.steps * config.warmup_steps_pct,
                config.steps,
                config.lr * config.final_lr_mult,
            ),
            config.get("clip_norm", 2.0),
            config.get("weight_decay", 1e-4),
        ),
    }

    def get_partition(path, value):
        if "vae_model" in path:
            return "vae"

        if "inference_model" in path or "generative_model" in path:
            return "nop"

    param_partitions = flax.core.freeze(
        traverse_util.path_aware_map(get_partition, params)
    )
    return optax.multi_transform(partition_optimizers, param_partitions)


def create_aug_vae_state(
    model, config, rng, input_shape, final_inf_state, final_gen_state
):
    state_rng, init_rng, dropout_rng = random.split(rng, 3)
    variables = model.init(
        {"params": init_rng, "sample": init_rng, "dropout": dropout_rng},
        jnp.empty(input_shape),
        train=True,
    )

    parameter_overview.log_parameter_overview(variables)

    params = variables["params"]
    params["inference_model"] = final_inf_state.params
    params["generative_model"] = final_gen_state.params
    params = flax.core.freeze(params)
    del variables

    parameter_overview.log_parameter_overview(params)

    opt = create_aug_vae_optimizer(params, config)

    return AugVaeTrainState.create(
        apply_fn=AUG_VAE.apply,
        params=params,
        tx=opt,
        metrics=AugVaeMetrics.empty(),
        rng=state_rng,
        β_schedule=optax.cosine_decay_schedule(
            config.β_schedule_init_value,
            config.steps,
            config.β_schedule_final_value / config.β_schedule_init_value,
        ),
    )
