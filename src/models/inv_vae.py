"""Invariant VAE implementation.

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
from src.models.vae import VaeMetrics as InvVaeMetrics
from src.models.vae import VaeTrainState as InvVaeTrainState
from src.models.vae import make_vae_plotting_fns as make_inv_vae_plotting_fns
from src.models.vae import make_vae_train_and_eval as make_inv_vae_train_and_eval
from src.transformations.transforms import Transform
from src.utils.types import KwArgs


class INV_VAE(nn.Module):
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
        x_hat = self.make_proto(x, train=train)

        return self.vae_model(x_hat, train=train)

    def sample(
        self,
        sample_x: bool = False,
        train: bool = True,
        return_z: bool = False,
        sample_η: bool = False,
    ) -> Array:
        x_hat = self.vae_model.sample(sample_x, train, return_z)
        if not sample_η:
            return x_hat

        p_Η_given_x_hat = self.generative_model(x_hat, train=train)
        η = p_Η_given_x_hat.sample(seed=self.make_rng("sample"))
        η_transform = self.transform(η)
        x = η_transform.apply(x_hat, **(self.transform_kwargs or {}))
        return x

    def reconstruct(
        self,
        x: Array,
        sample_z: bool = False,
        sample_xrecon: bool = False,
        train: bool = True,
    ) -> Array:
        x_hat, η_transform = self.make_proto(x, train=train, return_transform=True)

        x_hat_recon = self.vae_model.reconstruct(x_hat, sample_z, sample_xrecon, train)

        x_recon = η_transform.apply(x_hat_recon, **(self.transform_kwargs or {}))
        return x_recon

    def elbo(
        self,
        x: Array,
        train: bool = False,
        β: float = 1.0,
    ) -> float:
        x_hat = self.make_proto(x, train=train)

        return self.vae_model.elbo(x_hat, train, β)

    def vae_importance_weighted_lower_bound(
        self,
        x: Array,
        num_samples: int = 50,
        train: bool = False,
    ) -> float:
        x_hat = self.make_proto(x, train=train)

        return self.vae_model.importance_weighted_lower_bound(x_hat, num_samples, train)

    def importance_weighted_lower_bound(
        self,
        x: Array,
        num_samples: int = 50,
        train: bool = False,
    ) -> float:
        def single_sample_log_w(i):
            q_H_given_x = self.inference_model(x, train=train)
            η = q_H_given_x.sample(seed=random.fold_in(self.make_rng("sample"), i))
            η_transform = self.transform(η)
            η_transform_inv = η_transform.inverse()
            x_hat = η_transform_inv.apply(x, **(self.transform_kwargs or {}))

            q_Z_given_x_hat = self.vae_model.q_Z_given_X(x_hat, train=train)
            z = q_Z_given_x_hat.sample(seed=random.fold_in(self.make_rng("sample"), i))
            p_X_hat_given_z = self.vae_model.p_X_given_Z(z, train=train)

            p_Η_given_x_hat = self.generative_model(x_hat, train=train)

            log_p_η_given_x_hat = p_Η_given_x_hat.log_prob(η)
            log_q_η_given_x = q_H_given_x.log_prob(η)
            log_p_x_hat_given_z = p_X_hat_given_z.log_prob(x_hat)
            log_q_z_given_x_hat = q_Z_given_x_hat.log_prob(z)
            log_p_z = self.vae_model.p_Z.log_prob(z)

            return (
                log_p_η_given_x_hat
                + log_p_x_hat_given_z
                + log_p_z
                - log_q_η_given_x
                - log_q_z_given_x_hat
            )

        log_ws = jax.vmap(single_sample_log_w)(jnp.arange(num_samples))

        return jax.nn.logsumexp(log_ws, axis=0) - jnp.log(num_samples)

    def make_proto(
        self, x: Array, train: bool = True, return_transform: bool = False
    ) -> Array:
        q_H_given_x = self.inference_model(x, train=train)
        η = q_H_given_x.sample(seed=self.make_rng("sample"))
        η_transform = self.transform(η)
        η_transform_inv = η_transform.inverse()
        x_hat = η_transform_inv.apply(x, **(self.transform_kwargs or {}))

        if return_transform:
            return x_hat, η_transform
        else:
            return x_hat


def create_inv_vae_optimizer(params, config):
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


def create_inv_vae_state(
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

    opt = create_inv_vae_optimizer(params, config)

    return InvVaeTrainState.create(
        apply_fn=INV_VAE.apply,
        params=params,
        tx=opt,
        metrics=InvVaeMetrics.empty(),
        rng=state_rng,
        β_schedule=optax.cosine_decay_schedule(
            config.β_schedule_init_value,
            config.steps,
            config.β_schedule_final_value / config.β_schedule_init_value,
        ),
    )
