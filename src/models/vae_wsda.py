"""VAE with standard data augmentation implementation.

A note on notation. In order to distinguish between random variables and their values, we use upper
and lower case variable names. I.e., p(Z) or `p_Z` is the distribution over the r.v. Z, and is a
function, while p(z) or `p_z` is the probability that Z=z. Similarly, p(X|Z) or `p_X_given_Z` is a
a function which returns another function p(X|z) or `p_X_given_z`, which would return the proability
that X=x|Z=z a.k.a `p_x_given_z`.
"""

from typing import Optional, Tuple

import distrax
import jax.numpy as jnp
from chex import Array
from flax import linen as nn
from imax import transforms as augmentations
from jax import random

from src.models.vae import VAE
from src.models.vae import VaeMetrics as AugVaeMetrics
from src.models.vae import VaeTrainState as AugVaeTrainState
from src.models.vae import create_vae_optimizer as create_vae_wsda_optimizer
from src.models.vae import create_vae_state as create_vae_wsda_state
from src.models.vae import make_vae_plotting_fns as make_vae_wsda_plotting_fns
from src.models.vae import make_vae_train_and_eval as make_vae_wsda_train_and_eval
from src.utils.types import KwArgs


class VAE_WSDA(nn.Module):
    # data_aug_spec: str = (
    #     f"random_rotate(-15, 15)|random_zoom({jnp.log(0.9)},{jnp.log(1.1)},{jnp.log(0.9)},{jnp.log(1.1)})|pad(2)|random_crop(28)"
    # )
    rot_bounds: Tuple[float, float] = (-15, 15)
    scale_bounds: Tuple[float, float] = (0.9, 1.1)
    translate_bounds: Tuple[int, int] = (-2, 2)
    vae: Optional[KwArgs] = None

    def setup(self):
        self.vae_model = VAE(**(self.vae or {}))

    def __call__(
        self, x: Array, train: bool = True
    ) -> Tuple[distrax.Distribution, ...]:
        if train:
            x = self.augment(x)

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
            x = self.augment(x)

        return self.vae_model.reconstruct(x, sample_z, sample_xrecon, train)

    def elbo(
        self,
        x: Array,
        train: bool = False,
        β: float = 1.0,
    ) -> float:
        if train:
            x = self.augment(x)

        return self.vae_model.elbo(x, train, β)

    def importance_weighted_lower_bound(
        self,
        x: Array,
        num_samples: int = 50,
        train: bool = False,
    ) -> float:
        if train:
            x = self.augment(x)

        return self.vae_model.importance_weighted_lower_bound(x, num_samples, train)

    def augment(self, x: Array) -> Array:
        rng = self.make_rng("sample")
        rot_rng, scale_rng, translate_rng = random.split(rng, 3)

        rot_angle = random.uniform(
            rot_rng, (), minval=self.rot_bounds[0], maxval=self.rot_bounds[1]
        )
        scale_x, scale_y = random.uniform(
            scale_rng, (2,), minval=self.scale_bounds[0], maxval=self.scale_bounds[1]
        )
        translate_x, translate_y = random.randint(
            translate_rng,
            (2,),
            minval=self.translate_bounds[0],
            maxval=self.translate_bounds[1],
        )

        aug = (
            augmentations.rotate(rot_angle * jnp.pi / 180.0)
            @ augmentations.scale(scale_x, scale_y)
            @ augmentations.translate(translate_x, translate_y)
        )

        return augmentations.apply_transform(x, aug)[:, :, :3]
