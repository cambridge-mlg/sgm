from typing import Callable

import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map
import tensorflow_probability.substrates.jax.distributions as dists

from src.transformations.affine import gen_transform_mat, transform_image


MIN_η = jnp.array([0., 0., -jnp.pi/2, 0., 0., 0., 0.])
MAX_η = jnp.array([0., 0., jnp.pi/2, 0., 0., 0., 0.])
# ^ For now we are just working with rotations, so other transformations are off.


INV_SOFTPLUS_1 = jnp.log(jnp.exp(1) - 1.)
# ^ this value is softplus^{-1}(1), i.e. if we get σ as softplus(σ_),
# and we init σ_ to this value, we effectively init σ to 1.


def get_agg_fn(agg: str) -> Callable:
    raise_if_not_in_list(agg, ['mean', 'sum'], 'aggregation')

    if agg == 'mean':
        return jnp.mean
    else:
        return jnp.sum


def raise_if_not_in_list(val, valid_options, varname):
    if val not in valid_options:
       msg = f'`{varname}` should be one of `{valid_options}` but was `{val}` instead.'
       raise RuntimeError(msg)


def sample_transformed_data(x, rng, η_low, η_high):
    p_η = dists.Uniform(low=η_low, high=η_high)
    η = p_η.sample(sample_shape=(), seed=rng)

    T = gen_transform_mat(η)
    return transform_image(x, T)
    # TODO: this ^ breaks for flattened images, i.e. with a FC decoder.
    # A possible easiest fix is to make the FC encoder & decoder take and return
    # images, and do the flattening internally.


def make_invariant_encoder(enc, x, η_low, η_high, num_samples, rng, train):
    rngs = random.split(rng, num_samples)

    def sample_q_params(x, rng):
        x_trans = sample_transformed_data(x, rng, η_low, η_high)
        q_z_x = enc(x_trans, train=train)
        return q_z_x.loc, q_z_x.scale

    params = jax.vmap(sample_q_params, in_axes=(None, 0))(x, rngs)
    params = tree_map(lambda x: jnp.mean(x, axis=0), params)

    q_z_x = dists.Normal(*params)
    # TODO: support other distributions here.

    return q_z_x