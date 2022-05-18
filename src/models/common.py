import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map
import tensorflow_probability.substrates.jax.distributions as dists

from src.transformations.affine import gen_transform_mat, transform_image


def sample_transformed_data(x, rng, η_min, η_max):
    p_η = dists.Uniform(low=η_min, high=η_max)
    η = p_η.sample(sample_shape=(), seed=rng)

    T = gen_transform_mat(η)
    return transform_image(x, T)
    # TODO: this ^ breaks for flattened images, i.e. with a FC decoder.
    # A possible easiest fix is to make the FC encoder & decoder take and return
    # images, and do the flattening internally.


def make_invariant_encoder(enc, x, η_min, η_max, num_samples, rng, train):
    rngs = random.split(rng, num_samples)

    def sample_q_params(x, rng):
        x_trans = sample_transformed_data(x, rng, η_min, η_max)
        q_z_x = enc(x_trans, train=train)
        return q_z_x.loc, q_z_x.scale

    params = jax.vmap(sample_q_params, in_axes=(None, 0))(x, rngs)
    params = tree_map(lambda x: jnp.mean(x, axis=0), params)

    q_z_x = dists.Normal(*params)
    # TODO: support other distributions here.

    return q_z_x