import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map
import tensorflow_probability.substrates.jax.distributions as dists

from src.transformations.affine import gen_transform_mat, transform_image


def sample_transformed_data(xhat, rng, rotation):
    η = jnp.array([0, 0, rotation, 0, 0, 0])
    ε = random.uniform(rng, (6,), minval=-1., maxval=1.)
    # TODO: make this work for a different min and max - see src.data.image._transform_data
    T = gen_transform_mat(η * ε)
    return transform_image(xhat, T)
    # TODO: this ^ breaks for flattened images, i.e. with a FC decoder.


def make_invariant_encoder(enc, x, inv_rot, num_samples, rng, train):
    rngs = random.split(rng, num_samples)

    def sample_q_params(x, rng):
        x_trans = sample_transformed_data(x, rng, inv_rot)
        q_z_x = enc(x_trans, train=train)
        return q_z_x.loc, q_z_x.scale

    params = jax.vmap(sample_q_params, in_axes=(None, 0))(x, rngs)
    params = tree_map(lambda x: jnp.mean(x, axis=0), params)

    q_z_x = dists.Normal(*params)
    # TODO: support other distributions here.

    return q_z_x