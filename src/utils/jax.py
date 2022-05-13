import jax
from jax import numpy as jnp

def tree_transpose(list_of_trees):
    """Convert a list of trees of identical structure into a single tree of lists."""
    return jax.tree_map(lambda *xs: jnp.array(list(xs)), *list_of_trees)