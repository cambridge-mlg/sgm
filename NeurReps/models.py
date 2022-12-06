from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import jax
from jax import numpy as jnp
from jax import random, lax
from jax.tree_util import tree_map
from chex import Array
from flax import linen as nn
import flax.linen.initializers as init
import distrax

from src.transformations.affine import rotate_image

KwArgs = Mapping[str, Any]

# N(μ(x), σ(x))
class ConditionalNormalMLP(nn.Module):
    output_dim: int
    hidden_dims: Optional[Sequence[int]] = None
    act_fn: Callable = nn.relu
    σ_min: float = 1e-2

    @nn.compact
    def __call__(self, x):
        hidden_dims = self.hidden_dims if self.hidden_dims else [64, 32]

        h = x.reshape(-1)
        for i, hidden_dim in enumerate(hidden_dims):
            h = self.act_fn(nn.Dense(hidden_dim, name=f'hidden{i}')(h))

        μ = nn.Dense(self.output_dim, name='μ')(h)
        σ = jax.nn.softplus(nn.Dense(self.output_dim, name='σ_')(h))

        return distrax.Normal(loc=μ, scale=σ.clip(min=self.σ_min))

# p(z|x) = N(μ(x), σ(x))
class Encoder(nn.Module):
    latent_dim: int
    hidden_dims: Optional[Sequence[int]] = None
    act_fn: Callable = nn.relu
    norm_cls: nn.Module = nn.LayerNorm
    σ_min: float = 1e-2

    @nn.compact
    def __call__(self, x):
        hidden_dims = self.hidden_dims if self.hidden_dims else [64, 128, 256]

        h = x
        for i, hidden_dim in enumerate(hidden_dims):
            h = nn.Conv(
                hidden_dim,
                kernel_size=(3, 3),
                strides=(2, 2) if i==0 else (1, 1),
                name=f'hidden{i}'
            )(h)
            h = self.norm_cls(name=f'norm{i}')(h)
            h = self.act_fn(h)

        h = h.flatten()

        μ = nn.Dense(self.latent_dim, name=f'μ')(h)
        σ = jax.nn.softplus(nn.Dense(self.latent_dim, name='σ_')(h))

        return distrax.Normal(loc=μ, scale=σ.clip(min=self.σ_min))

INV_SOFTPLUS_1 = jnp.log(jnp.exp(1) - 1.)
# ^ this value is softplus^{-1}(1), i.e. if we get σ as softplus(σ_),
# and we init σ_ to this value, we effectively init σ to 1.

# p(x|z) = N(μ(z), σ)
class Decoder(nn.Module):
    image_shape: Tuple[int, int, int]
    hidden_dims: Optional[Sequence[int]] = None
    σ_init: Callable = init.constant(INV_SOFTPLUS_1)
    σ_min: float = 1e-2
    act_fn: Callable = nn.relu
    norm_cls: nn.Module = nn.LayerNorm

    @nn.compact
    def __call__(self, z):
        hidden_dims = self.hidden_dims if self.hidden_dims else [256, 128, 64]

        assert self.image_shape[0] == self.image_shape[1], "Images should be square."
        output_size = self.image_shape[0]
        first_hidden_size = output_size // 2

        h = nn.Dense(
            first_hidden_size * first_hidden_size * hidden_dims[0], name=f'resize'
        )(z)
        h = h.reshape(first_hidden_size, first_hidden_size, hidden_dims[0])

        for i, hidden_dim in enumerate(hidden_dims):
            h = nn.ConvTranspose(
                hidden_dim,
                kernel_size=(3, 3),
                strides=(2, 2) if i == 0 else (1, 1),
                name=f'hidden{i}'
            )(h)
            h = self.norm_cls(name=f'norm{i}')(h)
            h = self.act_fn(h)

        μ = nn.Conv(
            self.image_shape[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            name=f'μ'
        )(h)
        σ = jax.nn.softplus(self.param('σ_', self.σ_init, self.image_shape))

        return distrax.Normal(loc=μ, scale=σ.clip(min=self.σ_min))


# Adapted from https://github.com/deepmind/distrax/blob/master/examples/flow.py.
class Conditioner(nn.Module):
    output_dim: int
    hidden_dims: Optional[Sequence[int]] = None
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        hidden_dims = self.hidden_dims if self.hidden_dims else [64, 32]

        h = x.reshape(-1)
        for i, hidden_dim in enumerate(hidden_dims):
            h = self.act_fn(nn.Dense(hidden_dim, name=f'hidden{i}')(h))

        y = nn.Dense(
            self.output_dim,
            kernel_init=init.zeros,
            bias_init=init.zeros,
            name=f'final',
        )(h)

        return y


# Adapted from https://github.com/deepmind/distrax/blob/master/examples/flow.py.
class Bijector(nn.Module):
    num_layers: int
    num_bins: int
    hidden_dims: Optional[Sequence[int]] = None

    @nn.compact
    def __call__(self, x):
        hidden_dims = self.hidden_dims if self.hidden_dims else [64, 128]

        def bijector_fn(params: Array):
            return distrax.RationalQuadraticSpline(
                params, range_min=0., range_max=1.)

        # Number of parameters for the rational-quadratic spline:
        # - `num_bins` bin widths
        # - `num_bins` bin heights
        # - `num_bins + 1` knot slopes
        # for a total of `3 * num_bins + 1` parameters.
        num_bijector_params = 3 * self.num_bins + 1

        layers = []
        for i in range(self.num_layers):
            params = Conditioner(num_bijector_params, hidden_dims,
                                 name=f'cond{i}')(x)
            layers.append(bijector_fn(params))

        # We invert the flow so that the `forward` method is called with `log_prob`.
        bijector = distrax.Inverse(distrax.Chain(layers))
        return bijector


class LIVAE(nn.Module):
    latent_dim: int = 20
    image_shape: Tuple[int, int, int] = (28, 28, 1)
    encoder: Optional[KwArgs] = None
    decoder: Optional[KwArgs] = None
    θ_encoder: Optional[KwArgs] = None
    θ_decoder: Optional[KwArgs] = None
    θ_bijector: Optional[KwArgs] = None

    def setup(self):
        self.prior_μ = self.param(
            'prior_μ',
            init.zeros,
            (self.latent_dim,)
        )
        self.prior_σ = jax.nn.softplus(self.param(
            'prior_σ_',
            init.constant(INV_SOFTPLUS_1),
            # ^ this value is softplus^{-1}(1), i.e., σ starts at 1.
            (self.latent_dim,)
        ))

        # p(z)
        self.p_z = distrax.Normal(loc=self.prior_μ, scale=self.prior_σ)
        # q(z|x)
        self.q_z_x = Encoder(latent_dim=self.latent_dim, **(self.encoder or {}))
        # p(x̂|z)
        self.p_xhat_z = Decoder(image_shape=self.image_shape, **(self.decoder or {}))
        # p(θ|z)
        self.p_θ_z_base = ConditionalNormalMLP(output_dim=1, **(self.θ_decoder or {}))
        self.p_θ_z_bij = Bijector(num_layers=2, num_bins=4, **(self.θ_bijector or {}))
        # q(θ|x)
        self.q_θ_x_base = ConditionalNormalMLP(output_dim=1, **(self.θ_decoder or {}))
        self.q_θ_x_bij = Bijector(num_layers=2, num_bins=4, **(self.θ_bijector or {}))


    def __call__(self, x, rng):
        q_z_x = approx_invariance(self.q_z_x, x, 10, rng)
        z = q_z_x.sample(seed=rng)

        p_θ_z = distrax.Transformed(self.p_θ_z_base(z), self.p_θ_z_bij(z))

        q_θ_x = distrax.Transformed(self.q_θ_x_base(x), self.q_θ_x_bij(x))
        θ = q_θ_x.sample(seed=rng)[0]

        p_xhat_z = self.p_xhat_z(z)
        xhat = p_xhat_z.sample(seed=rng)

        x_ = rotate_image(xhat, θ, -1)
        p_x_xhat_θ = distrax.Normal(x_, 1.)

        return q_z_x, q_θ_x, p_x_xhat_θ, p_xhat_z, p_θ_z, self.p_z

    def sample(self, rng, prototype=False, sample_xhat=False, sample_θ=False):
        z = self.p_z.sample(seed=rng)

        p_xhat_z = self.p_xhat_z(z)
        xhat = p_xhat_z.sample(seed=rng) if sample_xhat else p_xhat_z.mean()
        if prototype:
            return xhat

        p_θ_z = distrax.Transformed(self.p_θ_z_base(z), self.p_θ_z_bij(z))
        θ = p_θ_z.sample(seed=rng)[0] if sample_θ else p_θ_z.mean()[0]

        x = rotate_image(xhat, θ, -1)
        return x

    def reconstruct(self, x, rng, prototype=False, sample_z=False,
                    sample_xhat=False, sample_θ=False):
        q_z_x = make_approx_invariant(self.q_z_x, x, 10, rng)
        z = q_z_x.sample(seed=rng) if sample_z else q_z_x.mean()

        p_xhat_z = self.p_xhat_z(z)
        xhat = p_xhat_z.sample(seed=rng) if sample_xhat else p_xhat_z.mean()
        if prototype:
            return xhat

        q_θ_x = distrax.Transformed(self.q_θ_x_base(x), self.q_θ_x_bij(x))
        θ = q_θ_x.sample(seed=rng)[0] if sample_θ else q_θ_x.mean()[0]

        x_recon = rotate_image(xhat, θ, -1)
        return x_recon


# TODO: generalize to other transformations.
def make_approx_invariant(p_z_x, x, num_samples, rng):
    """Construct an approximately invariant distribution by sampling parameters
    of the distribution for rotated inputs and then averaging.

    Args:
        p_z_x: A distribution whose parameters are a function of x.
        x: An image.
        num_samples: The number of samples to take.
        rng: A random number generator.

    Returns:
        An approximately invariant distribution of the same type as p.
    """
    p_θ = distrax.Uniform(low=-jnp.pi, high=jnp.pi)
    rngs = random.split(rng, num_samples)

    def sample_params(x, rng):
        θ = p_θ.sample(seed=rng)
        x_ = rotate_image(x, θ, -1)
        p_z_x_ = p_z_x(x_)
        return p_z_x_.loc, p_z_x_.scale

    params = jax.vmap(sample_params, in_axes=(None, 0))(x, rngs)
    params = tree_map(lambda x: jnp.mean(x, axis=0), params)

    dist_class = type(p_z_x)
    return dist_class(*params)

def calculate_livae_elbo(x, q_z_x, q_θ_x, p_x_xhat_θ, p_z, p_θ_z, β=1.):
    ll = p_x_xhat_θ.log_prob(x).sum()
    z_kld = q_z_x.kl_divergence(p_z).sum()
    θ_kld = q_θ_x.kl_divergence(p_θ_z).sum()

    elbo = ll - β * z_kld - θ_kld
    # TODO: add beta term for θ_kld? Use same beta?

    return elbo, {'ll': ll, 'z_kld': z_kld, 'θ_kld': θ_kld}

def make_livae_loss(
    model: LIVAE,
    x_batch: Array,
) -> Callable:
    def batch_loss(params, batch_rng, β: float = 1.):
        # TODO: this loss function is a 1 sample estimate, add an option for more samples?
        # Define loss func for 1 example.
        def loss_fn(x):
            rng = random.fold_in(batch_rng, lax.axis_index('batch'))
            q_z_x, q_θ_x, p_x_xhat_θ, _, p_θ_z, p_z = model.apply(
                {'params': params}, x, rng,
            )

            elbo, metrics = calculate_livae_elbo(x, q_z_x, q_θ_x, p_x_xhat_θ,
                                                 p_z, p_θ_z, β)

            return -elbo, metrics

        # Broadcast over batch and take aggregate.
        batch_losses, batch_metrics = jax.vmap(
            loss_fn, out_axes=(0, 0), in_axes=(0), axis_name='batch'
        )(x_batch)
        batch_metrics = tree_map(lambda x: x.mean(axis=0), batch_metrics)
        return batch_losses.mean(axis=0), (batch_metrics)

    return jax.jit(batch_loss)