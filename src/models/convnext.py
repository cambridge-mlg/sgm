import enum
from functools import partial
from typing import Any, Callable, Sequence

from jax import numpy as jnp
from jax import random
from jax.tree_util import tree_map
from jax.nn.initializers import Initializer
from jax._src import dtypes
from chex import Array, PRNGKey, Shape, assert_rank, assert_equal_shape
from flax import linen as nn
import flax.linen.initializers as init


class Block(nn.Module):
    """
    ConvNeXT Block. From https://github.com/facebookresearch/ConvNeXt-V2
    
    Args:
        dim (int): Number of input channels.
    """
    num_channels: int
    norm_cls: Callable[[], nn.Module] =  partial(nn.LayerNorm, epsilon=1e-6)

    @nn.compact
    def __call__(
        self,
        x: Array,  # [height, width, channel_dim]
    ) -> Array:
        residual = x
        x = nn.Conv(
            features=self.num_channels,
            kernel_size=(7, 7),
            padding='SAME',
            kernel_init=truncated_normal(stddev=0.02),
            bias_init=init.zeros,
            feature_group_count=self.num_channels,
        )(x)
        x = self.norm_cls()(x)
        # pointwise/1x1 convs, implemented with linear layers
        x = nn.Dense(4 * self.num_channels, use_bias=True, kernel_init=truncated_normal(0.02), bias_init=init.zeros)(x)
        x = nn.gelu(x)
        # pointwise/1x1 convs, implemented with linear layers
        x = nn.Dense(self.num_channels, use_bias=True, kernel_init=truncated_normal(0.02), bias_init=init.zeros)(x)

        # Use layer scale from ConvNext V1
        gamma = self.param("gamma", lambda key, shape: jnp.full(shape, 1e-6), (self.num_channels,))
        x = gamma * x

        return residual + x


class ConvNeXt(nn.Module):
    """
    ConvNeXt V1 adapted from https://github.com/facebookresearch/ConvNeXt-V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    in_channels: int = 1
    num_outputs: int = 10
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)
    head_init_scale: float = 1.
    init_downsample: int = 2  # The stride of the first conv layer. Use `1` for small res datasets
    norm_cls: Callable[[], nn.Module] = partial(nn.LayerNorm, epsilon=1e-6)

    @nn.compact
    def __call__(self, x: Array) -> Array:
        # --- Stem
        x = nn.Conv(
            features=self.dims[0],
            kernel_size=(4, 4),
            strides=(self.init_downsample, self.init_downsample),
            padding=0,
            kernel_init=truncated_normal(stddev=0.02),
            bias_init=init.zeros,
        )(x)
        x = self.norm_cls()(x)
        for _ in range(self.depths[0]):
            x = Block(self.dims[0])(x)
        # --- Body
        for i in range(1, len(self.depths)):
            # Downsample
            x = self.norm_cls()(x)
            x = nn.Conv(
                features=self.dims[i],
                kernel_size=(2, 2),
                strides=(2, 2),
                padding=0,
                kernel_init=truncated_normal(stddev=0.02),
                bias_init=init.zeros,
            )(x)
            for j in range(self.depths[i]):
                x = Block(self.dims[i])(x)
        # --- Head
        x = x.mean(axis=(-3, -2))  # Global average pooling across height, width axes
        x = nn.Dense(self.num_outputs, kernel_init=truncated_normal(0.02 * self.head_init_scale), bias_init=init.zeros)(x)
        return x


def truncated_normal(stddev = 1e-2, dtype = jnp.float_) -> Initializer:
    """Builds an initializer that returns real trunc. normally-distributed random arrays.

    Args:
    stddev: optional; the standard deviation of the distribution.
    dtype: optional; the initializer's default dtype.

    Returns:
    An initializer that returns arrays whose values are normally distributed
    with mean ``0`` and standard deviation ``stddev``.

    >>> import jax, jax.numpy as jnp
    >>> initializer = jax.nn.initializers.trunc_normal(5.0)
    >>> initializer(jax.random.PRNGKey(42), (2, 3), jnp.float32)  # doctest: +SKIP
    Array([[ 3.0613258 ,  5.6129413 ,  5.6866574 ],
            [-4.063663  , -4.4520254 ,  0.63115686]], dtype=float32)
    """
    def init(
            key: PRNGKey,
            shape: Shape,
            dtype = dtype,
        ) -> Array:
        dtype = dtypes.canonicalize_dtype(dtype)
        return random.truncated_normal(key, -2, 2, shape, dtype) * stddev
    return init


def convenext_atto(**kwargs) -> ConvNeXt:
    model = ConvNeXt(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convenext_femto(**kwargs) -> ConvNeXt:
    model = ConvNeXt(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convenext_pico(**kwargs)-> ConvNeXt:
    model = ConvNeXt(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convenext_nano(**kwargs) -> ConvNeXt:
    model = ConvNeXt(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convenext_tiny(**kwargs) -> ConvNeXt:
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convenext_base(**kwargs) -> ConvNeXt:
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convenext_large(**kwargs) -> ConvNeXt:
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convenext_huge(**kwargs) -> ConvNeXt:
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model

def convenext_pathetic(**kwargs) -> ConvNeXt:
    """Not an official ConvNeXt model, but a small one for testing purposes."""
    model = ConvNeXt(depths=[1, 1, 2, 1], dims=[20, 40, 80, 160], **kwargs)
    return model

def convenext_very_pathetic(**kwargs) -> ConvNeXt:
    """Not an official ConvNeXt model, but a small one for testing purposes."""
    model = ConvNeXt(depths=[1, 1, 1, 1], dims=[2, 4, 6, 8], **kwargs)
    return model


class ConvNextType(str, enum.Enum):
    PATHETIC = "pathetic"
    VERY_PATHETIC = "very_pathetic"
    ATTO = "atto"
    FEMTO = "femto"
    PICO = "pico"
    NANO = "nano"
    TINY = "tiny"
    BASE = "base"
    LARGE = "large"
    HUGE = "huge"


def get_convnext_constructor(convnext_type: ConvNextType) -> Callable[[], ConvNeXt]:
    """Get a ConvNeXt constructor.

    Args:
        convnext_type (ConvNextType): ConvNeXt type.

    Returns:
        Callable: ConvNeXt constructor.
    """
    if convnext_type == ConvNextType.ATTO:
        return convenext_atto
    elif convnext_type == ConvNextType.FEMTO:
        return convenext_femto
    elif convnext_type == ConvNextType.PICO:
        return convenext_pico
    elif convnext_type == ConvNextType.NANO:
        return convenext_nano
    elif convnext_type == ConvNextType.TINY:
        return convenext_tiny
    elif convnext_type == ConvNextType.BASE:
        return convenext_base
    elif convnext_type == ConvNextType.LARGE:
        return convenext_large
    elif convnext_type == ConvNextType.HUGE:
        return convenext_huge
    elif convnext_type == ConvNextType.PATHETIC:
        # Not an endorsed ConvNeXt model, but useful for testing.
        return convenext_pathetic
    elif convnext_type == ConvNextType.VERY_PATHETIC:
        # Not an endorsed ConvNeXt model, but useful for testing.
        return convenext_very_pathetic
    else:
        raise ValueError(f"Unrecognized ConvNeXt type {convnext_type}.")
        

