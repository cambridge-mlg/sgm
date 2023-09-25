from typing import Any, Callable, Optional, Sequence, Tuple
from functools import partial

from jax import numpy as jnp
from chex import Array
from flax import linen as nn


class BasicBlock(nn.Module):
    """Basic ResNet block."""

    filters: int
    conv: nn.Module
    act_fn: Callable = nn.relu
    norm_cls: nn.Module = nn.LayerNorm
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x: Array):
        """Applies the basic block to the input tensor."""

        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm_cls()(y)
        y = self.act_fn(y)

        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm_cls(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
                residual
            )
            residual = self.norm_cls(name="norm_proj")(residual)

        return self.act_fn(residual + y)


class BottleneckBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    conv: nn.Module
    act_fn: Callable = nn.relu
    norm_cls: nn.Module = nn.LayerNorm
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x: Array):
        """Applies the bottleneck block to the input tensor."""

        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm_cls(y)
        y = self.act_fn(y)

        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm_cls(y)
        y = self.act_fn(y)

        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm_cls(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.norm_cls(name="norm_proj")(residual)

        return self.act_fn(residual + y)


class ResNet(nn.Module):
    """ResNetV1."""

    stage_sizes: Sequence[int]
    block_cls: nn.Module
    num_classes: Optional[int] = None
    num_filters: int = 64
    dtype: Any = jnp.float32
    act_fn: Callable = nn.relu
    conv_cls: nn.Module = nn.Conv
    norm_cls: nn.Module = nn.LayerNorm
    lowres: bool = False
    head: bool = True

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv_cls, use_bias=False, dtype=self.dtype)

        if self.lowres:
            x = conv(
                self.num_filters,
                (3, 3),
                (1, 1),
                padding=[(1, 1), (1, 1)],
                name="conv_init",
            )(x)
        else:
            x = conv(
                self.num_filters,
                (7, 7),
                (2, 2),
                padding=[(3, 3), (3, 3)],
                name="conv_init",
            )(x)
        x = self.norm_cls(name="norm_init")(x)
        x = self.act_fn(x)
        x = (
            nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
            if not self.lowres
            else x
        )
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv=conv,
                    norm_cls=self.norm_cls,
                    act_fn=self.act_fn,
                )(x)
        x = jnp.mean(x, axis=(0, 1))
        if not self.head:
            return x

        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=BasicBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BasicBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckBlock)
