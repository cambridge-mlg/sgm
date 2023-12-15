from typing import Callable, Optional, Sequence

from chex import Array
from flax import linen as nn


class MLP(nn.Module):
    """A simple MLP."""

    hidden_dims: Sequence[int]
    act_fn: Callable = nn.gelu
    norm_cls: nn.Module = nn.LayerNorm
    out_dim: Optional[int] = None
    use_norm: bool = True
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: Array, train: bool = False):
        """Applies the MLP to the input tensor."""

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = self.act_fn(x)
            if self.use_norm:
                x = self.norm_cls()(x)
            if self.dropout_rate > 0.0:
                x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)

        if self.out_dim is not None:
            x = nn.Dense(self.out_dim)(x)

        return x
