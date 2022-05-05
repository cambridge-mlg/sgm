from flax.core.frozen_dict import FrozenDict
from flax.training import train_state


class TrainState(train_state.TrainState):
    """A Flax TrainState which also tracks model state such as BN state."""
    model_state: FrozenDict
