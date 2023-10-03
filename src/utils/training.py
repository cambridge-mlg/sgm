from typing import Tuple

from ciclo import Elapsed, Logs, LoopCallbackBase, LoopState
from ciclo.types import S
from ciclo.utils import is_scalar
from matplotlib.figure import Figure
from wandb.wandb_run import Run

import wandb

CallbackOutput = Tuple[Logs, S]


class custom_wandb_logger(LoopCallbackBase[S]):
    def __init__(self, run: Run):
        from wandb.wandb_run import Run

        self.run: Run = run

    def __call__(self, elapsed: Elapsed, logs: Logs) -> None:
        data = {}
        for collection, collection_logs in logs.items():
            for key, value in collection_logs.items():
                if collection in ["metrics", "stateful_metrics"]:
                    if key.endswith("_test"):
                        key = f"valid/{key[:-5]}"
                    else:
                        key = f"train/{key}"

                if key in data:
                    # we make the key unique by including the collection name, however we want the collection name to be
                    # after the prefix of train/ or valid/ (if there is one) so that the graphs are grouped together.
                    components = key.split("/", 1)
                    prefix = f"{components[0]}/" if len(components) > 1 else ""
                    key = f"{prefix}{collection}_{components[-1]}"

                if is_scalar(value):
                    data[key] = value

                if isinstance(value, Figure):
                    data[key] = wandb.Image(value)

        if len(data) > 0:
            self.run.log(data, step=elapsed.steps)

    def __loop_callback__(self, loop_state: LoopState[S]) -> CallbackOutput[S]:
        self(loop_state.elapsed, loop_state.logs)
        return Logs(), loop_state.state
