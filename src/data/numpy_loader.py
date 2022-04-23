"""A PyTorch dataloader that returns `np.ndarray` batches.

Taken from: colab.research.google.com/github/google/jax/blob/main/docs/notebooks/Neural_Network_and_Data_Loading.ipynb
"""
import numpy as np
from torch.utils import data


def _numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [_numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(data.DataLoader):
    """A PyTorch dataloader that returns `np.ndarray` batches, which can be used with Jax!
    """
    def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            generator=None
        ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=_numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )