from jax import numpy as jnp
from ml_collections import config_dict

from experiments.configs.datasets import add_galaxy_mnist_config
from src.transformations.transforms import AffineAndHSVWithoutShearTransform


def get_config(param) -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()
    config.num_trn = int(param)

    config.seed = 0

    config.eval_freq = 0.01

    config.transform_kwargs = {"order": 3}

    config.n_samples = 5
    config.x_mse_loss_mult = 1.0
    config.invertibility_loss_mult = 0.0
    config.η_loss_mult = 0.0
    config.steps = 10_000
    config.lr = 3e-4
    config.init_lr_mult = 1e-2
    config.final_lr_mult = 1e-4
    config.warmup_steps_pct = 0.1
    config.clip_norm = 2.0
    config.weight_decay = 1e-4
    config.σ_lr = 3e-3
    config.blur_filter_size = 5
    config.blur_σ_init = 3.0
    config.blur_end_pct = 0.01
    config.symmetrized_loss = True

    config.augment_bounds = (
        0.25,
        0.25,
        jnp.pi,
        0.25,
        0.25,
        0.5,
        2.31,
        0.51,
    )
    config.augment_offset = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
    )

    config.model_name = "inference_net"
    config.model = config_dict.ConfigDict()
    config.model.squash_to_bounds = True
    config.model.hidden_dims = (1024, 1024, 512, 256)
    config.model.transform = AffineAndHSVWithoutShearTransform

    # Dataset
    config.batch_size = 512
    config.dataset = "galaxy_mnist"
    config.num_val = 1000
    add_galaxy_mnist_config(
        config,
        num_trn=config.get("num_trn", None),
        num_val=config.num_val,
    )

    return config
