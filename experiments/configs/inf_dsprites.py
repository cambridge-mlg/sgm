from jax import numpy as jnp
from ml_collections import config_dict

from experiments.configs.datasets import add_aug_dsprites_config


def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.seed = 0

    config.eval_freq = 0.01

    config.interpolation_order = 3
    config.translate_last = False

    config.n_samples = 5
    config.x_mse_loss_mult = 1.0
    config.invertibility_loss_mult = 1.0
    config.η_loss_mult = 0.0
    config.steps = 60_000
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

    config.augment_bounds = (0.5, 0.5, jnp.pi, 0.5, 0.5)
    config.augment_offset = (0.0, 0.0, 0.0, 0.0, 0.0)

    config.model_name = "inference_net"
    config.model = config_dict.ConfigDict()
    config.model.squash_to_bounds = False
    config.model.hidden_dims = (2048, 1024, 512, 256)

    # Dataset
    config.batch_size = 512
    config.dataset = "aug_dsprites"
    add_aug_dsprites_config(config)

    return config
