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
    config.steps = 60_000
    config.lr = 3e-3
    config.init_lr_mult = 1e-2
    config.final_lr_mult = 1e-4
    config.warmup_steps_pct = 0.1
    config.clip_norm = 2.0
    config.weight_decay = 1e-4
    config.mae_loss_mult_initial = 0.0
    config.mae_loss_mult_final = 1.0

    config.augment_bounds = (0.5, 0.5, jnp.pi, 0.5, 0.5)
    config.augment_offset = (0.0, 0.0, 0.0, 0.0, 0.0)

    config.model_name = "generative_net"
    config.model = config_dict.ConfigDict()
    config.model.hidden_dims = (2048, 512, 512)
    config.model.squash_to_bounds = False
    config.model.num_flows = 6
    config.model.num_bins = 6
    config.model.conditioner = config_dict.ConfigDict()
    config.model.conditioner.hidden_dims = (256, 256)

    config.batch_size = 512
    config.dataset = "aug_dsprites"
    add_aug_dsprites_config(config)

    return config
