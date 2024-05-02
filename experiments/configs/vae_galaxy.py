from jax import numpy as jnp
from ml_collections import config_dict

from experiments.configs.datasets import add_galaxy_mnist_config


def get_config(param) -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()
    config.num_trn = int(param)

    config.seed = 0

    config.model_name = "VAE"
    config.model = config_dict.ConfigDict()
    config.model.latent_dim = 16
    config.model.conv_dims = (64, 128, 256)
    config.model.dense_dims = (256,)
    config.model.Z_given_X = config_dict.ConfigDict()
    config.model.Z_given_X.max_2strides = 2
    config.model.X_given_Z = config_dict.ConfigDict()
    config.model.X_given_Z.max_2strides = 2

    config.steps = 10_000
    config.eval_freq = 0.01
    config.plot_freq = 0.25

    config.lr = 6e-3
    config.init_lr_mult = 0.03
    config.final_lr_mult = 0.0001
    config.warmup_steps_pct = 0.1
    config.weight_decay = 1e-4
    config.clip_norm = 2.0

    config.β_schedule_init_value = 1.0
    config.β_schedule_final_value = 1.0

    config.run_iwlb = False
    config.iwlb_num_samples = 300

    config.batch_size = 512
    config.batch_size_eval = 50
    config.dataset = "galaxy_mnist"
    config.num_val = 1000
    add_galaxy_mnist_config(
        config,
        num_trn=config.get("num_trn", None),
        num_val=config.num_val,
    )
    config.test_split = ""

    return config
