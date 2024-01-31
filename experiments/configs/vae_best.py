from jax import numpy as jnp
from ml_collections import config_dict

from experiments.configs.datasets import add_mnist_config


def get_config(params) -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    params = params.split(",")
    config.model_name = params[0]
    if len(params) > 1:
        config.angle = int(params[1])
        config.num_trn = int(params[2])

    config.seed = 0
    config.dataset = "MNIST"

    config.model = config_dict.ConfigDict()
    config.model.latent_dim = 20
    config.model.dense_dims = (256,)
    config.model.Z_given_X = config_dict.ConfigDict()
    config.model.Z_given_X.max_2strides = 2
    config.model.X_given_Z = config_dict.ConfigDict()
    config.model.X_given_Z.max_2strides = 2

    config.eval_freq = 0.01
    config.plot_freq = 0.25

    config.init_lr_mult = 0.03
    config.final_lr_mult = 0.0001
    config.weight_decay = 1e-4
    config.clip_norm = 2.0

    config.β_schedule_init_value = 1.0
    config.β_schedule_final_value = 1.0

    match (
        config.model_name,
        config.dataset,
        config.get("angle", None),
        config.get("num_trn", None),
        config.seed,
    ):
        case ("InvVAE", "MNIST", 0, 25_000, 0):
            config.lr = 9e-3
            config.model.conv_dims = (64, 128)
            config.steps = 5_000
            config.warmup_steps_pct = 0.2
        case ("InvVAE", "MNIST", 0, 37_500, 0):
            config.lr = 9e-3
            config.model.conv_dims = (64, 128)
            config.steps = 5_000
            config.warmup_steps_pct = 0.15
        case ("InvVAE", "MNIST", 0, 50_000, 0):
            config.lr = 9e-3
            config.model.conv_dims = (64, 128, 256)
            config.steps = 5_000
            config.warmup_steps_pct = 0.2
        case ("InvVAE", "MNIST", 15, 25_000, 0):
            config.lr = 9e-3
            config.model.conv_dims = (64, 128)
            config.steps = 5_000
            config.warmup_steps_pct = 0.15
        case ("InvVAE", "MNIST", 15, 37_500, 0):
            config.lr = 6e-3
            config.model.conv_dims = (64, 128)
            config.steps = 10_000
            config.warmup_steps_pct = 0.15
        case ("InvVAE", "MNIST", 15, 50_000, 0):
            config.lr = 6e-3
            config.model.conv_dims = (64, 128)
            config.steps = 20_000
            config.warmup_steps_pct = 0.15
        case ("InvVAE", "MNIST", 90, 25_000, 0):
            config.lr = 9e-3
            config.model.conv_dims = (64, 128)
            config.steps = 5_000
            config.warmup_steps_pct = 0.15
        case ("InvVAE", "MNIST", 90, 37_500, 0):
            config.lr = 9e-3
            config.model.conv_dims = (64, 128)
            config.steps = 10_000
            config.warmup_steps_pct = 0.2
        case ("InvVAE", "MNIST", 90, 50_000, 0):
            config.lr = 9e-3
            config.model.conv_dims = (64, 128)
            config.steps = 20_000
            config.warmup_steps_pct = 0.15
        case ("InvVAE", "MNIST", 180, 25_000, 0):
            config.lr = 9e-3
            config.model.conv_dims = (64, 128)
            config.steps = 5_000
            config.warmup_steps_pct = 0.15
        case ("InvVAE", "MNIST", 180, 37_500, 0):
            config.lr = 9e-3
            config.model.conv_dims = (64, 128)
            config.steps = 10_000
            config.warmup_steps_pct = 0.2
        case ("InvVAE", "MNIST", 180, 50_000, 0):
            config.lr = 6e-3
            config.model.conv_dims = (64, 128)
            config.steps = 10_000
            config.warmup_steps_pct = 0.2

    config.run_iwlb = False
    config.iwlb_num_samples = 300

    config.batch_size = 512
    config.batch_size_eval = 50
    config.num_val = 10000
    add_mnist_config(
        config,
        angle=config.angle,
        num_trn=config.get("num_trn", None),
        num_val=config.num_val,
    )
    config.test_split = ""

    return config
