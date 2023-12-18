from jax import numpy as jnp
from ml_collections import config_dict

from experiments.configs.datasets import add_aug_dsprites_config, add_mnist_config


def get_config(params) -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    params = params.split(",")
    config.dataset = params[0]
    assert config.dataset in ["MNIST", "aug_dsprites"]
    config.seed = int(params[1])
    if len(params) > 2:
        config.angle = float(params[2])
        config.num_trn = int(params[3])

    config.eval_freq = 0.01
    config.checkpoint = ""

    config.interpolation_order = 3
    config.translate_last = False

    config.n_samples = 5
    config.x_mse_loss_mult = 1.0
    config.invertibility_loss_mult = 0.1 if config.dataset == "MNIST" else 1.0
    config.η_loss_mult = 0.0
    config.steps = 60_000
    config.σ_lr = 3e-3
    config.blur_filter_size = 5
    config.blur_end_pct = 0.01
    config.weight_decay = 1e-4

    config.augment_bounds = (
        (0.25, 0.25, jnp.pi, 0.25, 0.25)
        if config.dataset == "MNIST"
        else (0.5, 0.5, jnp.pi, 0.5, 0.5)
    )
    config.augment_offset = (0.0, 0.0, 0.0, 0.0, 0.0)

    config.model_name = "inference_net"
    config.model = config_dict.ConfigDict()
    config.model.squash_to_bounds = False
    config.model.hidden_dims = (2048, 1024, 512, 256)

    match (
        config.dataset,
        config.get("angle", None),
        config.get("num_trn", None),
        config.seed,
    ):
        case ("MNIST", 0.0, 50_000, 0) | ("MNIST", 0.0, 37_500, 0):
            # ^ xegejvhb | 59c3uexk
            config.blur_σ_init = 0.0
            config.clip_norm = 10.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 3e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.05
        case ("MNIST", 0.0, 50_000, 1):  # vdfhrd8g
            config.blur_σ_init = 0.0
            config.clip_norm = 10.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 1e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.2
        case ("MNIST", 0.0, 50_000, 2):  # vsnpg2te
            config.blur_σ_init = 0.0
            config.clip_norm = 10.0
            config.final_lr_mult = 3e-4
            config.init_lr_mult = 1e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.05
        case ("MNIST", 0.0, 37_500, 1):  # y6vxr0yb
            config.blur_σ_init = 3.0
            config.clip_norm = 10.0
            config.final_lr_mult = 3e-4
            config.init_lr_mult = 3e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.2
        case ("MNIST", 0.0, 37_500, 2):  # smpwgumg
            config.blur_σ_init = 3.0
            config.clip_norm = 3.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 1e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.05
        case ("MNIST", 0.0, 25_000, 0):  # kq36fzu9
            config.blur_σ_init = 0.0
            config.clip_norm = 10.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 1e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.1
        case ("MNIST", 0.0, 25_000, 1):  # 38kq6q8f
            config.blur_σ_init = 3.0
            config.clip_norm = 10.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 3e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.1
        case ("MNIST", 0.0, 25_000, 2):  # s7pj0ucn
            config.blur_σ_init = 0.0
            config.clip_norm = 3.0
            config.final_lr_mult = 3e-4
            config.init_lr_mult = 1e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.05
        case ("aug_dsprites", None, None, 0):  # co5jijn1
            config.blur_σ_init = 3.0
            config.clip_norm = 3.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 3e-2
            config.lr = 1e-3
            config.warmup_steps_pct = 0.1
        case ("aug_dsprites", None, None, 1):  # lcr9fjj6
            config.blur_σ_init = 0.0
            config.clip_norm = 10.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 1e-2
            config.lr = 1e-3
            config.warmup_steps_pct = 0.1
        case ("aug_dsprites", None, None, 2):  # sup5y2kj
            config.blur_σ_init = 0.0
            config.clip_norm = 3.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 3e-2
            config.lr = 1e-3
            config.warmup_steps_pct = 0.2

    config.batch_size = 512
    if config.dataset == "MNIST":
        config.num_val = 10000
        add_mnist_config(
            config,
            angle=config.angle,
            num_trn=config.get("num_trn", None),
            num_val=config.num_val,
        )
    else:
        add_aug_dsprites_config(config)

    return config
