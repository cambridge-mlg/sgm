import math

from jax import numpy as jnp
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.seed = 0

    # Dataset
    config.batch_size = 512
    config.dataset = "aug_dsprites"

    config.train_split = ""  # Doesn't matter for augmentedDsprites
    config.val_split = ""  # Doesn't matter for augmentedDsprites

    config.pp_train = f'resize(30, "bicubic")|value_range(-1, 1, 0, 1)|move_key("label_shape", "label")|keep(["image", "label"])'
    config.pp_eval = f'resize(30, "bicubic")|value_range(-1, 1, 0, 1)|move_key("label_shape", "label")|keep(["image", "mask", "label"])'

    config.aug_dsprites = config_dict.ConfigDict()

    config.aug_dsprites.square_distribution = config_dict.ConfigDict()
    config.aug_dsprites.square_distribution.orientation = f"biuniform(low1=0.0, high1={math.pi / 8}, low2={math.pi / 4}, high2={4*math.pi / 5})"
    config.aug_dsprites.square_distribution.scale = (
        "biuniform(low1=0.45, high1=0.65, low2=0.89, high2=1.0)"
    )
    config.aug_dsprites.square_distribution.x_position = (
        "biuniform(low1=0.15, high1=0.6, low2=0.4, high2=0.9)"
    )
    config.aug_dsprites.square_distribution.y_position = (
        "truncated_normal(minval=0.05, maxval=0.95, loc=0.5, scale=0.15)"
    )

    config.aug_dsprites.ellipse_distribution = config_dict.ConfigDict()
    config.aug_dsprites.ellipse_distribution = (
        config.aug_dsprites.square_distribution.copy_and_resolve_references()
    )

    config.aug_dsprites.heart_distribution = config_dict.ConfigDict()
    config.aug_dsprites.heart_distribution.orientation = (
        f"biuniform(low1=0.0, high1={math.pi/2}, low2={math.pi}, high2={math.pi*1.25})"
    )
    config.aug_dsprites.heart_distribution.scale = (
        "truncated_normal(loc=0.8, scale=0.05, minval=0.45, maxval=1.05)"
    )
    config.aug_dsprites.heart_distribution.x_position = "uniform(low=0.1, high=0.8)"
    config.aug_dsprites.heart_distribution.y_position = (
        "biuniform(low1=0.0, high1=0.3, low2=0.5, high2=0.8)"
    )

    config.aug_dsprites.heart_distribution.unnormalised_shape_prob = 1 / 3
    config.aug_dsprites.square_distribution.unnormalised_shape_prob = 1 / 3
    config.aug_dsprites.ellipse_distribution.unnormalised_shape_prob = 1 / 3

    # Training and model

    config.eval_freq = 0.01
    config.n_samples = 5
    config.interpolation_order = 3
    config.translate_last = False
    config.x_mse_loss_mult = 1.0
    config.invertibility_loss_mult = 0.1
    config.η_loss_mult = 0.0

    config.steps = 30_000
    config.lr = 3e-4
    config.init_lr_mult = 1 / 3
    config.final_lr_mult = 1 / 90
    config.warmup_steps_pct = 0.1
    config.clip_norm = 2.0
    config.weight_decay = 1e-4
    config.σ_lr = 1e-2

    config.augment_warmup_steps_pct = 0.0  # No augmentation warmup
    # Blur schedule
    config.blur_filter_shape = (15, 15)
    config.blur_sigma_init = 1.0
    config.blur_sigma_decay_end_pct = 0.5

    config.augment_bounds = (0.5, 0.5, jnp.pi, 0.5, 0.5)
    config.augment_offset = (0.0, 0.0, 0.0, 0.0, 0.0)

    config.model_name = "inference_net"
    config.model = config_dict.ConfigDict()
    config.model.squash_to_bounds = False
    config.model.hidden_dims = (4096, 2048, 512, 128)

    return config
