from ml_collections import config_dict
from jax import numpy as jnp
import math

from src.utils.datasets.augmented_dsprites import DistributionConfig


def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.seed = 0

    # --- Dataset ---
    config.batch_size = 512
    config.dataset = "aug_dsprites"
    config.shuffle_buffer_size = 1  # Doesn't matter for augmentedDsprites
    config.repeat_after_batching = False  # Doesn't matter for augmentedDsprites

    config.train_split = ""  # Doesn't matter for augmentedDsprites
    config.val_split = ""  # Doesn't matter for augmentedDsprites

    config.pp_train = f'value_range(-1, 1)|keep(["image", "mask"])'
    config.pp_eval = f'value_range(-1, 1)|keep(["image", "mask", "label_shape"])'

    # Dsprites specific settings:
    config.aug_dsprites = config_dict.ConfigDict()

    config.aug_dsprites.square_distribution = config_dict.ConfigDict()

    config.aug_dsprites.square_distribution.orientation = DistributionConfig(
        "uniform", {"low": 0.0, "high": math.pi / 4}
    )
    config.aug_dsprites.square_distribution.scale = DistributionConfig(
        "uniform",
        {"low": 0.8, "high": 1.0},
    )
    config.aug_dsprites.square_distribution.x_position = DistributionConfig(
        "biuniform",
        {"low1": 0.2, "high1": 0.6, "low2": 0.4, "high2": 0.8},
    )
    config.aug_dsprites.square_distribution.y_position = DistributionConfig(
        "truncated_normal",
        {"minval": 0.05, "maxval": 0.95, "loc": 0.5, "scale": 0.15},
    )

    config.aug_dsprites.ellipse_distribution = config_dict.ConfigDict()
    config.aug_dsprites.ellipse_distribution = (
        config.aug_dsprites.square_distribution
    )  # Same distributions

    config.aug_dsprites.heart_distribution = config_dict.ConfigDict()

    config.aug_dsprites.heart_distribution.orientation = DistributionConfig(
        "uniform", {"low": 0.0, "high": 2 * math.pi}
    )
    config.aug_dsprites.heart_distribution.scale = DistributionConfig(
        "uniform",
        {"low": 0.5, "high": 1.0},
    )
    config.aug_dsprites.heart_distribution.x_position = DistributionConfig(
        "uniform",
        {"low": 0.2, "high": 0.8},
    )
    config.aug_dsprites.heart_distribution.y_position = DistributionConfig(
        "biuniform",
        {
            "low1": 0.0,
            "high1": 0.3,
            "low2": 0.5,
            "high2": 0.8,
        },
    )

    # --- Training ---
    config.n_samples = 5
    config.eval_freq = 0.01
    config.difficulty_weighted_inf_loss = False

    config.gen_steps = 10_000
    config.gen_init_lr = 1e-4
    config.gen_peak_lr_mult = 3
    config.gen_final_lr_mult = 1 / 30
    config.gen_warmup_steps = 1_000
    config.σ_lr = 1e-2
    config.inf_steps = 10_000
    config.inf_init_lr = 3e-4
    config.inf_peak_lr_mult = 9
    config.inf_final_lr_mult = 1 / 300
    config.inf_warmup_steps = 1_000

    config.α_schedule_pct = 0.25
    config.α_schedule_final_value = 0.0
    config.β_schedule_pct = 1.0
    config.β_schedule_final_value = 0.99

    config.model = config_dict.ConfigDict()
    config.model.bounds = (0.25, 0.25, jnp.pi, 0.25, 0.25)
    config.model.offset = (0.0, 0.0, 0.0, 0.0, 0.0)
    config.model.inference = config_dict.ConfigDict()
    config.model.inference.hidden_dims = (2048, 1024, 512, 128)
    config.model.generative = config_dict.ConfigDict()
    config.model.generative.hidden_dims = (1024, 512, 256)
    config.model.generative.num_flows = 2
    config.model.generative.num_bins = 4
    config.model.generative.conditioner = config_dict.ConfigDict()
    config.model.generative.conditioner.hidden_dims = (256, 256)

    return config
