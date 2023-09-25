import math

from jax import numpy as jnp
from ml_collections import config_dict

from src.utils.datasets.augmented_dsprites import DistributionConfig


def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.seed = 0

    # --- Dataset ---
    config.batch_size = 512
    config.dataset = "aug_dsprites"
    # config.num_val_examples = 2048

    config.train_split = ""  # Doesn't matter for augmentedDsprites
    config.val_split = ""  # Doesn't matter for augmentedDsprites

    config.pp_train = f'value_range(-1, 1, 0, 1)|move_key("label_shape", "label")|keep(["image", "label"])'
    config.pp_eval = f'value_range(-1, 1, 0, 1)|move_key("label_shape", "label")|keep(["image", "mask", "label"])'

    # Dsprites specific settings:
    config.aug_dsprites = config_dict.ConfigDict()

    config.aug_dsprites.square_distribution = config_dict.ConfigDict()

    config.aug_dsprites.square_distribution.orientation = DistributionConfig(
        "uniform", {"low": 0.0, "high": math.pi / 4}
    )
    config.aug_dsprites.square_distribution.scale = DistributionConfig(
        "uniform",
        {"low": 0.98, "high": 1.0},
    )
    config.aug_dsprites.square_distribution.x_position = DistributionConfig(
        "uniform",
        {"low": 0.48, "high": 0.5},
    )
    config.aug_dsprites.square_distribution.y_position = DistributionConfig(
        "uniform",
        {"low": 0.48, "high": 0.5},
    )

    config.aug_dsprites.ellipse_distribution = config_dict.ConfigDict()
    config.aug_dsprites.ellipse_distribution = (
        config.aug_dsprites.square_distribution
    )  # Same distributions

    config.aug_dsprites.heart_distribution = config_dict.ConfigDict()
    config.aug_dsprites.heart_distribution = (
        config.aug_dsprites.square_distribution
    )  # Same distributions
    config.aug_dsprites.heart_distribution.orientation = DistributionConfig(
        "uniform", {"low": 0.0, "high": math.pi * 2}
    )

    # config.aug_dsprites.heart_distribution.shape_prob = 1.0
    # config.aug_dsprites.square_distribution.shape_prob = 0.0
    # config.aug_dsprites.ellipse_distribution.shape_prob = 0.0

    # --- Training ---
    config.n_samples = 5
    config.eval_freq = 0.01
    config.difficulty_weighted_inf_loss = True

    config.inf_steps = 20_000
    config.inf_init_lr = 1e-4
    config.inf_peak_lr_mult = 3
    config.inf_final_lr_mult = 1 / 30
    config.inf_warmup_steps = 1_000
    config.σ_lr = 1e-2
    config.gen_steps = 5_000
    config.gen_init_lr = 3e-4
    config.gen_peak_lr_mult = 9
    config.gen_final_lr_mult = 1 / 300
    config.gen_warmup_steps = 1_000

    config.α_schedule_pct = 0.25
    config.α_schedule_final_value = 0.0
    config.β_schedule_pct = 1.0
    config.β_schedule_final_value = 0.99

    config.model = config_dict.ConfigDict()
    config.model.bounds = (0.01, 0.01, jnp.pi, 0.5, 0.5)
    config.model.offset = (0.0, 0.0, 0.0, 0.0, 0.0)
    config.model.inference = config_dict.ConfigDict()
    config.model.inference.hidden_dims = (1024, 512, 256, 128)
    config.model.generative = config_dict.ConfigDict()
    config.model.generative.hidden_dims = (1024, 512, 256)
    config.model.generative.num_flows = 2
    config.model.generative.num_bins = 4
    config.model.generative.conditioner = config_dict.ConfigDict()
    config.model.generative.conditioner.hidden_dims = (256, 256)

    return config
