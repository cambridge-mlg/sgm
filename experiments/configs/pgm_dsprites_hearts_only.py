from ml_collections import config_dict
from jax import numpy as jnp
import math

import numpy as np

from src.utils.datasets.augmented_dsprites import DistributionConfig


def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.seed = 0

    # --- Dataset ---
    config.batch_size = 128
    config.dataset = "aug_dsprites"

    config.train_split = ""  # Doesn't matter for augmentedDsprites
    config.val_split = ""  # Doesn't matter for augmentedDsprites

    # config.pp_train = f'value_range(-1, 1, 0, 1)|resize(28, 28)|move_key("label_shape", "label")|keep(["image", "label"])'
    # config.pp_eval = f'value_range(-1, 1, 0, 1)|resize(28, 28)|move_key("label_shape", "label")|keep(["image", "mask", "label"])'
    config.pp_train = f'value_range(-1, 1, 0, 1)|move_key("label_shape", "label")|keep(["image", "label"])'
    config.pp_eval = f'value_range(-1, 1, 0, 1)|move_key("label_shape", "label")|keep(["image", "mask", "label"])'

    # Dsprites specific settings:
    # Only include the heart examples.
    config.aug_dsprites = config_dict.ConfigDict()
    config.aug_dsprites.heart_distribution = config_dict.ConfigDict()

    # config.aug_dsprites.heart_distribution.orientation = DistributionConfig(
    #     "delta", {"value": 0.0}
    # )
    config.aug_dsprites.heart_distribution.orientation = (
        f"uniform(low=0.0, high={2*np.pi})"
    )
    # config.aug_dsprites.heart_distribution.orientation = config_dict.ConfigDict()
    # config.aug_dsprites.heart_distribution.orientation.type = "uniform"
    # config.aug_dsprites.heart_distribution.orientation.kwargs = config_dict.ConfigDict()
    # config.aug_dsprites.heart_distribution.orientation.kwargs.low = 0.0
    # config.aug_dsprites.heart_distribution.orientation.kwargs.high = 0.0
    # config.aug_dsprites.heart_distribution.orientation.kwargs = dict(low=0.0, high=2 * math.pi)

    config.aug_dsprites.heart_distribution.scale = DistributionConfig(
        "delta", {"value": 1.0}
    )
    config.aug_dsprites.heart_distribution.x_position = DistributionConfig(
        "delta", {"value": np.linspace(0.0, 1.0, 32, dtype=np.float64)[16]}
    )
    config.aug_dsprites.heart_distribution.y_position = DistributionConfig(
        "delta", {"value": np.linspace(0.0, 1.0, 32, dtype=np.float64)[16]}
    )

    config.aug_dsprites.ellipse_distribution = config_dict.ConfigDict()
    config.aug_dsprites.ellipse_distribution = (
        config.aug_dsprites.heart_distribution.copy_and_resolve_references()
    )  # Same distributions
    config.aug_dsprites.square_distribution = config_dict.ConfigDict()
    config.aug_dsprites.square_distribution = (
        config.aug_dsprites.heart_distribution.copy_and_resolve_references()
    )  # Same distributions

    # Shape probabilities:
    config.aug_dsprites.heart_distribution.shape_prob = 1.0
    config.aug_dsprites.square_distribution.shape_prob = 0.0
    config.aug_dsprites.ellipse_distribution.shape_prob = 0.0

    # --- Training ---
    config.n_samples = 5
    config.symmetrised_samples_in_loss = True
    config.eval_freq = 0.01
    config.difficulty_weighted_inf_loss = False
    config.double_transform = False
    config.interpolation_order = 1

    config.gen_steps = 10_000
    config.gen_init_lr = 1e-4
    config.gen_peak_lr_mult = 3
    config.gen_final_lr_mult = 1 / 30
    config.gen_warmup_steps = 1_000

    config.σ_lr = 1e-2
    config.inf_steps = 30_000
    config.inf_lr = 4e-3
    config.inf_init_lr_mult = 1e-3
    config.inf_final_lr_mult = 1e-1
    config.inf_warmup_steps = 3_000  # roughly 10% of training

    config.inf_weight_decay = 0.05

    # config.α_schedule_pct = 1.
    # config.α_schedule_final_value = 1.
    # config.β_schedule_pct = 1.0
    # config.β_schedule_final_value = 1.0
    config.η_loss_mult_peak = 1.0
    config.η_loss_decay_start = 1.0
    config.η_loss_decay_end = 1.0
    config.augment_warmup_end = 0.2

    config.x_mse_loss_mult = 0.0
    config.invertibility_loss_mult = 0.0

    # Separated out model bounds, and augmentation bounds for augmentations sampled for the
    # training objective:
    config.augment_bounds = (0.0, 0.0, jnp.pi, 0.0, 0.0)
    config.augment_offset = (0.0, 0.0, 0.0, 0.0, 0.0)

    config.model = config_dict.ConfigDict()
    config.model.inference = config_dict.ConfigDict()
    config.model.inference.model_type = "mlp"
    config.model.inference.convnext_type = "atto"

    config.model.inference.bounds = (0.5, 0.5, jnp.pi, 0.5, 0.5)
    config.model.inference.offset = (0.0, 0.0, 0.0, 0.0, 0.0)
    config.model.inference.hidden_dims = (4096, 2048, 512, 128)
    config.model.inference.squash_to_bounds = False
    # config.model.inference.σ_init = 1.
    config.model.generative = config_dict.ConfigDict()
    config.model.generative.bounds = (0.25, 0.25, jnp.pi, 0.25, 0.25)
    config.model.generative.offset = (0.0, 0.0, 0.0, 0.0, 0.0)
    config.model.generative.hidden_dims = (1024, 512, 256)
    config.model.generative.num_flows = 2
    config.model.generative.num_bins = 4
    config.model.generative.conditioner = config_dict.ConfigDict()
    config.model.generative.conditioner.hidden_dims = (256, 256)

    return config
