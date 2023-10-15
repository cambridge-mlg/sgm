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

    config.train_split = ""  # Doesn't matter for augmentedDsprites
    config.val_split = ""  # Doesn't matter for augmentedDsprites

    config.pp_train = f'value_range(-1, 1, 0, 1)|move_key("label_shape", "label")|keep(["image", "label"])'
    config.pp_eval = f'value_range(-1, 1, 0, 1)|move_key("label_shape", "label")|keep(["image", "mask", "label"])'

    # Dsprites specific settings:
    config.aug_dsprites = config_dict.ConfigDict()

    config.aug_dsprites.square_distribution = config_dict.ConfigDict()

    config.aug_dsprites.square_distribution.orientation = f"uniform(low=0.0, high={math.pi / 2})"
    config.aug_dsprites.square_distribution.scale = "uniform(low=0.65, high=1.0)"
    config.aug_dsprites.square_distribution.x_position = "biuniform(low1=0.15, high1=0.6, low2=0.4, high2=0.9)"
    config.aug_dsprites.square_distribution.y_position = "truncated_normal(minval=0.05, maxval=0.95, loc=0.5, scale=0.15)"

    config.aug_dsprites.ellipse_distribution = config_dict.ConfigDict()
    config.aug_dsprites.ellipse_distribution = (
        config.aug_dsprites.square_distribution.copy_and_resolve_references()
    )  # Same distributions

    config.aug_dsprites.heart_distribution = config_dict.ConfigDict()

    config.aug_dsprites.heart_distribution.orientation = f"uniform(low=0.0, high={math.pi/2})"
    config.aug_dsprites.heart_distribution.scale = "uniform(low=0.75, high=1.0)"
    config.aug_dsprites.heart_distribution.x_position = "uniform(low=0.1, high=0.8)"
    config.aug_dsprites.heart_distribution.y_position = "biuniform(low1=0.0, high1=0.3, low2=0.5, high2=0.8)"
    # Shape probabilities:
    config.aug_dsprites.heart_distribution.unnormalised_shape_prob = 1 / 3
    config.aug_dsprites.square_distribution.unnormalised_shape_prob = 1 / 3
    config.aug_dsprites.ellipse_distribution.unnormalised_shape_prob = 1 / 3

    # --- Training (proto) ---
    config.n_samples = 5
    config.eval_freq = 0.01
    config.difficulty_weighted_inf_loss = False
    config.symmetrised_samples_in_loss = True
    config.interpolation_order = 3
    config.translate_last = False


    config.σ_lr = 1e-2
    config.inf_steps = 10_000
    config.inf_lr = 4e-3
    config.inf_init_lr_mult = 1e-3
    config.inf_final_lr_mult = 1e-1
    config.inf_warmup_steps = 1_000  # roughly 10% of training

    config.inf_weight_decay = 0.05

    config.η_loss_mult_peak = 0.0
    config.η_loss_decay_start = 0.0
    config.η_loss_decay_end = 1.0
    config.augment_warmup_end = 0.0

    # Training (gen)
    config.gen_steps = 10_000
    config.gen_lr = 2.7e-3
    config.gen_init_lr_mult = 1 / 9
    config.gen_final_lr_mult = 1 / 2700
    config.gen_warmup_steps = 1_000

    config.mae_loss_mult_initial = 0.0
    config.mae_loss_mult_final = 1.0

    # Blur schedule
    config.blur_filter_shape = (21, 21)
    config.blur_sigma_init = 1.0
    config.blur_sigma_decay_end = 0.5

    config.x_mse_loss_mult = 1.0
    config.invertibility_loss_mult = 1.0

    # Separated out model bounds, and augmentation bounds for augmentations sampled for the
    # training objective:
    config.augment_bounds = (.5, .5, jnp.pi, .5, .5)
    config.augment_offset = (0.0, 0.0, 0.0, 0.0, 0.0)
    config.eval_augment_bounds = (.5, .5, jnp.pi, .5, .5)

    config.model = config_dict.ConfigDict()
    config.model.inference = config_dict.ConfigDict()
    config.model.inference.model_type = "mlp"
    config.model.inference.use_layernorm = True
    # config.model.inference.model_type = "convnext"
    # config.model.inference.convnext_type = "tiny"
    config.model.inference.bounds = (0.5, 0.5, jnp.pi, 0.5, 0.5)
    config.model.inference.offset = (0.0, 0.0, 0.0, 0.0, 0.0)
    config.model.inference.hidden_dims = (4096, 2048, 512, 128)
    config.model.inference.squash_to_bounds = False

    config.model.generative = config_dict.ConfigDict()
    config.model.generative.bounds = (0.25, 0.25, jnp.pi, 0.25, 0.25)
    config.model.generative.offset = (0.0, 0.0, 0.0, 0.0, 0.0)
    config.model.generative.hidden_dims = (1024, 512, 256)
    config.model.generative.squash_to_bounds = False
    config.model.generative.num_flows = 2
    config.model.generative.num_bins = 4
    config.model.generative.conditioner = config_dict.ConfigDict()
    config.model.generative.conditioner.hidden_dims = (256, 256)

    return config
