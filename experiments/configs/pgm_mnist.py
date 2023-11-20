from jax import numpy as jnp
from ml_collections import config_dict


def get_config(params) -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    params = params.split(",")
    config.angle = float(params[0])
    config.num_val = 10000
    if len(params) > 1:
        config.num_trn = int(params[1])
        end_index = config.num_trn + config.num_val
    else:
        end_index = ""

    config.seed = 0

    config.batch_size = 512
    config.dataset = "MNIST"
    config.shuffle_buffer_size = 50_000
    config.shuffle = "preprocessed"
    config.repeat_after_batching = (
        True  # NOTE: ordering of PP, shuffle, and repeat is important!
    )
    config.n_samples = 5
    config.eval_freq = 0.01
    config.difficulty_weighted_loss = True
    config.interpolation_order = 3
    config.translate_last = False
    config.symmetrised_samples_in_loss = False
    config.x_mse_loss_mult = 1.0
    config.invertibility_loss_mult = 0.1

    config.inf_steps = 15_000
    config.inf_lr = 3e-4
    config.inf_init_lr_mult = 1 / 3
    config.inf_final_lr_mult = 1 / 90
    config.inf_warmup_steps = config.inf_steps // 10
    config.inf_clip_norm = 2.0
    config.inf_weight_decay = 1e-4
    config.σ_lr = 1e-2
    # Schedule of the loss in the η space (rather than the x_mse space) for the "inference" model
    config.η_loss_mult_peak = 0.0  # No η loss
    config.η_loss_decay_end = 0.0
    config.η_loss_decay_start = 0.0
    config.augment_warmup_end = 0.0  # No augmentation warmup
    # Blur schedule
    config.blur_filter_shape = (15, 15)
    config.blur_sigma_init = 1.0
    config.blur_sigma_decay_end = 0.5

    # Linearly increase MAE loss mult. from 0 to 1 (pairwise diffs between log-likelihoods for augmented samples)
    config.mae_loss_mult_initial = 0.0
    config.mae_loss_mult_final = 1.0
    config.augment_bounds = (0.25, 0.25, jnp.pi, 0.25, 0.25)
    config.augment_offset = (0.0, 0.0, 0.0, 0.0, 0.0)
    config.eval_augment_bounds = (0.25, 0.25, jnp.pi, 0.25, 0.25)

    config.gen_steps = 10_000
    config.gen_lr = 2.7e-3
    config.gen_init_lr_mult = 1 / 9
    config.gen_final_lr_mult = 1 / 2700
    config.gen_warmup_steps = config.gen_steps // 10
    config.gen_clip_norm = 2.0
    config.gen_weight_decay = 1e-4

    config.train_split = f"train[{config.num_val}:{end_index}]"
    config.pp_train = f'value_range(-1, 1)|random_rotate(-{config.angle}, {config.angle}, fill_value=-1)|keep(["image", "label"])'
    config.val_split = f"train[:{config.num_val}]"
    config.pp_eval = f'value_range(-1, 1)|random_rotate(-{config.angle}, {config.angle}, fill_value=-1)|keep(["image", "label"])'

    config.model_name = "PGM"
    config.model = config_dict.ConfigDict()
    config.model.inference = config_dict.ConfigDict()
    config.model.inference.offset = config.augment_offset
    config.model.inference.bounds = config.augment_bounds
    config.model.inference.squash_to_bounds = False
    config.model.inference.hidden_dims = (1024, 512, 256, 128)

    config.model.generative = config_dict.ConfigDict()
    config.model.generative.bounds = config.augment_bounds
    config.model.generative.offset = config.augment_offset
    config.model.generative.hidden_dims = (1024, 512, 256)
    config.model.generative.num_flows = 2
    config.model.generative.num_bins = 4
    config.model.generative.conditioner = config_dict.ConfigDict()
    config.model.generative.conditioner.hidden_dims = (256, 256)
    config.model.generative.squash_to_bounds = config.model.inference.squash_to_bounds

    return config
