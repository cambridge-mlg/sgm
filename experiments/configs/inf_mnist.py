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
    config.eval_freq = 0.01
    config.n_samples = 5
    config.interpolation_order = 3
    config.translate_last = False
    config.x_mse_loss_mult = 1.0
    config.invertibility_loss_mult = 0.1
    config.η_loss_mult = 0.0

    config.steps = 15_000
    config.lr = 3e-4
    config.init_lr_mult = 1 / 3
    config.final_lr_mult = 1 / 90
    config.warmup_steps_pct = 0.1
    config.clip_norm = 2.0
    config.weight_decay = 1e-4
    config.σ_lr = 1e-2

    # Blur schedule
    config.blur_filter_size = 15
    config.blur_sigma_init = 1.0
    config.blur_sigma_decay_end_pct = 0.5

    config.augment_bounds = (0.25, 0.25, jnp.pi, 0.25, 0.25)
    config.augment_offset = (0.0, 0.0, 0.0, 0.0, 0.0)

    config.train_split = f"train[{config.num_val}:{end_index}]"
    config.pp_train = f'value_range(-1, 1)|random_rotate(-{config.angle}, {config.angle}, fill_value=-1)|keep(["image", "label"])'
    config.val_split = f"train[:{config.num_val}]"
    config.pp_eval = f'value_range(-1, 1)|random_rotate(-{config.angle}, {config.angle}, fill_value=-1)|keep(["image", "label"])'

    config.model_name = "inference_net"
    config.model = config_dict.ConfigDict()
    config.model.squash_to_bounds = False
    config.model.hidden_dims = (2048, 1024, 512, 256)

    return config
