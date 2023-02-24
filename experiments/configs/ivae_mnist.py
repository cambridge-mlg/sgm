from ml_collections import config_dict
from jax import numpy as jnp


def get_config(angle) -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.seed = 0
    config.eval_every = 500

    # Dataset config
    config.dataset = "MNIST"
    # config.data_dir = '~/data'
    config.shuffle_buffer_size = 50_000
    config.repeat_after_batch = True  # NOTE: ordering of PP and repeat is important!
    config.train_split = "train[10000:]"
    config.pp_train = (
        f'value_range(-1, 1)|random_rotate(-{angle}, {angle}, fill_value=-1)|keep(["image"])'
    )
    config.val_split = "train[:10000]"
    config.pp_eval = (
        f'value_range(-1, 1)|random_rotate(-{angle}, {angle}, fill_value=-1)|keep(["image"])'
    )

    # Model config
    config.model_name = "LIVAE"
    config.model = config_dict.ConfigDict()
    config.model.latent_dim = 128
    config.model.bounds = (0.25, 0.25, jnp.pi, 0.25, 0.25, jnp.pi / 6, jnp.pi / 6)
    ## q(Z|X) config
    config.model.Z_given_X = config_dict.ConfigDict()
    config.model.Z_given_X.hidden_dims = [64, 128, 256]
    ## p(X̂|Z) config
    config.model.Xhat_given_Z = config_dict.ConfigDict()
    config.model.Xhat_given_Z.hidden_dims = [256, 128, 64]

    # Training config
    config.total_steps = 7501
    config.batch_size = 256

    ## Optimizer config
    config.optim_name = "adamw"
    config.learning_rate = 1e-4
    config.optim = config_dict.ConfigDict()
    config.optim.weight_decay = 1e-4

    ## β config
    config.β = 10
    config.β_schedule_name = "cosine_decay_schedule"
    config.β_schedule = config_dict.ConfigDict()
    β_end_value = 1
    config.β_schedule.alpha = β_end_value / config.β
    config.β_schedule.decay_steps = config.total_steps

    ## α config
    config.α = 1

    ## LR config
    config.lr_schedule_name = "warmup_cosine_decay_schedule"
    config.lr_schedule = config_dict.ConfigDict()
    config.lr_schedule.peak_value = 10 * config.learning_rate
    config.lr_schedule.end_value = config.learning_rate
    config.lr_schedule.decay_steps = config.total_steps
    config.lr_schedule.warmup_steps = 1000

    return config
