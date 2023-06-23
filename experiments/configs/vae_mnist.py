from ml_collections import config_dict
from jax import numpy as jnp


def get_config(params) -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    angle, num_trn, total_steps = params.split(",")[:3]
    config.angle = int(angle)
    config.num_trn = int(num_trn)
    config.num_val = 10000
    config.total_steps = int(total_steps)

    config.seed = 0
    # Dataset config
    config.dataset = "MNIST"
    # config.data_dir = '~/data'
    config.shuffle = "preprocessed"
    config.shuffle_buffer_size = 50_000
    config.repeat_after_batching = True  # NOTE: ordering of PP, shuffle, and repeat is important!
    config.train_split = f"train[{config.num_val}:{config.num_val+config.num_trn}]"
    config.pp_train = f'value_range(-1, 1)|random_rotate(-{config.angle}, {config.angle}, fill_value=-1)|keep(["image"])'
    config.val_split = f"train[:{config.num_val}]"
    config.pp_eval = f'value_range(-1, 1)|random_rotate(-{config.angle}, {config.angle}, fill_value=-1)|keep(["image", "label"])'

    # Model config
    config.model_name = "VAE"
    config.model = config_dict.ConfigDict()
    config.model.latent_dim = 16
    ## q(Z|X) config
    config.model.Z_given_X = config_dict.ConfigDict()
    config.model.Z_given_X.conv_dims = (64, 128, 256)
    config.model.Z_given_X.dense_dims = (256,)
    config.model.Z_given_X.max_2strides = 2
    ## p(X|Z) config
    config.model.X_given_Z = config_dict.ConfigDict()
    config.model.X_given_Z.conv_dims = (256, 128, 64)
    config.model.X_given_Z.dense_dims = (256,)
    config.model.X_given_Z.max_2strides = 2

    # Training config
    config.eval_every = 500
    config.batch_size = 512
    config.batch_size_eval = 256

    ## Optimizer config
    config.optim_name = "adamw"
    config.learning_rate = 1e-4
    config.optim = config_dict.ConfigDict()
    config.optim.weight_decay = 1e-4

    ## LR config
    config.lr_schedule_name = "warmup_cosine_decay_schedule"
    config.lr_schedule = config_dict.ConfigDict()
    config.lr_schedule.peak_value = 10 * config.learning_rate
    config.lr_schedule.end_value = 1 * config.learning_rate
    config.lr_schedule.decay_steps = config.total_steps
    config.lr_schedule.warmup_steps = config.total_steps // 10

    ## α config
    config.α = 1

    ## β config
    config.β = 10
    config.β_schedule_name = "cosine_decay_schedule"
    config.β_schedule = config_dict.ConfigDict()
    config.β_end_value = 1
    config.β_schedule.alpha = config.β_end_value / config.β
    config.β_schedule.decay_steps = config.total_steps

    ## γ config
    config.γ = 1

    # MLL config
    config.run_hais = False
    config.hais = config_dict.ConfigDict()
    config.hais.num_chains = 300
    config.hais.num_steps = 100
    config.hais.step_size = 8e-3
    config.hais.num_leapfrog_steps = 2
    config.run_iwlb = True
    config.iwlb = config_dict.ConfigDict()
    config.iwlb.num_samples = 100

    return config
