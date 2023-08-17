import copy
from ml_collections import config_dict
from jax import numpy as jnp


def get_config(params) -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    angle, num_trn = params.split(",")
    config.angle = int(angle)
    config.num_trn = int(num_trn)
    config.num_val = 10000
    # config.γ_mult = int(γ_mult)

    config.seed = 0
    # Dataset config
    config.dataset = "MNIST"
    # config.data_dir = '~/data'
    config.shuffle_buffer_size = 50_000
    config.repeat_after_batching = True  # NOTE: ordering of PP, shuffle, and repeat is important!
    config.train_split = f"train[{config.num_val}:{config.num_val+config.num_trn}]"
    config.pp_train = f'value_range(-1, 1)|random_rotate(-{config.angle}, {config.angle}, fill_value=-1)|keep(["image"])'
    config.val_split = f"train[:{config.num_val}]"
    config.pp_eval = f'value_range(-1, 1)|random_rotate(-{config.angle}, {config.angle}, fill_value=-1)|keep(["image", "label"])'

    # Model config
    config.model_name = "SSIL"
    config.model = config_dict.ConfigDict()
    config.model.bounds = (0.25, 0.25, jnp.pi, 0.25, 0.25)
    config.model.offset = (0.0, 0.0, 0.0, 0.0, 0.0)
    # p(η|Xhat) config
    config.model.Η_given_Xhat = config_dict.ConfigDict()
    config.model.Η_given_Xhat.num_layers = 1
    config.model.Η_given_Xhat.num_bins = 5
    config.model.Η_given_Xhat.base = config_dict.ConfigDict()
    config.model.Η_given_Xhat.base.dense_dims = (64, 32, 16)
    config.model.Η_given_Xhat.base.conv_dims = ()
    config.model.Η_given_Xhat.base.dropout_rate = 0.0
    config.model.Η_given_Xhat.conditioner = config_dict.ConfigDict()
    config.model.Η_given_Xhat.conditioner.hidden_dims = (256, 128)
    config.model.Η_given_Xhat.conditioner.dropout_rate = 0.0
    config.model.Η_given_Xhat.trunk = config_dict.ConfigDict()
    config.model.Η_given_Xhat.trunk.dense_dims = (256,)
    config.model.Η_given_Xhat.trunk.conv_dims = (32, 64, 128)
    config.model.Η_given_Xhat.trunk.dropout_rate = 0.0
    # p(η|X) config
    config.model.Η_given_X = copy.deepcopy(config.model.Η_given_Xhat)

    # Training config
    config.total_steps = 5001
    config.eval_every = 500
    config.batch_size = 512

    ## Optimizer config
    config.optim_name = "adamw"
    config.learning_rate = 3e-4
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

    ## γ config
    config.γ = 1
    config.γ_schedule_name = "cosine_decay_schedule"
    config.γ_schedule = config_dict.ConfigDict()
    config.γ_end_value = 0
    config.γ_schedule.alpha = config.γ_end_value / config.γ
    config.γ_schedule.decay_steps = config.total_steps // 2

    return config
