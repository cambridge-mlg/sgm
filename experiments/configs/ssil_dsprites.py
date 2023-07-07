import copy
from ml_collections import config_dict
from jax import numpy as jnp


def get_config(params) -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    (num_trn,) = params.split(",")
    config.num_trn = int(num_trn)
    config.num_val = 10000
    config.num_tst = 10000
    # Dataset config
    config.dataset = "dsprites"
    config.shuffle_buffer_size = 50_000
    config.repeat_after_batch = True  # NOTE: ordering of PP and repeat is important!
    config.train_split = f"train[{config.num_tst + config.num_val}:{config.num_tst + config.num_val + config.num_trn}]"
    config.pp_train = f'value_range(-1, 1, 0, 1, True)|keep(["image"])'
    config.val_split = f"train[{config.num_tst}:{config.num_tst + config.num_val}]"
    config.pp_eval = f'value_range(-1, 1, 0, 1, True)|move_key("label_shape", "label")|keep(["image", "label"])'

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
    config.model.Η_given_Xhat.base.dense_dims = (128,)
    config.model.Η_given_Xhat.base.conv_dims = ()
    config.model.Η_given_Xhat.base.dropout_rate = 0.0
    config.model.Η_given_Xhat.conditioner = config_dict.ConfigDict()
    config.model.Η_given_Xhat.conditioner.hidden_dims = (128,)
    config.model.Η_given_Xhat.conditioner.dropout_rate = 0.0
    config.model.Η_given_Xhat.trunk = config_dict.ConfigDict()
    config.model.Η_given_Xhat.trunk.dense_dims = (256, 128)
    config.model.Η_given_Xhat.trunk.conv_dims = (16, 32, 64)
    config.model.Η_given_Xhat.trunk.max_2strides = 2
    config.model.Η_given_Xhat.trunk.resize = False
    config.model.Η_given_Xhat.trunk.dropout_rate = 0.0
    # p(η|X) config
    config.model.Η_given_X = copy.deepcopy(config.model.Η_given_Xhat)

    # Training config
    config.total_steps = 5001
    config.eval_every = 500
    config.batch_size = 512

    ## Optimizer config
    config.optim_name = "adamw"
    config.learning_rate = 1e-4
    config.optim = config_dict.ConfigDict()
    config.optim.weight_decay = 1e-4

    ## LR config
    config.lr_schedule_name = "warmup_cosine_decay_schedule"
    config.lr_schedule = config_dict.ConfigDict()
    config.lr_schedule.peak_value = 10 * config.learning_rate
    config.lr_schedule.end_value = 0.3 * config.learning_rate
    config.lr_schedule.decay_steps = config.total_steps
    config.lr_schedule.warmup_steps = config.total_steps // 10

    ## α config
    config.α = 1

    ## γ config
    config.γ = 0

    return config
