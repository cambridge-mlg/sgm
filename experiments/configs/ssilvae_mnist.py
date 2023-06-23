import copy
from ml_collections import config_dict
from jax import numpy as jnp


def get_config(params) -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    angle, num_trn, per_stage_steps, γ_mult = params.split(",")
    config.angle = int(angle)
    config.num_trn = int(num_trn)
    config.num_val = 10000
    config.per_stage_steps = int(per_stage_steps)
    config.γ_mult = int(γ_mult)

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
    config.model_name = "SSILVAE"
    config.model = config_dict.ConfigDict()
    config.model.bounds = (0.25, 0.25, jnp.pi, 0.25, 0.25)
    config.model.offset = (0.0, 0.0, 0.0, 0.0, 0.0)
    config.model.latent_dim = 16
    ## q(Z|Xhat) config
    config.model.Z_given_Xhat = config_dict.ConfigDict()
    config.model.Z_given_Xhat.conv_dims = (64, 128, 256)
    config.model.Z_given_Xhat.dense_dims = (256,)
    config.model.Z_given_Xhat.max_2strides = 2
    ## p(Xhat|Z) config
    config.model.Xhat_given_Z = config_dict.ConfigDict()
    config.model.Xhat_given_Z.conv_dims = (256, 128, 64)
    config.model.Xhat_given_Z.dense_dims = (256,)
    config.model.Xhat_given_Z.max_2strides = 2
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
    config.total_steps = config.per_stage_steps * 2
    config.eval_every = 500
    config.batch_size = 512

    ## Parameter parition config
    config.reset_step = config.per_stage_steps
    config.partition_names = ("SSIL", "VAE")
    config.lr_schedule_halfs = ("first", "second")
    config.partition_fn = (
        lambda path, _: "SSIL" if path[0] in ("p_Η_given_Xhat", "q_Η_given_X", "σ_") else "VAE"
    )

    ## Optimizer config
    config.optim_name = ("adamw", "adamw")
    config.learning_rate = (3e-4, 1e-4)
    config.optim = (config_dict.ConfigDict(), config_dict.ConfigDict())
    config.optim[0].weight_decay = 1e-4
    config.optim[1].weight_decay = 1e-4

    ## LR config
    config.lr_schedule_name = (
        "warmup_cosine_decay_schedule",
        "warmup_cosine_decay_schedule",
    )
    config.lr_schedule = (config_dict.ConfigDict(), config_dict.ConfigDict())
    for i in range(len(config.lr_schedule)):
        config.lr_schedule[i].peak_value = 10 * config.learning_rate[i]
        config.lr_schedule[i].end_value = 1 * config.learning_rate[i]
        config.lr_schedule[i].decay_steps = config.per_stage_steps
        config.lr_schedule[i].warmup_steps = config.per_stage_steps // 10

    ## α config
    config.α = 1

    ## β config
    config.β_schedule_half = "second"
    config.β = 10
    config.β_schedule_name = "cosine_decay_schedule"
    config.β_schedule = config_dict.ConfigDict()
    config.β_end_value = 1
    config.β_schedule.alpha = config.β_end_value / config.β
    config.β_schedule.decay_steps = config.per_stage_steps

    ## γ config
    config.γ_schedule_half = "first"
    config.γ = 10 * config.γ_mult
    config.γ_schedule_name = "cosine_decay_schedule"
    config.γ_schedule = config_dict.ConfigDict()
    config.γ_end_value = 1 * config.γ_mult
    config.γ_schedule.alpha = config.γ_end_value / config.γ
    config.γ_schedule.decay_steps = config.per_stage_steps

    # MLL config
    config.run_hais = False
    config.hais = config_dict.ConfigDict()
    config.hais.num_chains = 300
    config.hais.num_steps = 100
    config.hais.step_size = 8e-3
    config.hais.num_leapfrog_steps = 2
    config.run_iwlb = False
    config.iwlb = config_dict.ConfigDict()
    config.iwlb.num_samples = 100

    return config
