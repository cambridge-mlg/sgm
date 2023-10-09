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
    config.difficulty_weighted_inf_loss = True

    config.inf_steps = 10_000
    config.inf_init_lr = 1e-4
    config.inf_peak_lr_mult = 3
    config.inf_final_lr_mult = 1 / 30
    config.inf_warmup_steps = config.inf_steps // 10
    config.σ_lr = 1e-2
    config.gen_steps = 10_000
    config.gen_init_lr = 3e-4
    config.gen_peak_lr_mult = 9
    config.gen_final_lr_mult = 1 / 300
    config.gen_warmup_steps = config.gen_steps // 10

    config.α_schedule_pct = 0.25
    config.α_schedule_final_value = 0.0
    config.β_schedule_pct = 1.0
    config.β_schedule_final_value = 0.99

    config.train_split = f"train[{config.num_val}:{end_index}]"
    config.pp_train = f'value_range(-1, 1)|random_rotate(-{config.angle}, {config.angle}, fill_value=-1)|keep(["image"])'
    config.val_split = f"train[:{config.num_val}]"
    config.pp_eval = f'value_range(-1, 1)|random_rotate(-{config.angle}, {config.angle}, fill_value=-1)|keep(["image", "label"])'

    config.model_name = "PGM"
    config.model = config_dict.ConfigDict()
    config.model.bounds = (0.25, 0.25, jnp.pi, 0.25, 0.25)
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
