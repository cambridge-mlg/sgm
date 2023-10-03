from ml_collections import config_dict
from jax import numpy as jnp


def get_config(params) -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    params = params.split(",")
    angle = int(params[0])
    num_val = 10000
    if len(params) > 1:
        num_trn = params[1]
        end_index = num_trn + num_val
    else:
        end_index = ""

    config.seed = 0

    config.batch_size = 512
    config.dataset = "MNIST"
    config.shuffle_buffer_size = 50_000
    config.repeat_after_batching = (
        True  # NOTE: ordering of PP, shuffle, and repeat is important!
    )
    config.n_samples = 5
    config.eval_freq = 0.01
    config.difficulty_weighted_inf_loss = True

    config.inf_steps = 10_000
    config.inf_lr = 3e-4
    config.inf_init_lr_mult = 1 / 3
    config.inf_final_lr_mult = 1 / 90
    config.inf_warmup_steps = 1_000
    config.σ_lr = 1e-2
    config.gen_steps = 10_000
    config.gen_lr = 2.7e-3
    config.gen_init_lr_mult = 1 / 9
    config.gen_final_lr_mult = 1 / 2700
    config.gen_warmup_steps = 1_000

    config.α_schedule_pct = 0.25
    config.α_schedule_final_value = 0.0
    config.β_schedule_pct = 1.0
    config.β_schedule_final_value = 0.99

    config.train_split = f"train[{num_val}:{end_index}]"
    config.pp_train = f'value_range(-1, 1)|random_rotate(-{angle}, {angle}, fill_value=-1)|keep(["image"])'
    config.val_split = f"train[:{num_val}]"
    config.pp_eval = f'value_range(-1, 1)|random_rotate(-{angle}, {angle}, fill_value=-1)|keep(["image", "label"])'

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
