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

    config.dataset = "MNIST"
    config.shuffle_buffer_size = 50_000
    config.shuffle = "preprocessed"
    config.repeat_after_batching = (
        True  # NOTE: ordering of PP, shuffle, and repeat is important!
    )
    config.train_split = f"train[{config.num_val}:{end_index}]"
    config.pp_train = f'value_range(-1, 1)|random_rotate(-{config.angle}, {config.angle}, fill_value=-1)|keep(["image"])'
    config.val_split = f"train[:{config.num_val}]"
    config.pp_eval = f'value_range(-1, 1)|random_rotate(-{config.angle}, {config.angle}, fill_value=-1)|keep(["image", "label"])'

    config.model_name = "AUG_VAE"
    config.model = config_dict.ConfigDict()
    config.model.vae = config_dict.ConfigDict()
    config.model.vae.latent_dim = 16
    config.model.vae.Z_given_X = config_dict.ConfigDict()
    config.model.vae.Z_given_X.conv_dims = (64, 128, 256)
    config.model.vae.Z_given_X.dense_dims = (256,)
    config.model.vae.Z_given_X.max_2strides = 2
    config.model.vae.X_given_Z = config_dict.ConfigDict()
    config.model.vae.X_given_Z.conv_dims = (256, 128, 64)
    config.model.vae.X_given_Z.dense_dims = (256,)
    config.model.vae.X_given_Z.max_2strides = 2
    config.model.pgm = config_dict.ConfigDict()
    config.model.pgm.bounds = (0.25, 0.25, jnp.pi, 0.25, 0.25)
    config.model.pgm.offset = (0.0, 0.0, 0.0, 0.0, 0.0)
    config.model.pgm.inference = config_dict.ConfigDict()
    config.model.pgm.inference.hidden_dims = (1024, 512, 256, 128)
    config.model.pgm.generative = config_dict.ConfigDict()
    config.model.pgm.generative.hidden_dims = (1024, 512, 256)
    config.model.pgm.generative.num_flows = 2
    config.model.pgm.generative.num_bins = 4
    config.model.pgm.generative.conditioner = config_dict.ConfigDict()
    config.model.pgm.generative.conditioner.hidden_dims = (256, 256)

    config.steps = 10_000
    config.eval_freq = 0.01
    config.plot_freq = 0.1
    config.batch_size = 512
    config.batch_size_eval = 64

    config.init_lr = 3e-4
    config.peak_lr_mult = 3
    config.final_lr_mult = 1
    config.warmup_steps = config.steps // 10

    config.β_schedule_init_value = 10.0
    config.β_schedule_final_value = 1.0

    config.run_iwlb = True
    config.iwlb_num_samples = 100

    return config
