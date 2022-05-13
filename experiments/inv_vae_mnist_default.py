from math import ceil

from ml_collections import config_dict
from jax import numpy as jnp

from src.data import METADATA, train_val_split_sizes

def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.dataset_name = 'MNIST'
    config.val_percent = 0.1
    config.batch_size = 256
    config.epochs = 50

    config.optim_name = 'adamw'
    config.optim = config_dict.ConfigDict()
    config.optim.weight_decay = 1e-4
    config.learning_rate = 1e-4

    config.lr_schedule_name = 'warmup_cosine_decay_schedule'
    config.lr_schedule = config_dict.ConfigDict()
    config.lr_schedule.peak_value = 3 * config.learning_rate
    config.lr_schedule.end_value = 1/3 * config.learning_rate
    num_train, _ = train_val_split_sizes(METADATA['num_train'][config.dataset_name], config.val_percent)
    num_batches_per_epoch = ceil(num_train / config.batch_size)
    config.lr_schedule.decay_steps = config.epochs * num_batches_per_epoch
    config.lr_schedule.warmup_steps = int(0.2 * config.lr_schedule.decay_steps)

    config.model_name = 'invVAE'
    config.model = config_dict.ConfigDict()
    config.model.latent_dim = 32
    config.model.learn_prior = False
    config.model.architecture = 'ConvNeXt'  # 'ConvNet'  # 'MLP'
    config.model.max_rotation = jnp.pi/4
    config.model.encoder_invariance = 'partial'  # 'none'  # 'full'
    config.model.invariance_samples = 4

    config.β = 10  # 10
    config.β_schedule_name = 'cosine_decay_schedule'
    config.β_schedule = config_dict.ConfigDict()
    β_end_value = 3
    config.β_schedule.alpha = β_end_value / config.β
    config.β_schedule.decay_steps = config.epochs * num_batches_per_epoch

    config.model.encoder = config_dict.ConfigDict()
    config.model.encoder.posterior = 'hetero-diag-normal'
    config.model.encoder.hidden_dims = [64, 128]  # [64, 128, 256]

    config.model.decoder = config_dict.ConfigDict()
    config.model.decoder.likelihood = 'iso-normal'
    config.model.decoder.σ_min = 1e-2
    config.model.decoder.hidden_dims = list(reversed(config.model.encoder.hidden_dims))
    config.model.decoder.hidden_dims = [384, 192, 96, 48]
    config.model.decoder.image_shape = METADATA['image_shape'][config.dataset_name]

    return config
