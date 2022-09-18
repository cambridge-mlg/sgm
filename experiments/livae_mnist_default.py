from math import ceil

from ml_collections import config_dict
from jax import numpy as jnp

from src.data import METADATA, train_val_split_sizes

UPPER_ROT = jnp.pi/4

def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.epochs = 25
    config.batch_size = 256
    config.random_seed = 1

    config.dataset_name = 'MNIST'
    config.dataset = config_dict.ConfigDict()
    config.dataset.data_dir = '/homes/jua23/Git/learning-invariances/raw_data'
    config.dataset.flatten_img = False
    config.dataset.val_percent = 0.1
    config.dataset.random_seed = 42
    config.dataset.train_augmentations = []
    config.dataset.test_augmentations = []
    config.dataset.η_low = [0., 0., -UPPER_ROT, 0., 0., 0., 0.]
    config.dataset.η_high = [0., 0., UPPER_ROT, 0., 0., 0., 0.]

    config.optim_name = 'adamw'
    config.optim = config_dict.ConfigDict()
    config.optim.weight_decay = 1e-4
    config.learning_rate = 1e-4

    num_train, _ = train_val_split_sizes(METADATA['num_train'][config.dataset_name], config.dataset.val_percent)
    num_batches_per_epoch = ceil(num_train / config.batch_size)

    config.lr_schedule_name = 'warmup_cosine_decay_schedule'
    config.lr_schedule = config_dict.ConfigDict()
    config.lr_schedule.peak_value = 3 * config.learning_rate
    config.lr_schedule.end_value = 1/3 * config.learning_rate
    config.lr_schedule.decay_steps = config.epochs * num_batches_per_epoch
    config.lr_schedule.warmup_steps = int(0.2 * config.lr_schedule.decay_steps)

    config.β = 10
    config.β_schedule_name = 'cosine_decay_schedule'
    config.β_schedule = config_dict.ConfigDict()
    config.β_end_value = 1
    config.β_schedule.alpha = config.β_end_value / config.β
    config.β_schedule.decay_steps = config.epochs * num_batches_per_epoch

    config.model_name = 'LIVAE'
    config.model = config_dict.ConfigDict()
    config.model.latent_dim = 128
    config.model.learn_prior = False
    config.model.architecture = 'ConvNet'
    config.model.encoder_invariance = 'none'  # 'none'  # 'partial'  # 'full'
    config.model.invariance_samples = 0
    config.model.learn_η_loc = False
    config.model.learn_η_scale = True

    config.model.η_encoder = config_dict.ConfigDict()
    # config.model.η_encoder.posterior = 'uniform'
    config.model.η_encoder.posterior = 'hetero-diag-normal'
    config.model.η_encoder.hidden_dims = [64, 128, 256]
    config.model.η_mask = jnp.array([0., 0., 1., 0., 0., 0., 0.])

    config.model.encoder = config_dict.ConfigDict()
    config.model.encoder.posterior = 'hetero-diag-normal'
    config.model.encoder.hidden_dims = [64, 128, 256]

    config.model.decoder = config_dict.ConfigDict()
    config.model.decoder.likelihood = 'iso-normal'
    config.model.decoder.σ_min = 1e-2
    config.model.decoder.hidden_dims = list(reversed(config.model.encoder.hidden_dims))
    config.model.decoder.image_shape = METADATA['image_shape'][config.dataset_name]

    return config
