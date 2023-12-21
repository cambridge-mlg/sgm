from jax import numpy as jnp
from ml_collections import config_dict

from experiments.configs.datasets import add_aug_dsprites_config, add_mnist_config


def get_config(params) -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    params = params.split(",")
    config.dataset = params[0]
    assert config.dataset in ["MNIST", "aug_dsprites"]
    config.seed = int(params[1])
    if len(params) > 2:
        config.angle = float(params[2])
        config.num_trn = int(params[3])

    config.eval_freq = 0.01
    config.checkpoint = ""

    config.interpolation_order = 3
    config.translate_last = False

    config.n_samples = 5
    config.init_lr_mult = 0.1
    config.warmup_steps_pct = 0.2
    config.clip_norm = 2.0
    config.weight_decay = 1e-4
    config.mae_loss_mult = 1.0

    config.augment_bounds = (
        (0.25, 0.25, jnp.pi, 0.25, 0.25)
        if config.dataset == "MNIST"
        else (0.5, 0.5, jnp.pi, 0.5, 0.5)
    )
    config.augment_offset = (0.0, 0.0, 0.0, 0.0, 0.0)

    config.model_name = "generative_net"
    config.model = config_dict.ConfigDict()
    config.model.squash_to_bounds = False
    config.model.hidden_dims = (1024, 512, 256)
    config.model.num_bins = 6
    config.model.conditioner = config_dict.ConfigDict()
    config.model.conditioner.hidden_dims = (256,)
    config.model.conditioner.dropout_rate = 0.1

    match (
        config.dataset,
        # config.get("angle", None),
        config.get("num_trn", None),
        config.seed,
    ):
        case (("MNIST", 50_000, 0) | ("MNIST", 50_000, 1) | ("MNIST", 50_000, 2)):
            config.final_lr_mult = 0.03
            config.lr = 0.003
            config.model.dropout_rate = 0.2
            config.model.num_flows = 6
            config.steps = 60000
        case ("MNIST", 37_500, 0):
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.2
            config.model.num_flows = 6
            config.steps = 15000
        case ("MNIST", 37_500, 1):
            config.final_lr_mult = 0.03
            config.lr = 0.003
            config.model.dropout_rate = 0.2
            config.model.num_flows = 6
            config.steps = 30000
        case ("MNIST", 37_500, 2):
            config.final_lr_mult = 0.03
            config.lr = 0.003
            config.model.dropout_rate = 0.2
            config.model.num_flows = 5
            config.steps = 30000
        case ("MNIST", 25_000, 0):
            config.final_lr_mult = 0.03
            config.lr = 0.003
            config.model.dropout_rate = 0.2
            config.model.num_flows = 5
            config.steps = 7500
        case ("MNIST", 25_000, 1) | ("MNIST", 25_000, 2):
            config.final_lr_mult = 0.03
            config.lr = 0.003
            config.model.dropout_rate = 0.2
            config.model.num_flows = 6
            config.steps = 7500
        case ("aug_dsprites", None, 0):
            config.final_lr_mult = 0.3
            config.lr = 0.0003
            config.model.dropout_rate = 0.05
            config.model.num_flows = 6
            config.steps = 60000
        case ("aug_dsprites", None, 1):
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.05
            config.model.num_flows = 6
            config.steps = 60000
        case ("aug_dsprites", None, 2):
            config.final_lr_mult = 0.3
            config.lr = 0.003
            config.model.dropout_rate = 0.05
            config.model.num_flows = 6
            config.steps = 60000

    config.batch_size = 512
    if config.dataset == "MNIST":
        config.num_val = 10000
        add_mnist_config(
            config,
            angle=config.angle,
            num_trn=config.get("num_trn", None),
            num_val=config.num_val,
        )
    else:
        add_aug_dsprites_config(config)

    return config
