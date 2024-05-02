from jax import numpy as jnp
from ml_collections import config_dict

from experiments.configs.datasets import (
    add_aug_dsprites_config,
    add_aug_dsprites_config_v2,
    add_galaxy_mnist_config,
    add_mnist_config,
)
from src.transformations.transforms import (
    AffineAndHSVWithoutShearTransform,
    AffineTransformWithoutShear,
)


def get_config(params) -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    params = params.split(",")
    config.dataset = params[0]
    v2 = False
    if config.dataset == "aug_dspritesv2":
        config.dataset = "aug_dsprites"
        v2 = True
    assert config.dataset in ["MNIST", "aug_dsprites", "galaxy_mnist"]
    config.seed = int(params[1])
    if len(params) == 3:
        config.num_trn = int(params[2])
    if len(params) > 3:
        config.angle = float(params[2])
        config.num_trn = int(params[3])

    config.eval_freq = 0.01
    config.checkpoint = ""

    config.transform_kwargs = {"order": 3}

    config.n_samples = 5
    config.init_lr_mult = 0.1
    config.warmup_steps_pct = 0.2
    config.clip_norm = 2.0
    config.weight_decay = 1e-4
    config.mae_loss_mult = 1.0
    # config.bounds_mult = 1.0 if not (config.dataset == "aug_dsprites" and v2) else 0.75
    match (config.dataset, v2):
        case ("aug_dsprites", True):
            config.bounds_mult = 0.75
        case ("galaxy_mnist", _):
            config.bounds_mult = 0.75
        case (_, _):
            config.bounds_mult = 1.0

    match (config.dataset, v2):
        case ("MNIST", _):
            config.augment_bounds = (0.25, 0.25, jnp.pi, 0.25, 0.25)
        case ("aug_dsprites", True):
            config.augment_bounds = (0.75, 0.75, jnp.pi, 0.75, 0.75)
        case ("aug_dsprites", False):
            config.augment_bounds = (0.5, 0.5, jnp.pi, 0.5, 0.5)
        case ("galaxy_mnist", _):
            config.augment_bounds = (0.25, 0.25, jnp.pi, 0.25, 0.25, 0.5, 2.31, 0.51)

    match config.dataset:
        case "galaxy_mnist":
            config.augment_offset = (0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0)
        case _:
            config.augment_offset = (0.0, 0.0, 0.0, 0.0, 0.0)

    config.model_name = "generative_net"
    config.model = config_dict.ConfigDict()
    match (config.dataset, v2):
        case ("aug_dsprites", True) | ("galaxy_mnist", _):
            config.model.squash_to_bounds = True
        case (_, _):
            config.model.squash_to_bounds = False
    config.model.hidden_dims = (1024, 512, 256)
    config.model.num_bins = 6
    config.model.conditioner = config_dict.ConfigDict()
    config.model.conditioner.hidden_dims = (256,)
    config.model.conditioner.dropout_rate = 0.1
    config.model.transform = (
        AffineTransformWithoutShear
        if config.dataset != "galaxy_mnist"
        else AffineAndHSVWithoutShearTransform
    )

    match (
        config.dataset,
        config.get("num_trn", None),
        config.seed,
    ):
        case ("MNIST", 50_000, 0) | ("MNIST", 50_000, 1) | ("MNIST", 50_000, 2):
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
        case ("MNIST", 12_500, 0):  # 10lsa6lh
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.2
            config.model.num_flows = 6
            config.steps = 7500
        case ("MNIST", 12_500, 1):  # tnjpu9b4
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.2
            config.model.num_flows = 5
            config.steps = 7500
        case ("MNIST", 12_500, 2):  # 5h56yb82
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.2
            config.model.num_flows = 5
            config.steps = 7500
        case ("aug_dsprites", None, 0):
            if not v2:
                config.final_lr_mult = 0.3
                config.lr = 0.0003
                config.model.dropout_rate = 0.05
                config.model.num_flows = 6
                config.steps = 60000
            else:  # q4r7bi1r
                config.final_lr_mult = 0.03
                config.lr = 0.003
                config.mae_loss_mult = 0.0
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
        case ("galaxy_mnist", 7_000, 0):  # i8yrjt9f
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.05
            config.model.num_flows = 4
            config.steps = 15000
        case ("galaxy_mnist", 7_000, 1):  # o8m0r0v5
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.05
            config.model.num_flows = 5
            config.steps = 7500
        case ("galaxy_mnist", 7_000, 2):  # owjbmfaz
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.05
            config.model.num_flows = 5
            config.steps = 7500
        case ("galaxy_mnist", 3_500, 0):  #
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.05
            config.model.num_flows = 4
            config.steps = 7500
        case ("galaxy_mnist", 3_500, 1):  # jtsuv2a8
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.05
            config.model.num_flows = 5
            config.steps = 3750
        case ("galaxy_mnist", 3_500, 2):  # gf3dj73k
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.05
            config.model.num_flows = 4
            config.steps = 7500

    config.batch_size = 512
    if config.dataset == "MNIST":
        config.num_val = 10000
        add_mnist_config(
            config,
            angle=config.angle,
            num_trn=config.get("num_trn", None),
            num_val=config.num_val,
        )
    elif config.dataset == "aug_dsprites":
        if not v2:
            add_aug_dsprites_config(config)
        else:
            add_aug_dsprites_config_v2(config)
    else:
        config.num_val = 1000
        add_galaxy_mnist_config(
            config,
            num_trn=config.get("num_trn", None),
            num_val=config.num_val,
        )

    return config
