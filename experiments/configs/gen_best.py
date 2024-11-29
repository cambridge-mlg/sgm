from jax import numpy as jnp
from ml_collections import config_dict

from experiments.configs.datasets import (
    add_aug_dsprites_config,
    add_galaxy_mnist_config,
    add_mnist_config,
    add_patch_camelyon_config,
)
from src.transformations.transforms import (
    AffineAndHSVWithoutShearTransform,
    AffineTransformWithoutShear,
)


def get_config(params) -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    params = params.split(",")
    config.dataset = params[0]
    assert config.dataset in ["MNIST", "aug_dsprites", "galaxy_mnist", "patch_camelyon"]
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
    config.consistency_loss_mult = 1.0
    match config.dataset:
        case "aug_dsprites":
            config.bounds_mult = 0.75
        case "galaxy_mnist" | "patch_camelyon":
            config.bounds_mult = 0.75
        case _:
            config.bounds_mult = 1.0

    match config.dataset:
        case "MNIST":
            config.augment_bounds = (0.25, 0.25, jnp.pi, 0.25, 0.25)
        case "aug_dsprites":
            config.augment_bounds = (0.75, 0.75, jnp.pi, 0.75, 0.75)
        case "galaxy_mnist" | "patch_camelyon":
            config.augment_bounds = (0.25, 0.25, jnp.pi, 0.25, 0.25, 0.5, 2.31, 0.51)

    match config.dataset:
        case "galaxy_mnist" | "patch_camelyon":
            config.augment_offset = (0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0)
        case _:
            config.augment_offset = (0.0, 0.0, 0.0, 0.0, 0.0)

    config.model_name = "generative_net"
    config.model = config_dict.ConfigDict()
    match config.dataset:
        case "aug_dsprites" | "galaxy_mnist" | "patch_camelyon":
            config.model.squash_to_bounds = True
        case _:
            config.model.squash_to_bounds = False
    config.model.hidden_dims = (1024, 512, 256)
    config.model.num_bins = 6
    config.model.conditioner = config_dict.ConfigDict()
    config.model.conditioner.hidden_dims = (256,)
    config.model.conditioner.dropout_rate = 0.1
    config.model.transform = (
        AffineTransformWithoutShear
        if config.dataset != "galaxy_mnist" and config.dataset != "patch_camelyon"
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
            # q4r7bi1r
            config.final_lr_mult = 0.03
            config.lr = 0.003
            config.consistency_loss_mult = 0.0
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
        case ("galaxy_mnist", 7_000, 0):
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.05
            config.model.num_flows = 4
            config.steps = 15000
        case ("galaxy_mnist", 7_000, 1):
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.05
            config.model.num_flows = 5
            config.steps = 7500
        case ("galaxy_mnist", 7_000, 2):
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.05
            config.model.num_flows = 5
            config.steps = 7500
        case ("galaxy_mnist", 3_500, 0):
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.05
            config.model.num_flows = 4
            config.steps = 7500
        case ("galaxy_mnist", 3_500, 1):
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.05
            config.model.num_flows = 5
            config.steps = 3750
        case ("galaxy_mnist", 3_500, 2):
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.05
            config.model.num_flows = 4
            config.steps = 7500
        case ("patch_camelyon", 262_144, 0):  # jrdy9r66
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.1
            config.model.num_flows = 5
            config.steps = 60000
            # config.consistency_loss_mult = 0.0
            # ^ This is actually the best setting but we made a mistake
        case ("patch_camelyon", 65_536, 0):  # iewzgy5a
            config.final_lr_mult = 0.03  # 0.3 is actually better but we made a mistake
            config.lr = 0.0003
            config.model.dropout_rate = 0.2
            config.model.num_flows = 6  # 5 is actually better but we made a mistake
            config.steps = 60000
        case ("patch_camelyon", 16_384, 0):  # k6w6dbb6
            config.final_lr_mult = 0.03
            config.lr = 0.0003
            config.model.dropout_rate = 0.2
            config.model.num_flows = 4  # 5 is actually better but we made a mistake
            config.steps = 15000

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
        add_aug_dsprites_config(config)
    elif config.dataset == "patch_camelyon":
        config.num_val = 32_768
        add_patch_camelyon_config(
            config,
            num_trn=config.get("num_trn", None),
            num_val=config.num_val,
        )
    else:
        config.num_val = 1000
        add_galaxy_mnist_config(
            config,
            num_trn=config.get("num_trn", None),
            num_val=config.num_val,
        )

    return config
