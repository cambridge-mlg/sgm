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
    config.x_mse_loss_mult = 1.0
    match config.dataset:
        case "MNIST":
            config.invertibility_loss_mult = 0.1
        case "aug_dsprites":
            config.invertibility_loss_mult = 1.0
        case "galaxy_mnist" | "patch_camelyon":
            config.invertibility_loss_mult = 0.0

    config.η_loss_mult = 0.0
    match config.dataset:
        case "galaxy_mnist":
            config.steps = 10_000
        case "patch_camelyon":
            config.steps = 20_000
        case _:
            config.steps = 60_000
    config.σ_lr = 3e-3
    config.blur_filter_size = 5
    config.blur_end_pct = 0.01
    config.weight_decay = 1e-4

    match config.dataset:
        case "aug_dsprites":
            config.symmetrized_loss = False
        case _:
            config.symmetrized_loss = True

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

    config.model_name = "inference_net"
    config.model = config_dict.ConfigDict()
    match config.dataset:
        case "aug_dsprites" | "galaxy_mnist" | "patch_camelyon":
            config.model.squash_to_bounds = True
        case _:
            config.model.squash_to_bounds = False

    config.model.hidden_dims = (
        (2048, 1024, 512, 256)
        if config.dataset != "galaxy_mnist" and config.dataset != "patch_camelyon"
        else (1024, 1024, 512, 256)
    )

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
        case ("MNIST", 50_000, 0) | ("MNIST", 37_500, 0):
            # ^ xegejvhb | 59c3uexk
            config.blur_σ_init = 0.0
            config.clip_norm = 10.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 3e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.05
        case ("MNIST", 50_000, 1):  # vdfhrd8g
            config.blur_σ_init = 0.0
            config.clip_norm = 10.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 1e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.2
        case ("MNIST", 50_000, 2):  # vsnpg2te
            config.blur_σ_init = 0.0
            config.clip_norm = 10.0
            config.final_lr_mult = 3e-4
            config.init_lr_mult = 1e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.05
        case ("MNIST", 37_500, 1):  # y6vxr0yb
            config.blur_σ_init = 3.0
            config.clip_norm = 10.0
            config.final_lr_mult = 3e-4
            config.init_lr_mult = 3e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.2
        case ("MNIST", 37_500, 2):  # smpwgumg
            config.blur_σ_init = 3.0
            config.clip_norm = 3.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 1e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.05
        case ("MNIST", 25_000, 0):  # kq36fzu9
            config.blur_σ_init = 0.0
            config.clip_norm = 10.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 1e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.1
        case ("MNIST", 25_000, 1):  # 38kq6q8f
            config.blur_σ_init = 3.0
            config.clip_norm = 10.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 3e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.1
        case ("MNIST", 25_000, 2):  # s7pj0ucn
            config.blur_σ_init = 0.0
            config.clip_norm = 3.0
            config.final_lr_mult = 3e-4
            config.init_lr_mult = 1e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.05
        case ("MNIST", 12_500, 0):  # nl0ipwa6
            config.blur_σ_init = 3.0
            config.clip_norm = 10.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 3e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.2
        case ("MNIST", 12_500, 1):  # 598xpjbr
            config.blur_σ_init = 3.0
            config.clip_norm = 10.0
            config.final_lr_mult = 3e-4
            config.init_lr_mult = 1e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.1
        case ("MNIST", 12_500, 2):  # t99ei9o5
            config.blur_σ_init = 0.0
            config.clip_norm = 3.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 1e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.1
        case ("aug_dsprites", None, 0):
            # 0vnue3s0
            config.blur_σ_init = 3.0
            config.clip_norm = 3.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 3e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.05
        case ("aug_dsprites", None, 1):  # lcr9fjj6
            config.blur_σ_init = 0.0
            config.clip_norm = 10.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 1e-2
            config.lr = 1e-3
            config.warmup_steps_pct = 0.1
        case ("aug_dsprites", None, 2):  # sup5y2kj
            config.blur_σ_init = 0.0
            config.clip_norm = 3.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 3e-2
            config.lr = 1e-3
            config.warmup_steps_pct = 0.2
        case ("galaxy_mnist", 7_000, 0):  # s7g82eq0
            config.blur_σ_init = 0.0
            config.clip_norm = 10.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 1e-2
            config.lr = 1e-3
            config.warmup_steps_pct = 0.1
        case ("galaxy_mnist", 7_000, 1):  # kyl4c0rk
            config.blur_σ_init = 0.0
            config.clip_norm = 10.0
            config.final_lr_mult = 3e-4
            config.init_lr_mult = 1e-2
            config.lr = 1e-3
            config.warmup_steps_pct = 0.2
        case ("galaxy_mnist", 7_000, 2):  # uy8ixhe3
            config.blur_σ_init = 3.0
            config.clip_norm = 10.0
            config.final_lr_mult = 3e-4
            config.init_lr_mult = 3e-2
            config.lr = 1e-3
            config.warmup_steps_pct = 0.2
        case ("galaxy_mnist", 3_500, 0):  # mjjvylei
            config.blur_σ_init = 3.0
            config.clip_norm = 10.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 1e-2
            config.lr = 1e-3
            config.warmup_steps_pct = 0.1
        case ("galaxy_mnist", 3_500, 1):  # mcjzbob9
            config.blur_σ_init = 0.0
            config.clip_norm = 3.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 1e-2
            config.lr = 1e-3
            config.warmup_steps_pct = 0.1
        case ("galaxy_mnist", 3_500, 2):  # uz676nx8
            config.blur_σ_init = 3.0
            config.clip_norm = 10.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 3e-2
            config.lr = 1e-3
            config.warmup_steps_pct = 0.2
        case ("patch_camelyon", 16_384, 0):  # 82r7tvy2
            config.blur_σ_init = 3.0
            config.clip_norm = 10.0
            config.final_lr_mult = 3e-4
            config.init_lr_mult = 3e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.05
        case ("patch_camelyon", 65_536, 0):  # g39czjr8
            config.blur_σ_init = 3.0
            config.clip_norm = 10.0
            config.final_lr_mult = 3e-4
            config.init_lr_mult = 1e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.05
        case ("patch_camelyon", 262_144, 0):  # 0pc7lq4t
            config.blur_σ_init = 0.0
            config.clip_norm = 3.0
            config.final_lr_mult = 1e-3
            config.init_lr_mult = 3e-2
            config.lr = 3e-4
            config.warmup_steps_pct = 0.05

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
