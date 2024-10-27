import math
from typing import Optional

from ml_collections import config_dict


def add_mnist_config(
    config: config_dict.ConfigDict, angle: float, num_trn: Optional[int], num_val: int
) -> config_dict.ConfigDict:
    if num_trn is not None:
        end_index = num_trn + num_val
    else:
        end_index = ""

    config.train_split = f"train[{num_val}:{end_index}]"
    config.pp_train = f'value_range(-1, 1)|random_rotate(-{angle}, {angle}, fill_value=-1)|keep(["image", "label"])'
    config.val_split = f"train[:{num_val}]"
    config.pp_eval = f'value_range(-1, 1)|random_rotate(-{angle}, {angle}, fill_value=-1)|keep(["image", "label"])'

    config.shuffle_buffer_size = 50_000
    config.shuffle = "preprocessed"
    config.repeat_after_batching = (
        True  # NOTE: ordering of PP, shuffle, and repeat is important!
    )

    return config


def add_galaxy_mnist_config(
    config: config_dict.ConfigDict, num_trn: Optional[int], num_val: int
) -> config_dict.ConfigDict:
    if num_trn is not None:
        end_index = num_trn + num_val
    else:
        end_index = ""

    config.train_split = f"train[{num_val}:{end_index}]"
    config.pp_train = (
        f'value_range(-1, 1)|resize(64, "bilinear")|keep(["image", "label"])'
    )
    config.val_split = f"train[:{num_val}]"
    config.pp_eval = (
        f'value_range(-1, 1)|resize(64, "bilinear")|keep(["image", "label"])'
    )

    config.shuffle_buffer_size = 10_000
    config.shuffle = "preprocessed"
    config.repeat_after_batching = (
        True  # NOTE: ordering of PP, shuffle, and repeat is important!
    )

    return config


def add_patch_camelyon_config(
    config: config_dict.ConfigDict, num_trn: Optional[int], num_val: int
) -> config_dict.ConfigDict:
    if num_trn is not None:
        end_index = num_trn
    else:
        end_index = ""

    config.train_split = f"train[:{end_index}]"
    config.pp_train = (
        f'value_range(-1, 1)|resize(64, "bilinear")|keep(["image", "label"])'
    )
    config.val_split = f"validation[:{num_val}]"
    config.pp_eval = (
        f'value_range(-1, 1)|resize(64, "bilinear")|keep(["image", "label"])'
    )

    config.shuffle_buffer_size = 10_000
    config.shuffle = "preprocessed"
    config.repeat_after_batching = (
        True  # NOTE: ordering of PP, shuffle, and repeat is important!
    )

    return config


def add_aug_dsprites_config(config: config_dict.ConfigDict) -> config_dict.ConfigDict:
    config.train_split = ""  # Doesn't matter for augmentedDsprites
    config.val_split = ""  # Doesn't matter for augmentedDsprites

    config.pp_train = f'value_range(-1, 1, 0, 1)|move_key("label_shape", "label")|keep(["image", "label"])'
    config.pp_eval = f'value_range(-1, 1, 0, 1)|move_key("label_shape", "label")|keep(["image", "mask", "label"])'

    config.aug_dsprites = config_dict.ConfigDict()

    config.aug_dsprites.square_distribution = config_dict.ConfigDict()
    config.aug_dsprites.square_distribution.orientation = f"biuniform(low1=0.0, high1={math.pi / 8}, low2={math.pi / 4}, high2={4*math.pi / 5})"
    config.aug_dsprites.square_distribution.scale = (
        "biuniform(low1=0.45, high1=0.65, low2=0.89, high2=1.0)"
    )
    config.aug_dsprites.square_distribution.x_position = (
        "biuniform(low1=0.15, high1=0.6, low2=0.4, high2=0.9)"
    )
    config.aug_dsprites.square_distribution.y_position = (
        "truncated_normal(minval=0.05, maxval=0.95, loc=0.5, scale=0.15)"
    )

    config.aug_dsprites.ellipse_distribution = config_dict.ConfigDict()
    config.aug_dsprites.ellipse_distribution.orientation = (
        f"uniform(low={math.pi / 3}, high={2*math.pi / 3})"
    )
    config.aug_dsprites.ellipse_distribution.scale = "uniform(low=0.1, high=0.9)"
    config.aug_dsprites.ellipse_distribution.x_position = "uniform(low=0.25, high=0.75)"
    config.aug_dsprites.ellipse_distribution.y_position = "uniform(low=0.25, high=0.75)"

    config.aug_dsprites.heart_distribution = config_dict.ConfigDict()
    config.aug_dsprites.heart_distribution.orientation = (
        f"biuniform(low1=0.0, high1={math.pi/2}, low2={math.pi}, high2={math.pi*1.25})"
    )
    config.aug_dsprites.heart_distribution.scale = (
        "truncated_normal(loc=0.8, scale=0.05, minval=0.45, maxval=1.05)"
    )
    config.aug_dsprites.heart_distribution.x_position = "uniform(low=0.1, high=0.8)"
    config.aug_dsprites.heart_distribution.y_position = (
        "biuniform(low1=0.0, high1=0.3, low2=0.5, high2=0.8)"
    )

    config.aug_dsprites.heart_distribution.unnormalised_shape_prob = 1 / 3
    config.aug_dsprites.square_distribution.unnormalised_shape_prob = 1 / 3
    config.aug_dsprites.ellipse_distribution.unnormalised_shape_prob = 1 / 3

    return config


def add_aug_dsprites_config_v2(
    config: config_dict.ConfigDict,
) -> config_dict.ConfigDict:
    config.train_split = ""  # Doesn't matter for augmentedDsprites
    config.val_split = ""  # Doesn't matter for augmentedDsprites

    config.pp_train = f'value_range(-1, 1, 0, 1)|move_key("label_shape", "label")|keep(["image", "label"])'
    config.pp_eval = f'value_range(-1, 1, 0, 1)|move_key("label_shape", "label")|keep(["image", "mask", "label"])'

    config.aug_dsprites = config_dict.ConfigDict()

    config.aug_dsprites.square_distribution = config_dict.ConfigDict()
    config.aug_dsprites.square_distribution.orientation = (
        f"uniform(low=0.0, high={math.pi * 2})"
    )
    config.aug_dsprites.square_distribution.scale = (
        "truncated_normal(minval=0.55, maxval=1.0, loc=0.75, scale=0.2)"
    )
    config.aug_dsprites.square_distribution.x_position = "uniform(low=0.5, high=0.95)"
    config.aug_dsprites.square_distribution.y_position = "uniform(low=0.5, high=0.95)"

    config.aug_dsprites.ellipse_distribution = config_dict.ConfigDict()
    config.aug_dsprites.ellipse_distribution.orientation = (
        f"uniform(low=0.0, high={math.pi / 2})"
    )
    config.aug_dsprites.ellipse_distribution.scale = (
        "truncated_normal(minval=0.5, maxval=0.85, loc=0.65, scale=0.15)"
    )
    config.aug_dsprites.ellipse_distribution.x_position = (
        "truncated_normal(minval=0.1, maxval=0.9, loc=0.5, scale=0.25)"
    )
    config.aug_dsprites.ellipse_distribution.y_position = (
        "truncated_normal(minval=0.35, maxval=0.65, loc=0.5, scale=0.15)"
    )

    config.aug_dsprites.heart_distribution = config_dict.ConfigDict()
    config.aug_dsprites.heart_distribution.orientation = f"delta(value=0.0)"
    config.aug_dsprites.heart_distribution.scale = "uniform(low=0.9, high=1.0)"
    config.aug_dsprites.heart_distribution.x_position = "uniform(low=0.1, high=0.5)"
    config.aug_dsprites.heart_distribution.y_position = (
        "biuniform(low1=0.1, high1=0.3, low2=0.7, high2=0.9)"
    )

    config.aug_dsprites.heart_distribution.unnormalised_shape_prob = 1 / 3
    config.aug_dsprites.square_distribution.unnormalised_shape_prob = 1 / 3
    config.aug_dsprites.ellipse_distribution.unnormalised_shape_prob = 1 / 3

    return config
