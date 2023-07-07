import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".45"

import random as py_random
import functools
from absl import app
from absl import flags
from absl import logging
import jax
import jax.random as random
from jax.config import config
from ml_collections import config_flags

import jax.numpy as jnp

jnp.log(jnp.exp(1) - 1.0)
# TODO: figure out why we get CUDA failures this ^ isn't here.

import tensorflow as tf
import flax

# from flax.training import checkpoints
import orbax.checkpoint
from absl import logging

from src.utils.training import setup_model, get_dataset_splits, train_loop
import src.utils.input as input_utils
from experiments.configs.ssilvae_mnist import get_config as get_ssilvae_config
from experiments.configs.vae_mnist import get_config as get_vae_config


logging.set_verbosity(logging.INFO)


FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("angle", 45, "Angle for rotation.")
flags.DEFINE_integer("num_trn", 50000, "Number of training examples.")
flags.DEFINE_integer("total_steps", 7501, "Number of training steps.")
flags.DEFINE_enum("wandb_mode", "online", ["online", "offline", "disabled"], "Mode for wandb.run.")
flags.DEFINE_list("wandb_tags", [], "Tags for wandb.run.")
flags.DEFINE_string("wandb_notes", "", "Notes for wandb.run.")
flags.DEFINE_string("wandb_project", "iclr2024experiments", "Project for wandb.run.")
flags.DEFINE_string("wandb_entity", "invariance-learners", "Entity for wandb.run.")


def random_word(length=5):
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"

    return "".join(py_random.choice((consonants, vowels)[i % 2]) for i in range(length))


def main(argv):
    common_tag = random_word(9)

    # Set the seed
    rng = random.PRNGKey(FLAGS.seed)
    data_rng, model_rng = random.split(rng)

    ssilvae_config = get_ssilvae_config(f"{FLAGS.angle},{FLAGS.num_trn},{FLAGS.total_steps},1")
    ssilvae_config.seed = FLAGS.seed
    vae_config = get_vae_config(f"{FLAGS.angle},{FLAGS.num_trn}")
    vae_config.seed = FLAGS.seed
    vae_config.total_steps = FLAGS.total_steps

    train_ds, val_ds, _ = get_dataset_splits(
        vae_config, data_rng, vae_config.batch_size, vae_config.batch_size
    )

    vae_model, vae_state = setup_model(vae_config, model_rng, train_ds)
    ssilvae_model, ssilvae_state = setup_model(ssilvae_config, model_rng, train_ds)

    ssilvae_best_state, ssilvae_final_state = train_loop(
        ssilvae_config,
        ssilvae_model,
        ssilvae_state,
        train_ds,
        val_ds,
        wandb_kwargs={
            "mode": FLAGS.wandb_mode,
            "tags": FLAGS.wandb_tags + [common_tag, "ssilvae"],
            "notes": FLAGS.wandb_notes,
            "project": FLAGS.wandb_project,
            "entity": FLAGS.wandb_entity,
        },
    )

    train_ds_1_epoch, valid_ds_1_epoch, _ = get_dataset_splits(vae_config, data_rng, 500, 500, 1)

    @jax.pmap
    @functools.partial(jax.vmap, in_axes=(0,), axis_name="image")
    def get_proto(x):
        local_rng = random.fold_in(rng, jax.lax.axis_index("image"))
        return ssilvae_model.apply(
            {"params": ssilvae_best_state.params},
            x,
            local_rng,
            return_xhat=True,
            reconstruct_xhat=False,
            sample_η_proto=False,
            sample_η_recon=False,
            sample_z=False,
            sample_xrecon=False,
            sample_xhat=False,
            α=ssilvae_best_state.α,
            train=False,
            method=ssilvae_model.reconstruct,
        )

    def modify_dataset(dataset_iter):
        modified_dataset = []
        for batch in dataset_iter:
            modified_images = get_proto(batch["image"])
            batch["image"] = modified_images
            modified_dataset.append(batch)

        return modified_dataset

    modified_train_data = modify_dataset(
        input_utils.start_input_pipeline(train_ds_1_epoch, vae_config.get("prefetch_to_device", 1))
    )
    modified_valid_data = modify_dataset(
        input_utils.start_input_pipeline(valid_ds_1_epoch, vae_config.get("prefetch_to_device", 1))
    )

    def drop_mask(batch):
        batch.pop("mask", None)
        return batch

    valid_dataset = (
        tf.data.Dataset.from_generator(
            lambda: (batch for batch in modified_valid_data),
            output_signature={
                "image": tf.TensorSpec(shape=(1, 500, 28, 28, 1), dtype=tf.float32),  # image shape
                "label": tf.TensorSpec(shape=(1, 500), dtype=tf.int32),  # label shape
                "mask": tf.TensorSpec(shape=(1, 500), dtype=tf.float32),  # mask shape
            },
        )
        .unbatch()
        .unbatch()
        .filter(lambda x: x["mask"] == 1.0)
        .map(drop_mask)
    )

    train_dataset = (
        tf.data.Dataset.from_generator(
            lambda: (batch for batch in modified_train_data),
            output_signature={
                "image": tf.TensorSpec(shape=(1, 500, 28, 28, 1), dtype=tf.float32),  # image shape
                "mask": tf.TensorSpec(shape=(1, 500), dtype=tf.float32),  # mask shape
            },
        )
        .unbatch()
        .unbatch()
        .filter(lambda x: x["mask"] == 1.0)
        .map(drop_mask)
    )

    data_rng, train_ds_rng = jax.random.split(data_rng)
    train_ds_rng = jax.random.fold_in(train_ds_rng, jax.process_index())

    train_ds_aug = input_utils.get_data(
        dataset=train_dataset,
        split="train",
        rng=train_ds_rng,
        process_batch_size=vae_config.batch_size,
        preprocess_fn=None,
        shuffle=vae_config.get("shuffle", "preprocessed"),  # type: ignore
        shuffle_buffer_size=vae_config.shuffle_buffer_size,  # type: ignore
        repeat_after_batching=vae_config.get("repeat_after_batching", True),  # type: ignore
        prefetch_size=vae_config.get("prefetch_to_host", 2),  # type: ignore
        drop_remainder=False,
        num_epochs=None,
    )

    data_rng, val_ds_rng = jax.random.split(data_rng)
    val_ds_rng = jax.random.fold_in(val_ds_rng, jax.process_index())

    valid_ds_aug = input_utils.get_data(
        dataset=valid_dataset,
        split="valid",  # type: ignore
        rng=val_ds_rng,
        process_batch_size=vae_config.batch_size,
        preprocess_fn=None,
        cache=vae_config.get("val_cache", "batched"),  # type: ignore
        num_epochs=1,
        repeat_after_batching=True,
        shuffle=False,
        prefetch_size=vae_config.get("prefetch_to_host", 2),  # type: ignore
        drop_remainder=False,
    )

    augvae_best_state, augvae_final_state = train_loop(
        vae_config,
        vae_model,
        vae_state,
        train_ds_aug,
        valid_ds_aug,
        wandb_kwargs={
            "mode": FLAGS.wandb_mode,
            "tags": FLAGS.wandb_tags + [common_tag, "augvae"],
            "notes": FLAGS.wandb_notes,
            "project": FLAGS.wandb_project,
            "entity": FLAGS.wandb_entity,
        },
    )


if __name__ == "__main__":
    config.config_with_absl()
    app.run(main)
