import orbax.checkpoint
import wandb
from absl import flags, logging
from ml_collections import config_dict

FLAGS = flags.FLAGS


def load_checkpoint(checkpoint_path, init_state, config):
    logging.info(f"Loading model checkpoint from {checkpoint_path}.")
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = checkpointer.restore(
        checkpoint_path,
        item={"state": init_state, "config": config},
    )
    final_state, config_ = ckpt["state"], ckpt["config"]
    config_ = config_dict.ConfigDict(config_)
    if config_.to_json() != config.to_json():
        logging.warning(
            "The config loaded from the checkpoint is different from the one passed as a flag.\n"
            "Loaded config:\n"
            f"{config_}\n"
            "Passed config:\n"
            f"{config}"
        )
    return final_state, config_


def assert_inf_gen_compatiblity(inf_config, gen_config):
    assert inf_config.interpolation_order == gen_config.interpolation_order
    assert inf_config.get("augment_bounds", None) == gen_config.get(
        "augment_bounds", None
    )
    assert inf_config.get("augment_offset", None) == gen_config.get(
        "augment_offset", None
    )
    assert inf_config.model.squash_to_bounds == gen_config.model.squash_to_bounds
    assert inf_config.translate_last == gen_config.translate_last
    assert inf_config.get("shuffle", "loaded") == gen_config.get("shuffle", "loaded")
    assert inf_config.get("repeat_after_batching", False) == gen_config.get(
        "repeat_after_batching", False
    )


def duplicated_run(config):
    fake_run = wandb.init(
        mode="disabled",
        config=config.to_dict(),
    )
    # ^ we create this fake run to get the config dict in the same format as the existing runs in wandb

    logging.info("Checking if config already exists in wandb.")
    runs = wandb.Api().runs(f"{FLAGS.wandb_entity}/{FLAGS.wandb_project}")
    finished_runs = [run for run in runs if run.state == "finished"]
    frozen_config = config_dict.FrozenConfigDict(fake_run.config)
    logging.info(f"checking {len(finished_runs)} runs")

    for run in finished_runs:
        run_config = config_dict.FrozenConfigDict(run.config)
        if frozen_config == run_config:
            logging.info(
                f"Found matching config in run with id {run.id} and name {run.name}."
                "Skipping training. Use --rerun to rerun the experiment. Config was: \n"
                f"{frozen_config}"
            )
            return True

    return False


def format_thousand(num):
    thousands = num // 1000
    remainder = num % 1000 // 100
    return f"{thousands}k{remainder if remainder != 0 else ''}"
