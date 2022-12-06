import sys

import jax
from jax import numpy as jnp
from jax import random, lax
import flax
from flax.training import checkpoints
import optax
import distrax
import tensorflow as tf
import tensorflow_datasets as tfds
from clu import preprocess_spec, parameter_overview
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd

from NeurReps import preprocess_utils
from src.utils.training import TrainState
from src.utils.plotting import plot_img_array


BATCH_SIZE = 256
BATCH_SIZE_VAL = 100
PLOT_NUM_EXAMPLES = 16
# LOW_ANGLE = -45
# HIGH_ANGLE = 45
# SEED = 42

# seed = 42 + int(sys.argv[1])
# low_angle = int(sys.argv[2])
# high_angle = int(sys.argv[3])


def run(seed, low_angle, high_angle):
    print(seed, low_angle, high_angle)
    # raise Exception("STOP")

    rng = random.PRNGKey(seed)

    rng, shuffle_rng, pp_rng = random.split(rng, num=3)

    pp_train = f'value_range(-1, 1)|random_rotate({low_angle}, {high_angle}, fill_value=-1)|keep(["image"])'

    def make_pp_with_rng(pp_str):
        preprocess_fn = preprocess_spec.parse(
            spec=pp_train, available_ops=preprocess_utils.all_ops())

        def preprocess_with_rng(example_index, features):
            example_index = tf.cast(example_index, tf.int32)
            features["rng"] = tf.random.experimental.stateless_fold_in(
                tf.cast(pp_rng, tf.int64), example_index)
            processed = preprocess_fn(features)
            # del processed["rng"]
            return processed

        return preprocess_with_rng

    train_pp_fn = make_pp_with_rng(pp_train)


    train_ds = tfds.load('MNIST', split='train[10000:]')
    train_ds = train_ds.shuffle(50_000, seed=shuffle_rng[0])
    # NOTE: ordering of PP and repeat is important!
    # PP first -> same seed each time we see same example.
    # PP last -> diff seed each time we see same example. Effectively more data.
    train_ds = train_ds.enumerate().map(train_pp_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(batch_size=BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.as_numpy_iterator()


    plot_img_array(next(train_ds)['image'][:PLOT_NUM_EXAMPLES], ncol=PLOT_NUM_EXAMPLES, title='Train data', pad_value=1);


    val_pp_fn = make_pp_with_rng(pp_train)

    val_ds = tfds.load('MNIST', split='train[:10000]')
    val_ds = val_ds.shuffle(10_000, seed=27)
    val_ds = val_ds.enumerate().map(val_pp_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(PLOT_NUM_EXAMPLES)
    val_ds = val_ds.take(1)
    val_ds = val_ds.as_numpy_iterator()
    plot_data = next(val_ds)

    fig = plot_img_array(plot_data['image'], ncol=PLOT_NUM_EXAMPLES, pad_value=1, padding=1);

    # ### Build Model

    from NeurReps.models import LIVAE, make_livae_loss, calculate_livae_elbo


    model = LIVAE()

    init_rng, rng = random.split(rng)
    variables = model.init(init_rng, jnp.ones((28, 28, 1)), rng)

    print(parameter_overview.get_parameter_overview(variables))


    STEPS =  15001
    LR = 1e-4
    WD = 1e-4
    BETA = 10.

    lr = optax.warmup_cosine_decay_schedule(
        init_value=LR,
        peak_value=LR*10,
        warmup_steps=500,
        decay_steps=STEPS,
        end_value=LR,
    )

    optim = optax.adamw
    optim = optax.inject_hyperparams(optim)
    # This ^ allows us to access the lr as opt_state.hyperparams['learning_rate'].

    β = optax.warmup_cosine_decay_schedule(
        init_value=BETA,
        peak_value=BETA,
        warmup_steps=1,
        decay_steps=STEPS-1,
        end_value=1.,
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optim(learning_rate=lr, weight_decay=WD),
        model_state=flax.core.frozen_dict.FrozenDict({}),
        β_val_or_schedule=β,
    )


    @jax.jit
    def train_step(state, x_batch, rng):
        loss_fn = make_livae_loss(model, x_batch)
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        (loss, metrics), grads = grad_fn(
            state.params, rng, state.β,
        )

        return state.apply_gradients(grads=grads), (loss, metrics)

    # ### Training
    sampling_rng = random.PRNGKey(0)

    for step in tqdm(range(STEPS)):
        x_batch = next(train_ds)['image']
        train_step_rng, rng = random.split(rng)
        state, (loss, metrics) = train_step(state, x_batch, rng)

        if step % 500 == 0:
            print(f"Step {step:5}: loss={loss:8.3f}")

            x_batch = plot_data['image']
            # print("Original images")
            orig_fig = plot_img_array(x_batch, ncol=PLOT_NUM_EXAMPLES, pad_value=1, padding=1)
            def reconstruct(x, prototype=False, sample_z=False,
                            sample_xhat=False, sample_θ=False):
                rng = random.fold_in(sampling_rng, lax.axis_index('batch'))
                return model.apply({'params': state.params}, x, rng,
                                prototype=prototype, sample_z=sample_z,
                                sample_xhat=sample_xhat, sample_θ=sample_θ,
                                method=model.reconstruct)

            x_recon_modes = jax.vmap(
                reconstruct, axis_name='batch', in_axes=(0, None, None, None, None)
            )(x_batch, False, False, False, True)
            # print("Reconstructions")
            recon_fig = plot_img_array(x_recon_modes, ncol=PLOT_NUM_EXAMPLES, pad_value=1, padding=1)

            x_recon_proto_modes = jax.vmap(
                reconstruct, axis_name='batch', in_axes=(0, None, None, None, None)
            )(x_batch, True, False, False, True)
            # print("Prototypes")
            proto_fig = plot_img_array(x_recon_proto_modes, ncol=PLOT_NUM_EXAMPLES, pad_value=1, padding=1)


    orig_fig.savefig(f"results/orig_{low_angle}_{high_angle}_{seed}.pdf", bbox_inches='tight', pad_inches=0, dpi=400)
    recon_fig.savefig(f"results/recon_{low_angle}_{high_angle}_{seed}.pdf", bbox_inches='tight', pad_inches=0, dpi=400)
    proto_fig.savefig(f"results/proto_{low_angle}_{high_angle}_{seed}.pdf", bbox_inches='tight', pad_inches=0, dpi=400)


    CKPT_DIR = f'results/ckpts/ckpt_{low_angle}_{high_angle}_{seed}'
    checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=step)

    # ### Eval


    val_pp_fn = make_pp_with_rng(pp_train)

    val_ds = tfds.load('MNIST', split='train[:10000]')
    val_ds = val_ds.shuffle(10_000, seed=shuffle_rng[0])
    val_ds = val_ds.enumerate().map(val_pp_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE_VAL)
    val_ds = val_ds.as_numpy_iterator()


    @jax.jit
    def angle_and_elbo(x, batch_rng):
        rng = random.fold_in(batch_rng, lax.axis_index('batch'))
        q_z_x, q_θ_x, p_x_xhat_θ, _, p_θ_z, p_z = model.apply(
            {'params': state.params}, x, rng,
        )

        θ = q_θ_x.sample(sample_shape=(10,), seed=rng).mean()
        elbo, _ = calculate_livae_elbo(x, q_z_x, q_θ_x, p_x_xhat_θ, p_z, p_θ_z, state.β)

        return θ, elbo


    eval_key = random.PRNGKey(0)
    θs = []
    elbos = []
    for val_batch in val_ds:
        batch_θ, batch_elbo = jax.vmap(angle_and_elbo, axis_name='batch', in_axes=(0, None))(val_batch['image'], eval_key)
        θs.append(batch_θ)
        elbos.append(batch_elbo)
    θs = jnp.concatenate(θs, axis=0)
    θs = jnp.degrees(θs)
    jnp.save(f"results/angles_{low_angle}_{high_angle}_{seed}.npy", θs)

    elbo = jnp.concatenate(elbos, axis=0).mean(axis=0)
    print(elbo)

    df = pd.DataFrame({'seed': [seed],
                    'start': [low_angle],
                    'stop': [high_angle],
                    'test': [elbo],
                    'train': [-loss]})
    df.to_csv('results/elbos.csv', mode='a', index=False, header=False)


if __name__ == '__main__':
    for seed in range(0, 1):
        for lower, higher in ((0, 0), (-15, 15), (-30, 30), (-45, 45), (-90, 90), (-180, 180)):
            # if seed == 1 and (lower == 0 or lower == -15):
            #     continue
            try:
                run(seed + 42, lower, higher)
            except:
                continue