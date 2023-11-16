from typing import Optional
import jax
import jax.numpy as jnp
from matplotlib.figure import Figure
import numpy as np
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import seaborn as sns

from src.transformations.affine import gen_affine_matrix_no_shear, transform_image_with_affine_matrix


def plot_transform_gen_model_training_metrics(history) -> Figure:
    colors = sns.color_palette("husl", 3)

    # plot the training history
    steps, loss, log_p_η_x_hat, mae, lr_gen,  = history.collect(
        "steps",
        "loss",
        "log_p_η_x_hat",
        "mae",
        "lr_gen"
    )
    mae_loss_mult = history.collect("mae_loss_mult")
    steps_test, loss_test, log_p_η_x_hat_test, mae_test = history.collect(
        "steps", "loss_test", "log_p_η_x_hat_test", "mae_test"
    )

    n_plots = 4
    fig, axs = plt.subplots(
        n_plots, 1, figsize=(15, n_plots * 3.0), dpi=300, sharex=True
    )

    for train_metric, test_metric, metric_name, ax in zip(
        [loss, log_p_η_x_hat, mae],
        [loss_test, log_p_η_x_hat_test, mae_test],
        ["loss", "log_p_η_x_hat", "mae"],
        axs,
    ):
        ax.plot(steps, train_metric, label=f"train {train_metric[-1]:.4f}", color=colors[0])
        ax.plot(steps_test, test_metric, label=f"test  {test_metric[-1]:.4f}", color=colors[1])
        ax.legend()
        ax.set_title(metric_name)

    # Schedule axis:
    lr_ax = axs[-1]
    multiplier_ax = lr_ax.twinx()
    # par2 = host.twinx()

    p1, = lr_ax.plot(steps, lr_gen, "--", label=f"lr_gen {lr_gen[-1]:.4f}", color=colors[0])
    p2, = multiplier_ax.plot(
        steps,
        mae_loss_mult,
        label=f"mae_loss_mult {mae_loss_mult[-1]:.4f}",
        color=colors[2]
    )
    lines = [p1, p2]
    lr_ax.legend(lines, [l.get_label() for l in lines])

    lr_ax.set_yscale("log")
    multiplier_ax.set_yscale("log")
    # par2.set_yscale("log")

    lr_ax.set_ylabel(f"LR")
    # par1.set_ylabel("σ LR")
    multiplier_ax.set_ylabel("Multipliers")

    lr_ax.yaxis.label.set_color(p1.get_color())
    multiplier_ax.yaxis.label.set_color(p2.get_color())
    # par2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    lr_ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    multiplier_ax.tick_params(axis='y', colors=p2.get_color(), **tkw)
    # par2.tick_params(axis='y', colors=p3.get_color(), **tkw)

    axs[-1].set_xlim(min(steps), max(steps))
    axs[-1].set_xlabel("Steps")

    for ax in axs:
        ax.grid(color=(0.9, 0.9, 0.9))
    return fig



# function to plot the histograms of p(η|x_hat) in each dimmension
def plot_generative_histograms(x, rng, prototype_function, transform_gen_distribution_function, interpolation_order, fig: Optional[Figure]=None) -> Figure:
    rng_proto, rng_gen_samples = jax.random.split(rng, 2)
    η = prototype_function(x, rng_proto)
    η_aff_mat = gen_affine_matrix_no_shear(η)
    η_aff_mat_inv = jnp.linalg.inv(η_aff_mat)
    xhat = transform_image_with_affine_matrix(x, η_aff_mat_inv, order=interpolation_order)

    # p_H_x_hat = gen_model.apply({"params": gen_final_state.params}, xhat)
    p_H_x_hat = transform_gen_distribution_function(xhat)
    
    ηs_p = p_H_x_hat.sample(seed=rng_gen_samples, sample_shape=(20_000,))

    transform_param_dim = η.shape[0]
    if fig is None:
        fig = plt.figure(figsize=(3*(transform_param_dim+2), 3))
    axs = fig.subplots(nrows=1, ncols=transform_param_dim + 2)

    axs[0].imshow(x, cmap='gray', vmin=-1, vmax=1)
    axs[0].axis('off')
    axs[0].set_title("x")
    axs[1].imshow(xhat, cmap='gray', vmin=-1, vmax=1)
    axs[1].axis('off')
    axs[1].set_title("x_hat")

    for i, ax in enumerate(axs[2:]):
        x = np.linspace(ηs_p[:, i].min(), ηs_p[:, i].max(), 1000)

        # plot p(η|x_hat)
        ax.hist(ηs_p[:, i], bins=100, density=True, alpha=0.5, color="C0")
        kde = gaussian_kde(ηs_p[:, i])
        ax.plot(x, kde(x), color="C0")

        # make a axvline to plot η (the transformation inferred by transformation inference net.)
        ax.axvline(η[i], color="C1", linestyle="--")

        ax.set_title(f"dim {i}")
        ax.set_xlim(x.min(), x.max())

    plt.tight_layout()
    return fig
