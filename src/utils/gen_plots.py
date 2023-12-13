import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from absl import logging
from jax import numpy as jnp
from jax import random
from scipy.stats import gaussian_kde

from src.transformations.affine import (
    gen_affine_matrix_no_shear,
    transform_image_with_affine_matrix,
)
from src.utils.plotting import rescale_for_imshow


def plot_gen_model_training_metrics(history):
    colors = sns.color_palette("husl", 3)

    # plot the training history
    (
        steps,
        loss,
        log_p_η_x_hat,
        mae,
        lr_gen,
    ) = history.collect("steps", "loss", "log_p_η_x_hat", "mae", "lr_gen")
    mae_loss_mult = history.collect("mae_loss_mult")
    steps_test, loss_test, log_p_η_x_hat_test, mae_test = history.collect(
        "steps", "loss_test", "log_p_η_x_hat_test", "mae_test"
    )

    n_plots = 4
    fig, axs = plt.subplots(
        n_plots, 1, figsize=(15, n_plots * 3.0), dpi=100, sharex=True
    )

    for train_metric, test_metric, metric_name, ax in zip(
        [loss, log_p_η_x_hat, mae],
        [loss_test, log_p_η_x_hat_test, mae_test],
        ["loss", "log_p_η_x_hat", "mae"],
        axs,
    ):
        ax.plot(
            steps, train_metric, label=f"train {train_metric[-1]:.4f}", color=colors[0]
        )
        ax.plot(
            steps_test,
            test_metric,
            label=f"test  {test_metric[-1]:.4f}",
            color=colors[1],
        )
        ax.legend()
        ax.set_title(metric_name)

    # Schedule axis:
    lr_ax = axs[-1]
    multiplier_ax = lr_ax.twinx()
    # par2 = host.twinx()

    (p1,) = lr_ax.plot(
        steps, lr_gen, "--", label=f"lr_gen {lr_gen[-1]:.4f}", color=colors[0]
    )
    (p2,) = multiplier_ax.plot(
        steps,
        mae_loss_mult,
        label=f"mae_loss_mult {mae_loss_mult[-1]:.4f}",
        color=colors[2],
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
    lr_ax.tick_params(axis="y", colors=p1.get_color(), **tkw)
    multiplier_ax.tick_params(axis="y", colors=p2.get_color(), **tkw)
    # par2.tick_params(axis='y', colors=p3.get_color(), **tkw)

    axs[-1].set_xlim(min(steps), max(steps))
    axs[-1].set_xlabel("Steps")

    for ax in axs:
        ax.grid(color=(0.9, 0.9, 0.9))

    return fig


def plot_gen_dists(x, prototype_function, rng, gen_model, gen_params, config, n=10_000):
    # function to plot the histograms of p(η|x_hat) in each dimmension
    proto_rng, sample_rng = random.split(rng)
    η = prototype_function(x, proto_rng)
    η_aff_mat = gen_affine_matrix_no_shear(η)
    η_aff_mat_inv = jnp.linalg.inv(η_aff_mat)
    xhat = transform_image_with_affine_matrix(
        x, η_aff_mat_inv, order=config.interpolation_order
    )

    p_H_x_hat = gen_model.apply({"params": gen_params}, xhat)

    ηs_p = p_H_x_hat.sample(seed=sample_rng, sample_shape=(n,))

    transform_param_dim = η.shape[0]
    fig, axs = plt.subplots(
        1, transform_param_dim + 2, figsize=(3 * (transform_param_dim + 2), 3)
    )

    axs[0].imshow(rescale_for_imshow(x), cmap="gray")
    axs[1].imshow(rescale_for_imshow(xhat), cmap="gray")

    for i, ax in enumerate(axs[2:]):
        THRESHOLD = 0.01
        try:
            # First fit a KDE estimate and use it to filter out outliers
            kde_ = gaussian_kde(ηs_p[:, i])
            ηs_p_ = ηs_p[kde_(ηs_p[:, i]) > THRESHOLD, i]

            # Then fit a KDE and plot it, as well as the histogram with outliers removed
            kde = gaussian_kde(ηs_p_)

            xs = np.linspace(ηs_p_.min(), ηs_p_.max(), 1000)
            ax.plot(xs, kde(xs), color="C0")

            ax.hist(ηs_p_, bins=100, density=True, alpha=0.5, color="C0")

            ax.set_xlim(xs.min(), xs.max())

        except Exception as e:
            logging.warning(f"Failed to plot KDE for dim {i}: {e}")
            try:
                ax.hist(ηs_p[:, i], bins=100, density=True, alpha=0.5, color="C0")
            except Exception as e:
                logging.warning(f"Failed to plot histogram for dim {i}: {e}")

        ax.axvline(η[i], color="C1", linestyle="--")

        ax.set_title(f"dim {i}")

    plt.tight_layout()

    return fig
