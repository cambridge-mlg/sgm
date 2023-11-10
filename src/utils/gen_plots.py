import matplotlib.pyplot as plt
import seaborn as sns


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
