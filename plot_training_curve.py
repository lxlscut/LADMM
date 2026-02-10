#!/usr/bin/env python3
import argparse
import os
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


EPOCH_LINE_RE = re.compile(
    r"^Epoch\s+(\d+):\s*train_loss=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)
METRIC_LINE_RE = re.compile(
    r"^acc:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
    r"nmi:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
    r"kappa[:\s]+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


def parse_log(log_path: str):
    loss_epochs = []
    train_losses = []
    loss_steps = []

    metric_epochs = []
    acc_values = []
    nmi_values = []
    kappa_values = []
    metric_steps = []

    current_epoch = None
    current_step = 1
    previous_epoch = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            epoch_match = EPOCH_LINE_RE.match(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                if previous_epoch is not None and current_epoch <= previous_epoch:
                    current_step += 1
                previous_epoch = current_epoch
                loss_epochs.append(current_epoch)
                train_losses.append(float(epoch_match.group(2)))
                loss_steps.append(current_step)
                continue

            metric_match = METRIC_LINE_RE.match(line)
            if metric_match and current_epoch is not None:
                metric_epochs.append(current_epoch)
                acc_values.append(float(metric_match.group(1)))
                nmi_values.append(float(metric_match.group(2)))
                kappa_values.append(float(metric_match.group(3)))
                metric_steps.append(current_step)

    return (
        loss_epochs,
        train_losses,
        loss_steps,
        metric_epochs,
        acc_values,
        nmi_values,
        kappa_values,
        metric_steps,
    )


def build_default_output_path(log_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(log_path))[0]
    return os.path.join(os.path.dirname(log_path), f"{base_name}_step2_train_loss_metric_curve.pdf")


def ema_smooth(values, alpha):
    if not values:
        return []
    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(alpha * float(value) + (1.0 - alpha) * smoothed[-1])
    return smoothed


def plot_loss_curves(
    loss_epochs,
    train_losses,
    loss_steps,
    metric_epochs,
    acc_values,
    nmi_values,
    kappa_values,
    metric_steps,
    save_path,
    step_idx=2,
    smooth_alpha=0.35,
    export_dpi=600,
):
    max_step = max(loss_steps) if loss_steps else 0
    if step_idx < 1 or step_idx > max_step:
        raise RuntimeError(f"Requested step {step_idx}, but only found {max_step} step(s).")
    if smooth_alpha <= 0.0 or smooth_alpha > 1.0:
        raise RuntimeError("--smooth_alpha must be in (0, 1].")

    step_epochs = [e for e, s in zip(loss_epochs, loss_steps) if s == step_idx]
    step_train_losses = [v for v, s in zip(train_losses, loss_steps) if s == step_idx]
    step_metric_epochs = [e for e, s in zip(metric_epochs, metric_steps) if s == step_idx]
    step_acc = [v for v, s in zip(acc_values, metric_steps) if s == step_idx]
    step_nmi = [v for v, s in zip(nmi_values, metric_steps) if s == step_idx]
    step_kappa = [v for v, s in zip(kappa_values, metric_steps) if s == step_idx]
    step_train_losses = ema_smooth(step_train_losses, smooth_alpha)
    step_acc = ema_smooth(step_acc, smooth_alpha)
    step_nmi = ema_smooth(step_nmi, smooth_alpha)
    step_kappa = ema_smooth(step_kappa, smooth_alpha)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.plot(step_epochs, step_train_losses, label="train_loss", linewidth=2.4, color="tab:blue")
    ax.set_xlabel("Epoch index", fontsize=20)
    ax.set_ylabel("Train loss", color="tab:blue", fontsize=20)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelcolor="tab:blue", labelsize=16)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, linestyle="--", alpha=0.4)

    ax_right = ax.twinx()
    if step_metric_epochs:
        ax_right.plot(step_metric_epochs, step_acc, linewidth=2.2, label="OA(%)", color="tab:orange")
        ax_right.plot(step_metric_epochs, step_nmi, linewidth=2.2, label="NMI", color="tab:green")
        ax_right.plot(step_metric_epochs, step_kappa, linewidth=2.2, label=r"$\mathcal{K}$", color="tab:red")
    ax_right.set_ylabel(r"OA(%) / NMI / $\mathcal{K}$", color="tab:red", fontsize=20)
    ax_right.tick_params(axis="y", labelcolor="tab:red", labelsize=16)
    ax_right.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_right.yaxis.set_label_position("right")
    ax_right.yaxis.tick_right()

    lines_left, labels_left = ax.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        frameon=True,
        framealpha=0.95,
        fontsize=15,
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=export_dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Parse training log and draw selected-step loss curves.")
    parser.add_argument("--log_file", type=str, required=True, help="Path to result_seed*.txt log file")
    parser.add_argument("--save_path", type=str, default=None, help="Output figure path (.pdf)")
    parser.add_argument("--step_idx", type=int, default=2, help="Which training step to plot (1-based)")
    parser.add_argument("--smooth_alpha", type=float, default=0.35, help="EMA smoothing factor in (0,1].")
    parser.add_argument("--dpi", type=int, default=600, help="Export dpi for high-resolution output.")
    args = parser.parse_args()

    save_path = args.save_path or build_default_output_path(args.log_file)
    (
        loss_epochs,
        train_losses,
        loss_steps,
        metric_epochs,
        acc_values,
        nmi_values,
        kappa_values,
        metric_steps,
    ) = parse_log(args.log_file)

    if not loss_epochs:
        raise RuntimeError("No epoch loss lines matched. Please check log format.")

    plot_loss_curves(
        loss_epochs,
        train_losses,
        loss_steps,
        metric_epochs,
        acc_values,
        nmi_values,
        kappa_values,
        metric_steps,
        save_path,
        step_idx=args.step_idx,
        smooth_alpha=args.smooth_alpha,
        export_dpi=args.dpi,
    )
    print(f"Saved curve figure to: {save_path}")
    print(f"Parsed train_loss points: {len(loss_epochs)}, metric points: {len(metric_epochs)}")


if __name__ == "__main__":
    main()
