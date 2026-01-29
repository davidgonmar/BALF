"""
This experiment serves as a simple experiment to illustrate the benefits of
activation-aware low-rank factorization over standard low-rank factorization.
It sweeps the rank used in factorization for three simple models:
- A single conv layer
- A single grouped conv layer
- A single linear layer
It plots the normalized squared Frobenius error in the outputs of the approximated
model vs the original model, for both standard and activation-aware factorization.
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from lib.factorization.factorize import (
    to_low_rank_activation_aware_manual,
    to_low_rank_manual,
    collect_activation_cache,
)

from lib.utils import seed_everything, cifar10_mean, cifar10_std

# to remove folders
import shutil
from pathlib import Path
import os

# plot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "font.size": 14,  # default text size
        "axes.titlesize": 18,  # axes title
        "axes.labelsize": 18,  # x and y labels
        "xtick.labelsize": 16,  # x tick labels
        "ytick.labelsize": 16,  # y tick labels
        "legend.fontsize": 13,  # legend
        "legend.title_fontsize": 15,
    }
)

# Define simple models for testing


# A single conv layer
class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride=1, padding=None, bias=False):
        super().__init__()
        if padding is None:
            padding = ksize // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, ksize, stride=stride, padding=padding, bias=bias
        )

    def forward(self, x):
        return self.conv(x)


# A single grouped conv layer
class SingleGroupedConv(nn.Module):
    def __init__(
        self, in_ch, out_ch, ksize, stride=1, padding=None, bias=False, groups=1
    ):
        super().__init__()
        if padding is None:
            padding = ksize // 2
        self.gconv = nn.Conv2d(
            in_ch,
            out_ch,
            ksize,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )

    def forward(self, x):
        return self.gconv(x)


# A single linear layer
class SingleLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.fc(x)


def matrixize_weight_for_rank(w: torch.Tensor):
    if w.dim() == 4:
        oc, ic_pg, kh, kw = w.shape
        return w.view(oc, ic_pg * kh * kw), oc, ic_pg * kh * kw
    elif w.dim() == 2:
        return w, w.shape[0], w.shape[1]
    else:
        raise ValueError(f"Unsupported weight shape {w.shape}")


def max_feasible_rank_module(mod: nn.Module) -> int:
    if isinstance(mod, nn.Conv2d):
        W = mod.weight.data
        oc, ic_pg, kh, kw = W.shape
        g = mod.groups
        out_pg = oc // g
        in_pg = ic_pg
        per_group_rank = min(out_pg, in_pg * kh * kw)
        return per_group_rank  # rank is per group in grouped convs
    elif isinstance(mod, nn.Linear):
        W = mod.weight.data
        _, o, i = matrixize_weight_for_rank(W)
        return min(o, i)
    else:
        raise ValueError(f"Unsupported module type {type(mod)}")


@torch.no_grad()
def frobenius_norm_outputs_squared(model: nn.Module, dl: DataLoader, device) -> float:
    model.eval()
    total_sq = 0.0
    for xb, _ in dl:
        xb = xb.to(device)
        yr = model(xb)
        total_sq += torch.sum(yr.float() ** 2).item()
    return total_sq


@torch.no_grad()
def output_frobenius_error_squared(model_ref, model_approx, dl, device, norm_ref_sq):
    model_ref.eval()
    model_approx.eval()
    total_sq = 0.0
    for xb, _ in dl:
        xb = xb.to(device)
        yr = model_ref(xb)
        ya = model_approx(xb)
        diff = (yr - ya).float()
        total_sq += torch.sum(diff * diff).item()
    return total_sq / (norm_ref_sq + 1e-12)


def sweep_layer(
    base_model,
    layer_key,
    dl_calib,
    dl_eval,
    device,
    sweep_max_rank,
    save_dir_tmp="./whitening-cache-tmp/",
):
    act_cache = collect_activation_cache(base_model, dl_calib, keys=[layer_key])
    norm_ref_sq = frobenius_norm_outputs_squared(base_model, dl_eval, device)

    rows = []
    # minimum ratio is 0.15
    min_rank = int(max(1, sweep_max_rank * 0.15))
    max_n_steps = 20  # fixed
    stepsize = int(max(1, (sweep_max_rank - min_rank) // max_n_steps))
    for r in range(min_rank, sweep_max_rank + 1, stepsize):
        ratio = r / sweep_max_rank
        cfg = {layer_key: {"name": "rank_ratio_to_keep", "value": ratio}}

        m_plain = to_low_rank_manual(base_model, cfg_dict=cfg, inplace=False)
        err_plain_sq = output_frobenius_error_squared(
            base_model, m_plain, dl_eval, device, norm_ref_sq
        )

        m_act = to_low_rank_activation_aware_manual(
            base_model, act_cache, cfg_dict=cfg, inplace=False, save_dir=save_dir_tmp
        )
        err_act_sq = output_frobenius_error_squared(
            base_model, m_act, dl_eval, device, norm_ref_sq
        )
        rows.append(
            {
                "ratio": ratio,
                "frob_output_error_sq_plain": err_plain_sq,
                "frob_output_error_sq_actaware": err_act_sq,
            }
        )
    # remove folder
    if os.path.exists(save_dir_tmp):
        shutil.rmtree(save_dir_tmp)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--in_ch", type=int, default=3)
    ap.add_argument(
        "--out_ch",
        type=int,
        default=24,
        help="for grouped conv, make this divisible by in_ch (e.g. 33 with in_ch=3)",
    )
    ap.add_argument("--ksize", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--bias", action="store_true")
    ap.add_argument(
        "--groups", type=int, default=3, help="use 3 for CIFAR depthwise-style conv"
    )
    ap.add_argument("--linear_out", type=int, default=128)
    ap.add_argument("--n_samples_cache", type=int, default=1024)
    ap.add_argument("--n_samples_eval", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=1024)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
    )
    ds_calib = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
    ds_eval = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
    dl_calib = DataLoader(
        Subset(ds_calib, torch.randperm(len(ds_calib))[: args.n_samples_cache]),
        batch_size=args.batch_size,
    )
    dl_eval = DataLoader(
        Subset(ds_eval, torch.randperm(len(ds_eval))[: args.n_samples_eval]),
        batch_size=args.batch_size,
    )

    base_conv = (
        SingleConv(
            args.in_ch, args.out_ch, args.ksize, stride=args.stride, bias=args.bias
        )
        .to(device)
        .eval()
    )
    base_gconv = (
        SingleGroupedConv(
            args.in_ch,
            args.out_ch,
            args.ksize,
            stride=args.stride,
            bias=args.bias,
            groups=args.groups,
        )
        .to(device)
        .eval()
    )
    base_linear = (
        SingleLinear(args.in_ch * 32 * 32, args.linear_out, bias=args.bias)
        .to(device)
        .eval()
    )

    fr_conv = max_feasible_rank_module(base_conv.conv)
    fr_gconv = max_feasible_rank_module(base_gconv.gconv)
    fr_linear = max_feasible_rank_module(base_linear.fc)

    rows_conv = sweep_layer(base_conv, "conv", dl_calib, dl_eval, device, fr_conv)
    rows_gconv = sweep_layer(base_gconv, "gconv", dl_calib, dl_eval, device, fr_gconv)
    rows_linear = sweep_layer(base_linear, "fc", dl_calib, dl_eval, device, fr_linear)

    plt.figure(figsize=(7.5, 3.7))

    layer_colors = {
        "Conv": "C0",
        "GConv": "C1",
        "Linear": "C2",
    }
    layer_markers = {
        "Conv": "o",
        "GConv": "s",
        "Linear": "^",
    }

    def plot_layer(rows, color, marker):
        plt.plot(
            [r["ratio"] for r in rows],
            [r["frob_output_error_sq_plain"] for r in rows],
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=1.6,
            markersize=3,
        )
        plt.plot(
            [r["ratio"] for r in rows],
            [r["frob_output_error_sq_actaware"] for r in rows],
            color=color,
            marker=marker,
            linestyle="--",
            linewidth=1.6,
            markersize=3,
        )

    plot_layer(rows_conv, layer_colors["Conv"], layer_markers["Conv"])
    plot_layer(rows_gconv, layer_colors["GConv"], layer_markers["GConv"])
    plot_layer(rows_linear, layer_colors["Linear"], layer_markers["Linear"])

    plt.xlabel("Rank ratio")
    plt.ylabel("$\|\Delta Y\|_F^2 / \|Y\|_F^2$")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0.2, 1.0)  # min ratio show = 0.2

    style_handles = [
        Line2D([0], [0], color="0.25", linestyle="-", linewidth=1.6, label="Standard"),
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle="--",
            linewidth=1.6,
            label="Activation-aware",
        ),
    ]

    color_handles = [
        Line2D(
            [0],
            [0],
            color=layer_colors["Conv"],
            linestyle="-",
            linewidth=1.6,
            label="Conv (g=1)",
        ),
        Line2D(
            [0],
            [0],
            color=layer_colors["GConv"],
            linestyle="-",
            linewidth=1.6,
            label=f"Conv (g={base_gconv.gconv.groups})",
        ),
        Line2D(
            [0],
            [0],
            color=layer_colors["Linear"],
            linestyle="-",
            linewidth=1.6,
            label="Linear",
        ),
    ]
    leg_style = plt.legend(
        handles=style_handles,
        title="Scheme",
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        alignment="right",
        markerfirst=False,
        handlelength=2.2,
        handletextpad=0.6,
    )

    leg_color = plt.legend(
        handles=color_handles,
        title="Layer type",
        loc="upper right",
        bbox_to_anchor=(1.0, 0.7),
        frameon=False,
        alignment="right",
        markerfirst=False,
        handlelength=2.2,
        handletextpad=0.6,
    )

    plt.gca().add_artist(leg_style)  # keep both legends

    plt.tight_layout()
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "rankratio_vs_output_frob.pdf")


if __name__ == "__main__":
    main()
