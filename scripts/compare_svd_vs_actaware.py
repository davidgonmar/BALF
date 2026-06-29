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
from plot_style import (
    COLORS,
    FIGSIZE_WIDE,
    apply_paper_style,
    line_handle,
    paper_line_kwargs,
    save_pdf,
    style_axes,
)

apply_paper_style()

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
    save_dir_tmp="./whitening-cache-tmp/",
):
    act_cache = collect_activation_cache(base_model, dl_calib, keys=[layer_key])
    norm_ref_sq = frobenius_norm_outputs_squared(base_model, dl_eval, device)

    rows = []
    min_ratio = 0.15
    max_n_steps = 20
    for i in range(max_n_steps + 1):
        ratio = min_ratio + (1.0 - min_ratio) * i / max_n_steps
        cfg = {layer_key: {"name": "params_ratio_to_keep", "value": ratio}}

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
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--tmp_dir", required=True)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
    )
    ds_calib = datasets.CIFAR10(args.data_root, train=True, download=True, transform=tfm)
    ds_eval = datasets.CIFAR10(args.data_root, train=False, download=True, transform=tfm)
    calib_g = torch.Generator().manual_seed(args.seed + 12345)
    eval_g = torch.Generator().manual_seed(args.seed + 54321)
    dl_calib = DataLoader(
        Subset(ds_calib, torch.randperm(len(ds_calib), generator=calib_g)[: args.n_samples_cache]),
        batch_size=args.batch_size,
    )
    dl_eval = DataLoader(
        Subset(ds_eval, torch.randperm(len(ds_eval), generator=eval_g)[: args.n_samples_eval]),
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

    tmp_root = Path(args.tmp_dir)
    rows_conv = sweep_layer(base_conv, "conv", dl_calib, dl_eval, device, tmp_root / "conv")
    rows_gconv = sweep_layer(base_gconv, "gconv", dl_calib, dl_eval, device, tmp_root / "gconv")
    rows_linear = sweep_layer(base_linear, "fc", dl_calib, dl_eval, device, tmp_root / "linear")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    layer_colors = {
        "Conv": COLORS["blue"],
        "GConv": COLORS["orange"],
        "Linear": COLORS["green"],
    }
    layer_markers = {
        "Conv": "o",
        "GConv": "s",
        "Linear": "^",
    }

    def plot_layer(rows, color, marker):
        ax.plot(
            [r["ratio"] for r in rows],
            [r["frob_output_error_sq_plain"] for r in rows],
            **paper_line_kwargs(color, marker=marker, linestyle="-"),
        )
        ax.plot(
            [r["ratio"] for r in rows],
            [r["frob_output_error_sq_actaware"] for r in rows],
            **paper_line_kwargs(color, marker=marker, linestyle="--"),
        )

    plot_layer(rows_conv, layer_colors["Conv"], layer_markers["Conv"])
    plot_layer(rows_gconv, layer_colors["GConv"], layer_markers["GConv"])
    plot_layer(rows_linear, layer_colors["Linear"], layer_markers["Linear"])

    ax.set_xlabel("Parameters ratio")
    ax.set_ylabel(r"$\|\Delta Y\|_F^2 / \|Y\|_F^2$")
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(bottom=-0.02)
    style_axes(ax)

    style_handles = [
        line_handle(COLORS["gray"], "Standard", linestyle="-"),
        line_handle(COLORS["gray"], "Act.-aware", linestyle="--"),
    ]

    color_handles = [
        line_handle(layer_colors["Conv"], "Conv (g=1)", marker=layer_markers["Conv"]),
        line_handle(
            layer_colors["GConv"],
            f"Conv (g={base_gconv.gconv.groups})",
            marker=layer_markers["GConv"],
        ),
        line_handle(layer_colors["Linear"], "Linear", marker=layer_markers["Linear"]),
    ]
    leg_style = ax.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=False,
        ncol=2,
        handlelength=2.2,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    ax.add_artist(leg_style)
    ax.legend(
        handles=color_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        frameon=False,
        ncol=3,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.45,
    )

    fig.subplots_adjust(bottom=0.32)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pdf(fig, out_dir / "paramsratio_vs_output_frob.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
