import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from lib.factorization.factorize import (
    to_low_rank_activation_aware_manual,
    to_low_rank_manual,
    collect_activation_cache,
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
        return g * per_group_rank
    elif isinstance(mod, nn.Linear):
        W = mod.weight.data
        _, o, i = matrixize_weight_for_rank(W)
        return min(o, i)
    else:
        raise ValueError(f"Unsupported module type {type(mod)}")


@torch.no_grad()
def frobenius_norm_outputs(model: nn.Module, dl: DataLoader, device) -> float:
    model.eval()
    total_sq = 0.0
    for xb, _ in dl:
        xb = xb.to(device)
        yr = model(xb)
        total_sq += torch.sum(yr.float() ** 2).item()
    return total_sq**0.5


@torch.no_grad()
def output_frobenius_error(model_ref, model_approx, dl, device, norm_ref):
    model_ref.eval()
    model_approx.eval()
    total_sq = 0.0
    for xb, _ in dl:
        xb = xb.to(device)
        yr = model_ref(xb)
        ya = model_approx(xb)
        diff = (yr - ya).float()
        total_sq += torch.sum(diff * diff).item()
    return (total_sq**0.5) / (norm_ref + 1e-12)


def sweep_layer(
    base_model,
    layer_key,
    dl_cache,
    dl_eval,
    device,
    sweep_max_rank,
    save_dir_tmp="./whitening-cache-tmp/",
):
    act_cache = collect_activation_cache(base_model, dl_cache, keys=[layer_key])
    norm_ref = frobenius_norm_outputs(base_model, dl_eval, device)

    rows = []
    for r in range(1, sweep_max_rank + 1):
        ratio = r / sweep_max_rank
        cfg = {layer_key: {"name": "rank_ratio_to_keep", "value": ratio}}

        m_plain = to_low_rank_manual(base_model, cfg_dict=cfg, inplace=False)
        err_plain = output_frobenius_error(
            base_model, m_plain, dl_eval, device, norm_ref
        )

        m_act = to_low_rank_activation_aware_manual(
            base_model, act_cache, cfg_dict=cfg, inplace=False, save_dir=save_dir_tmp
        )
        err_act = output_frobenius_error(base_model, m_act, dl_eval, device, norm_ref)

        rows.append(
            {
                "rank": r,
                "ratio": ratio,
                "frob_output_error_plain": err_plain,
                "frob_output_error_actaware": err_act,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--in_ch", type=int, default=3)
    ap.add_argument(
        "--out_ch",
        type=int,
        default=33,
        help="for grouped conv, make this divisible by in_ch (e.g. 33 with in_ch=3)",
    )
    ap.add_argument("--ksize", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--bias", action="store_true")
    ap.add_argument(
        "--groups", type=int, default=3, help="use 3 for CIFAR depthwise-style conv"
    )
    ap.add_argument("--linear_out", type=int, default=128)
    ap.add_argument("--n_samples_cache", type=int, default=256)
    ap.add_argument("--n_samples_eval", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_rank_cap", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.ToTensor()
    ds_cache = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
    ds_eval = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
    dl_cache = DataLoader(
        Subset(ds_cache, range(args.n_samples_cache)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    dl_eval = DataLoader(
        Subset(ds_eval, range(args.n_samples_eval)),
        batch_size=args.batch_size,
        shuffle=False,
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
    if args.max_rank_cap:
        cap = int(args.max_rank_cap)
        fr_conv, fr_gconv, fr_linear = (
            min(fr_conv, cap),
            min(fr_gconv, cap),
            min(fr_linear, cap),
        )
    sweep_max_rank = max(1, min(fr_conv, fr_gconv, fr_linear))

    rows_conv = sweep_layer(
        base_conv, "conv", dl_cache, dl_eval, device, sweep_max_rank
    )
    rows_gconv = sweep_layer(
        base_gconv, "gconv", dl_cache, dl_eval, device, sweep_max_rank
    )
    rows_linear = sweep_layer(
        base_linear, "fc", dl_cache, dl_eval, device, sweep_max_rank
    )

    # --- Plot with single color per setting ---
    import matplotlib.pyplot as plt

    ranks = [r["rank"] for r in rows_conv]
    plt.figure(figsize=(7.5, 4.5))

    def plot_rows(rows, label, color, marker):
        plt.plot(
            ranks,
            [r["frob_output_error_plain"] for r in rows],
            label=f"{label} Plain",
            color=color,
            marker=marker,
            linewidth=1.6,
            markersize=3,
        )
        plt.plot(
            ranks,
            [r["frob_output_error_actaware"] for r in rows],
            label=f"{label} Act-aware",
            color=color,
            marker=marker,
            linestyle="--",
            linewidth=1.6,
            markersize=3,
        )

    plot_rows(rows_conv, "Conv", "C0", "o")
    plot_rows(rows_gconv, f"GConv (g={base_gconv.gconv.groups})", "C1", "s")
    plot_rows(rows_linear, "Linear", "C2", "^")

    plt.xlabel("Rank")
    plt.ylabel("Normalized output error ||ΔY|| / ||Y||")
    plt.title("Rank vs Normalized Frobenius Error")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(1, sweep_max_rank)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "rank_vs_output_frob.png", dpi=200)
    plt.savefig(out_dir / "rank_vs_output_frob.pdf")


if __name__ == "__main__":
    main()
