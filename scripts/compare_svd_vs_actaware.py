import argparse
import json
from pathlib import Path
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from torchvision import datasets, transforms

from lib.factorization.factorize import (
    to_low_rank_activation_aware_manual,
    to_low_rank_manual,
    collect_activation_cache,
)


class SingleConv(nn.Module):
    """A tiny wrapper so factorizers can target layer key 'conv'."""

    def __init__(self, in_ch, out_ch, ksize, stride=1, padding=None, bias=False):
        super().__init__()
        if padding is None:
            padding = ksize // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, ksize, stride=stride, padding=padding, bias=bias
        )

    def forward(self, x):
        return self.conv(x)


def matrixize_weight_for_rank(w: torch.Tensor):
    # Conv2d: [out_c, in_c, kh, kw] -> [out_c, in_c*kh*kw]
    if w.dim() == 4:
        oc, ic, kh, kw = w.shape
        return w.view(oc, ic * kh * kw), oc, ic * kh * kw
    elif w.dim() == 2:
        return w, w.shape[0], w.shape[1]
    else:
        raise ValueError(f"Unsupported weight shape {w.shape}")


def max_feasible_rank(conv_module: nn.Conv2d) -> int:
    W = conv_module.weight.data
    _, o, i = matrixize_weight_for_rank(W)
    return min(o, i)


@torch.no_grad()
def output_frobenius_error(
    model_ref: nn.Module, model_approx: nn.Module, dl: DataLoader, device
) -> float:
    """
    Sum over batches of ||Y_ref - Y_approx||_F^2, then sqrt at the end.
    """
    model_ref.eval()
    model_approx.eval()

    total_sq = 0.0
    for xb, _ in dl:
        xb = xb.to(device)
        yr = model_ref(xb)
        ya = model_approx(xb)
        diff = (yr - ya).float()
        total_sq += torch.sum(diff * diff).item()

    return total_sq**0.5


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)

    # single conv hyperparams
    ap.add_argument("--in_ch", type=int, default=3)  # CIFAR-10 has 3 channels
    ap.add_argument("--out_ch", type=int, default=32)
    ap.add_argument("--ksize", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--bias", action="store_true")

    # input / cache generation
    ap.add_argument(
        "--img_h", type=int, default=32
    )  # kept for compatibility; CIFAR-10 is 32x32
    ap.add_argument("--img_w", type=int, default=32)
    ap.add_argument(
        "--n_samples_cache",
        type=int,
        default=512,
        help="CIFAR train images to build activation cache",
    )
    ap.add_argument(
        "--n_samples_eval",
        type=int,
        default=512,
        help="CIFAR test images to evaluate output error",
    )
    ap.add_argument("--batch_size", type=int, default=128)

    # sweep control
    ap.add_argument(
        "--max_rank_cap",
        type=int,
        default=None,
        help="optional cap on max rank for faster runs",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- build single randomly initialized conv ---
    if args.in_ch != 3:
        raise ValueError(f"--in_ch must be 3 to match CIFAR images; got {args.in_ch}")
    if args.img_h != 32 or args.img_w != 32:
        print(
            f"[warn] CIFAR-10 images are 32x32; ignoring img_h/img_w ({args.img_h}x{args.img_w})"
        )

    base = (
        SingleConv(
            in_ch=args.in_ch,
            out_ch=args.out_ch,
            ksize=args.ksize,
            stride=args.stride,
            bias=args.bias,
        )
        .to(device)
        .eval()
    )

    layer_key = "conv"
    full_rank = max_feasible_rank(base.conv)
    if args.max_rank_cap is not None:
        full_rank = min(full_rank, int(args.max_rank_cap))

    # <<< CHANGE 1: cap the sweep to rank 15 >>>
    sweep_max_rank = min(full_rank, 15)

    print(
        f"[info] Conv weight shape={tuple(base.conv.weight.shape)} | "
        f"feasible_max_rank={full_rank} | sweep_max_rank={sweep_max_rank}"
    )

    # --- CIFAR-10 data for activation cache (act-aware needs this) ---
    # Use train split for cache, test split for eval.
    tfm = transforms.ToTensor()
    ds_cache = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    ds_eval = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)

    n_cache = min(args.n_samples_cache, len(ds_cache))
    n_eval = min(args.n_samples_eval, len(ds_eval))

    dl_cache = DataLoader(
        Subset(ds_cache, range(n_cache)), batch_size=args.batch_size, shuffle=False
    )
    dl_eval = DataLoader(
        Subset(ds_eval, range(n_eval)), batch_size=args.batch_size, shuffle=False
    )

    act_cache = collect_activation_cache(base, dl_cache, keys=[layer_key])

    rows = []
    for r in range(
        1, sweep_max_rank + 1
    ):  # <<< CHANGE 2: iterate only up to sweep_max_rank >>>
        ratio = r / full_rank
        cfg = {layer_key: {"name": "rank_ratio_to_keep", "value": ratio}}

        # plain (no activation awareness)
        m_plain = to_low_rank_manual(base, cfg_dict=cfg, inplace=False)
        err_plain = output_frobenius_error(base, m_plain, dl_eval, device)

        # activation-aware
        m_act = to_low_rank_activation_aware_manual(
            base,
            act_cache,
            cfg_dict=cfg,
            inplace=False,
            save_dir="./whitening-cache-tmp/",
        )
        err_act = output_frobenius_error(base, m_act, dl_eval, device)

        rows.append(
            {
                "rank": r,
                "ratio": ratio,
                "frob_output_error_plain": err_plain,
                "frob_output_error_actaware": err_act,
            }
        )
        print(
            f"[rank {r:4d}/{sweep_max_rank}] ratio={ratio:.6f} | ||ΔY||_F plain={err_plain:.6f} | act-aware={err_act:.6f}"
        )

    # --- save results ---
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "rank_output_frob_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "ratio",
                "frob_output_error_plain",
                "frob_output_error_actaware",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)

    json_path = out_dir / "rank_output_frob_results.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    # --- plot (smaller, cleaner) ---
    import matplotlib.pyplot as plt

    ranks = [r["rank"] for r in rows]
    e_plain = [r["frob_output_error_plain"] for r in rows]
    e_act = [r["frob_output_error_actaware"] for r in rows]

    plt.figure(figsize=(5, 3))  # smaller footprint
    plt.plot(
        ranks, e_plain, label="Plain low-rank", linewidth=1.6, marker="o", markersize=4
    )
    plt.plot(
        ranks, e_act, label="Activation-aware", linewidth=1.6, marker="s", markersize=4
    )
    plt.xlabel("Rank")
    plt.ylabel(r"Output diff  $\|Y - \tilde{Y}\|_F$")
    plt.title(
        f"Single Conv {tuple(base.conv.weight.shape)} (up to rank {sweep_max_rank})",
        pad=6,
    )
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(1, sweep_max_rank)
    plt.tight_layout()

    # save both PNG + PDF
    plot_path_png = out_dir / "rank_vs_output_frob.png"
    plot_path_pdf = out_dir / "rank_vs_output_frob.pdf"
    plt.savefig(plot_path_png, bbox_inches="tight", dpi=200)
    plt.savefig(plot_path_pdf, bbox_inches="tight")

    print(
        f"\nSaved:\n - {csv_path}\n - {json_path}\n - {plot_path_png}\n - {plot_path_pdf}"
    )


if __name__ == "__main__":
    main()

    # delete whitening cache tmp
    import shutil

    shutil.rmtree("./whitening-cache-tmp/", ignore_errors=True)
