# whitening_drift_all_layers.py
# --------------------------------------------------------------
# No part of YOUR original code is altered. This file only
#   • hooks every applicable layer,
#   • builds its *own* SVD whitener,
#   • reports Procrustes discrepancies batch-wise.
# --------------------------------------------------------------

import argparse
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from compress.experiments import load_vision_model, cifar10_mean, cifar10_std

# ───────────────────────────────────────────────────────────────
# small helpers that DO NOT touch your code base
# ───────────────────────────────────────────────────────────────


def compute_whitener(x: torch.Tensor) -> torch.Tensor:
    """Σ⁻¹ᐟ² V  for x∈ℝ^{n×d}.  Shape:  (r, d)"""
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    keep = S > 1e-6
    if not torch.any(keep):
        raise RuntimeError("all singular values ≈ 0")
    S, V = S[keep], Vh[keep]  # V : (r, d)
    return torch.diag(S.rsqrt()) @ V  # (r, d)


def avgdiff(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B must have the same number of columns")
    return (A - B).abs().mean().item() / (
        A.abs().mean().item() / 2 + B.abs().mean().item() / 2
    )


class CatchInputs:
    def __init__(self, module):
        self.buf = None
        self.hook = module.register_forward_pre_hook(self.s)

    def s(self, m, inp):
        self.buf = inp[0].detach()

    def clr(self):
        self.buf = None

    def close(self):
        self.hook.remove()


# ───────────────────────────────────────────────────────────────


def collect(model, loader, layers, batches, device):
    hooks = {m: CatchInputs(m) for m in layers}
    whitens = {m: [] for m in layers}

    with torch.no_grad():
        for b, (x, _) in enumerate(loader):
            if b == batches:
                break
            model(x.to(device))
            for m, catcher in hooks.items():
                a = catcher.buf
                if isinstance(m, nn.Conv2d):
                    a = (
                        F.unfold(
                            a,
                            kernel_size=m.kernel_size,
                            padding=m.padding,
                            stride=m.stride,
                        )
                        .permute(0, 2, 1)
                        .reshape(-1, a.size(1))
                    )
                else:  # Linear / LazyLinear
                    a = a.view(-1, a.shape[-1])
                whitens[m].append(compute_whitener(a.cpu()))
                catcher.clr()

    for c in hooks.values():
        c.close()
    return whitens


# ───────────────────────────────────────────────────────────────


def main(cfg):
    # ---------- your model / data go here ----------------------
    model = load_vision_model("resnet20", pretrained_path="resnet20.pth").to(cfg.device)
    from torchvision.datasets import CIFAR10

    from torchvision.transforms import ToTensor, Normalize, Compose

    loader = torch.utils.data.DataLoader(
        CIFAR10(
            root="data",
            train=True,
            transform=Compose(
                [ToTensor(), Normalize(mean=cifar10_mean, std=cifar10_std)]
            ),
            download=True,
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
    )
    # ---------- choose layers ---------------------------------
    layers = [
        m
        for n, m in model.named_modules()
        if isinstance(m, (nn.Linear, nn.LazyLinear, nn.Conv2d))
        and n.startswith(cfg.layer_prefix)
    ]

    if not layers:
        print("No matching layers found")
        return

    W = collect(model, loader, layers, cfg.batches, cfg.device)

    # ---------- pair-wise discrepancies ------------------------
    for m in layers:
        L = W[m]
        if len(L) < 2:
            continue
        print(f"\nLayer: {m._get_name()}  (id={id(m)})")
        dists = []
        for i, j in itertools.combinations(range(len(L)), 2):
            d = avgdiff(L[i], L[j])
            dists.append(d)
            print(f"  batch {i} vs {j}:  {d:.4e}")
        dists = torch.tensor(dists)
        print(f"  mean ± std  {dists.mean():.4e}  ±  {dists.std():.4e}")


# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--layer-prefix",
        default="",
        help="limit to modules whose name starts with this prefix",
    )
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args())
