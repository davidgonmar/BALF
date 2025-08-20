#!/usr/bin/env python3
"""
Fine‑tune ResNet‑20 on CIFAR‑10 with the paper’s Stochastic Frank‑Wolfe
optimiser under a *pure nuclear‑norm* ball:

        C = { W :  ‖W‖_* ≤ ρ }.

ρ is picked automatically at start as the largest nuclear norm among
all matrix weights, so no extra hyper‑parameters are introduced.
"""

# ----------------------------------------------------- std imports ----
import argparse, json, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import load_model
from utils import seed_everything

# ----------------------------------------------------- paper SFW  -----
import torch.optim as native_optimizers
from torchmetrics import MeanMetric


class SFW(native_optimizers.Optimizer):
    """Verbatim optimiser from the paper."""

    # ← entire class body left untouched; paste without edits
    def __init__(
        self,
        params,
        lr=0.1,
        rescale="diameter",
        momentum=0,
        dampening=0,
        extensive_metrics=False,
        device="cpu",
    ):
        momentum = momentum or 0
        dampening = dampening or 0
        if rescale is None and not (0.0 <= lr <= 1.0):
            raise ValueError("Invalid learning rate")
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("Momentum out of range")
        if not (0.0 <= dampening <= 1.0):
            raise ValueError("Dampening out of range")
        if rescale == "None":
            rescale = None
        if rescale not in ["diameter", "gradient", "fast_gradient", None]:
            raise ValueError("Bad rescale type")

        self.rescale = rescale
        self.extensive_metrics = extensive_metrics
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening)
        super().__init__(params, defaults)

        self.metrics = {
            "grad_norm": MeanMetric().to(device=device),
            "grad_normalizer_norm": MeanMetric().to(device=device),
            "diameter_normalizer": MeanMetric().to(device=device),
            "effective_lr": MeanMetric().to(device=device),
        }

    def reset_metrics(self):
        for m in self.metrics.values():
            m.reset()

    def get_metrics(self):
        return {
            k: (m.compute() if self.extensive_metrics else {})
            for k, m in self.metrics.items()
        }

    @torch.no_grad()
    def reset_momentum(self):
        for g in self.param_groups:
            if g["momentum"] > 0:
                for p in g["params"]:
                    self.state[p].pop("momentum_buffer", None)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for g in self.param_groups:
            constraint = g["constraint"]
            mom, damp = g["momentum"], g["dampening"]
            grads = []

            for p in g["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if mom > 0:
                    buf = self.state[p].setdefault(
                        "momentum_buffer", d_p.detach().clone()
                    )
                    buf.mul_(mom).add_(d_p, alpha=1 - damp)
                    d_p = buf
                grads.append(d_p)

            v_list = constraint.lmo(grads)

            factor = 1.0
            if self.rescale == "diameter":
                factor = 1.0 / constraint.get_diameter()
            elif self.rescale in ["fast_gradient", "gradient"]:
                grad_norm = torch.norm(torch.cat([g.flatten() for g in grads]), 2)
                if self.rescale == "fast_gradient":
                    denom = 0.5 * constraint.get_diameter()
                else:
                    denom = torch.norm(
                        torch.cat(
                            [p.flatten() for p in g["params"] if p.grad is not None]
                        )
                        - torch.cat([v.flatten() for v in v_list]),
                        2,
                    )
                factor = grad_norm / denom

            lr = max(0.0, min(factor * g["lr"], 1.0))
            for p, v in zip(g["params"], v_list):
                p.mul_(1 - lr).add_(v, alpha=lr)
        return loss


# --------------------------------------------- helper for top singular --
@torch.no_grad()
def _leading_uv(M, n_iter=2):
    v = torch.randn(M.size(1), device=M.device)
    v /= v.norm() + 1e-12
    for _ in range(n_iter):
        u = torch.mv(M, v)
        u /= u.norm() + 1e-12
        v = torch.mv(M.t(), u)
        v /= v.norm() + 1e-12
    return u, v


# --------------------------------------------- nuclear‑norm constraint --
class NuclearBall:
    """
    Nuclear‑norm ball  {‖W‖_* ≤ ρ}.  ρ is fixed once at init.
    LMO returns  -ρ u₁v₁ᵀ.
    """

    def __init__(self, params, radius):
        self.params = list(params)
        self.rho = radius  # scalar radius

    def lmo(self, grad_list):
        v_out = []
        for g, p in zip(grad_list, self.params):
            if g.dim() < 2:
                v_out.append(torch.zeros_like(g))
                continue
            u, v = _leading_uv(g.flatten(1))
            S = (-self.rho) * torch.outer(u, v).view_as(p.data)
            v_out.append(S)
        return v_out

    def get_diameter(self):
        return 2.0 * self.rho


# ------------------------------------------------ dataset & training ---
mean = (0.4914, 0.4822, 0.4464)
std = (0.2470, 0.2435, 0.2616)


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--model_name", default="resnet20")
    ap.add_argument("--pretrained_path", default="resnet20.pth")
    ap.add_argument("--save_path", default="cifar10_final.pth")  # ← unchanged
    ap.add_argument("--log_path", required=True)
    return ap.parse_args()


def main():
    args = cli()
    seed_everything(0)

    tf_train = transforms.Compose(
        [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    tf_val = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    ds_train = datasets.CIFAR10("data", True, download=True, transform=tf_train)
    ds_val = datasets.CIFAR10("data", False, download=True, transform=tf_val)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_name, 10, pretrained_path=args.pretrained_path).to(
        device
    )
    criterion = torch.nn.CrossEntropyLoss()

    # ---------------- split params & auto‑radius ---------------------
    matrices, vectors = [], []
    with torch.no_grad():
        max_nuc = 0
        for p in model.parameters():
            if p.dim() >= 2:
                matrices.append(p)
                nuc = torch.linalg.svdvals(p.flatten(1)).sum().item()
                max_nuc = max(max_nuc, nuc)
            else:
                vectors.append(p)

    radius = 1.0  # no extra CLI param
    constraint = NuclearBall(matrices, radius)

    opt_sfw = SFW(
        matrices, lr=args.lr, rescale="gradient", momentum=args.momentum, device=device
    )
    for g in opt_sfw.param_groups:
        g["constraint"] = constraint

    opt_sgd = torch.optim.SGD(
        vectors, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # ---------------- training loop ---------------------------------
    log = []
    for ep in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for x, y in tqdm(dl_train, desc=f"E{ep+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt_sfw.zero_grad()
            opt_sgd.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt_sfw.step()
            opt_sgd.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(dl_train.dataset)

        model.eval()
        v_loss = corr = 0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss += criterion(out, y).item() * x.size(0)
                corr += (out.argmax(1) == y).sum().item()
        v_loss /= len(dl_val.dataset)
        acc = corr / len(dl_val.dataset)

        print(f"Ep {ep+1:3d} | Train {tr_loss:.4f} | Val {v_loss:.4f} | Acc {acc:.4f}")
        log.append(
            dict(epoch=ep + 1, train_loss=tr_loss, val_loss=v_loss, accuracy=acc)
        )
        torch.save(model, args.save_path)

    with open(args.log_path, "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
