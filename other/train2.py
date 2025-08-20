#!/usr/bin/env python3
"""
CIFAR‑10 fine‑tuning with an *adaptive* nuclear‑norm constraint
enforced by Stochastic Frank–Wolfe (rank‑1 LMO).
"""

import argparse, json, math, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from statistics import median

# ---- project helpers ----------------------------------------------------------
from models import load_model
from utils import seed_everything

# -------------------------------------------------------------------------------


@torch.no_grad()
def leading_singular_vectors(M: torch.Tensor, n_iter: int = 2):
    v = torch.randn(M.size(1), device=M.device)
    v /= v.norm() + 1e-12
    for _ in range(n_iter):
        u = torch.mv(M, v)
        u /= u.norm() + 1e-12
        v = torch.mv(M.t(), u)
        v /= v.norm() + 1e-12
    return u, v


class SFWNuclear(torch.optim.Optimizer):
    """
    Stochastic Frank‑Wolfe on a *per‑parameter* nuclear‑norm ball.
    Each param group holds its own `tau`.
    """

    def __init__(self, param_groups, lr=0.1, rescale_grad=True):
        # param_groups is a list of dicts already containing 'params' and 'tau'
        for g in param_groups:
            g.setdefault("lr", lr)
            g.setdefault("rescale_grad", rescale_grad)
            g.setdefault("step", 0)
        super().__init__(param_groups, {})

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for g in self.param_groups:
                tau, lr, rescale = g["tau"], g["lr"], g["rescale_grad"]
                for p in g["params"]:
                    if p.grad is None:
                        continue

                    if p.dim() < 2:  # bias / BN
                        p.data.add_(p.grad, alpha=-lr)
                        continue

                    grad = p.grad.view(p.grad.size(0), -1)
                    if rescale:
                        scale = (p.data.norm() + 1e-12) / (grad.norm() + 1e-12)
                        grad = grad * scale

                    u, v = leading_singular_vectors(grad)
                    S = (-tau) * torch.outer(u, v).view_as(p.data)

                    t = g["step"]
                    gamma = 2.0 / (t + 2.0)
                    p.data.mul_(1 - gamma).add_(S, alpha=gamma)

                g["step"] += 1
        return loss


# ----------------------------------------------------------------------
#                              T R A I N I N G
# ----------------------------------------------------------------------

cifar10_mean = (0.4914, 0.4822, 0.4464)
cifar10_std = (0.2470, 0.2435, 0.2616)


def parse_args():
    P = argparse.ArgumentParser("Adaptive nuclear‑norm Frank‑Wolfe on CIFAR‑10")
    P.add_argument("--batch_size", type=int, default=128)
    P.add_argument("--epochs", type=int, default=200)
    P.add_argument(
        "--tau_init", type=float, default=0.8, help="initial τ for every layer"
    )
    P.add_argument(
        "--tau_min", type=float, default=0.4, help="lower bound for adaptive τ"
    )
    P.add_argument(
        "--tau_max", type=float, default=2.0, help="upper bound for adaptive τ"
    )
    P.add_argument(
        "--lr", type=float, default=0.1, help="SGD lr for vector params (bias/BN)"
    )
    P.add_argument("--momentum", type=float, default=0.9)
    P.add_argument("--weight_decay", type=float, default=5e-4)
    P.add_argument("--model_name", type=str, default="resnet20")
    P.add_argument("--pretrained_path", type=str, default="resnet20.pth")
    P.add_argument("--save_path", type=str, default="cifar10_resnet20_anneal.pth")
    P.add_argument("--log_path", type=str, required=True)
    return P.parse_args()


def make_param_groups(model, tau_init, lr):
    """Return list of param groups with their own τ."""
    groups = []
    for p in model.parameters():
        if p.dim() >= 2:  # matrices / conv
            groups.append(dict(params=[p], tau=tau_init, lr=lr))
        else:  # vector‑like
            groups.append(dict(params=[p], tau=None, lr=lr))
    return groups


@torch.no_grad()
def layer_boundary_ratios(model, param_groups):
    """Return list of ρ = ‖W‖_* / τ for all constrained layers."""
    ratios = []
    for g in param_groups:
        tau = g["tau"]
        if tau is None:
            continue
        p = g["params"][0]
        mat = p.data.view(p.data.size(0), -1)
        # nuclear norm via svdvals (fast enough per epoch)
        sigma = torch.linalg.svdvals(mat)
        nuc = sigma.sum().item()
        ratios.append(nuc / tau)
    return ratios


def main():
    args = parse_args()
    seed_everything(0)

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    train_ds = datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
    val_ds = datasets.CIFAR10("data", train=False, download=True, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_name, 10, pretrained_path=args.pretrained_path)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    param_groups = make_param_groups(model, args.tau_init, args.lr)
    optimizer = SFWNuclear(param_groups)

    log = []
    for epoch in range(args.epochs):
        # -------------------------- TRAIN --------------------------
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)

            def closure():
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        # ------------------ ADAPT τ (at epoch level) ---------------
        ratios = layer_boundary_ratios(model, param_groups)
        if ratios:  # could be empty if model had no matrices
            r = median(ratios)
            if r > 0.95:  # too tight → loosen
                factor = 1.25
            elif r < 0.60:  # too loose → tighten
                factor = 0.90
            else:
                factor = 1.0
            if factor != 1.0:
                for g in param_groups:
                    if g["tau"] is None:
                        continue
                    new_tau = g["tau"] * factor
                    new_tau = max(args.tau_min, min(new_tau, args.tau_max))
                    g["tau"] = new_tau

        # -------------------------- VAL ----------------------------
        model.eval()
        val_loss = correct = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
        val_loss /= len(val_loader.dataset)
        acc = correct / len(val_loader.dataset)

        current_tau = median([g["tau"] for g in param_groups if g["tau"] is not None])
        print(
            f"Epoch {epoch+1:3d}: "
            f"TrainLoss {train_loss:.4f}  ValLoss {val_loss:.4f}  "
            f"Acc {acc:.4f}  τ~{current_tau:.3f}"
        )

        log.append(
            dict(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                accuracy=acc,
                tau=current_tau,
            )
        )
        torch.save(model, args.save_path)

    with open(args.log_path, "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
