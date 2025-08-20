#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  CIFAR‑10 fine‑tuning with batch‑wise whitening‑aware energy‑based shrinkage
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from factorization.regularizers import Regularizer  # external lib you already have
from models import load_model  # external lib you already have
from utils import seed_everything  # external util you already have


# ─────────────────────────────────────────────────────────────────────────────
#  Whitening utilities (self‑contained)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def obtain_whitening_matrix_eigh(acts: torch.Tensor, module: nn.Module):
    """
    Return Π and Π⁻¹ (whitening and its inverse) built from *acts* using eig‑decomp.
    """
    if isinstance(module, nn.Conv2d):
        assert acts.dim() == 4
        im2col = nn.functional.unfold(
            acts,
            kernel_size=module.kernel_size,
            padding=module.padding,
            stride=module.stride,
        )
        im2col = im2col.permute(0, 2, 1).reshape(-1, im2col.size(1))
    elif isinstance(module, nn.Linear):
        assert acts.dim() == 2
        im2col = acts
    else:
        raise ValueError("Module should be either Conv2d or Linear")

    cov = im2col.T @ im2col
    eigvals, eigvecs = torch.linalg.eigh(cov)
    svals = torch.sqrt(torch.clamp(eigvals, min=1e-12))
    keep = svals > 1e-6
    svals = svals[keep]
    eigvecs = eigvecs[:, keep]

    Π = eigvecs @ torch.diag(1.0 / svals)  # whitening
    Π_inv = torch.diag(svals) @ eigvecs.T  # inverse whitening
    return Π, Π_inv


def obtain_whitening_matrix(acts: torch.Tensor, module: nn.Module):
    return obtain_whitening_matrix_eigh(acts, module)


# ─────────────────────────────────────────────────────────────────────────────
#  Energy‐based singular‐value shrinkage via cumulative energy removal
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _energy_threshold(mat: torch.Tensor, energy_frac: float):
    """
    Remove the smallest singular values whose cumulative squared‐energy
    fraction ≤ energy_frac. I.e., set to zero all smallest S_i such that
        cumsum(sorted(S^2)) / sum(S^2) ≤ energy_frac
    """
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    S2 = S**2
    total = S2.sum()
    # sort squared singular values ascending
    perm = torch.argsort(S2)
    sorted_S2 = S2[perm]
    cumsum = torch.cumsum(sorted_S2, dim=0)
    # find mask of values to zero
    thresh_idx = (cumsum / total <= energy_frac).sum().item()
    # zero out the smallest 'thresh_idx' singular values
    S_new = S.clone()
    if thresh_idx > 0:
        zero_idxs = perm[:thresh_idx]
        S_new[zero_idxs] = 0.0
    return (U * S_new.unsqueeze(0)) @ Vh


# ─────────────────────────────────────────────────────────────────────────────
#  Shrinkage done *per mini‑batch*
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def shrink_model_batch(
    model: nn.Module, batch_acts: dict[nn.Module, torch.Tensor], *, energy_frac: float
):
    """
    For each Linear or Conv2d layer in `model`, compute the whitening
    from this batch's activations and zero out the smallest singular
    values whose cumulative squared‐energy fraction ≤ energy_frac.
    """
    for m, acts in batch_acts.items():
        if not isinstance(m, (nn.Linear, nn.Conv2d)):
            continue

        Π, Π_inv = obtain_whitening_matrix(acts, m)
        Π, Π_inv = Π.to(m.weight.device), Π_inv.to(m.weight.device)

        W = m.weight.data
        if isinstance(m, nn.Conv2d):
            C_o, C_i, H, Wk = W.shape
            W_flat = W.reshape(C_o, -1).T  # (in, out)
            W_hat = Π_inv @ W_flat
            W_thr = _energy_threshold(W_hat, energy_frac)
            W_new = (Π @ W_thr).T.reshape_as(W)
        else:  # Linear
            W_flat = W.data.T  # (in, out)
            W_hat = Π_inv @ W_flat
            W_thr = _energy_threshold(W_hat, energy_frac)
            W_new = (Π @ W_thr).T

        m.weight.data.copy_(W_new)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        "CIFAR10 + batch‑wise energy‑based shrinkage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # optimization
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--step_size", type=int, default=80)
    p.add_argument("--gamma", type=float, default=0.1)

    # model / reg
    p.add_argument("--model_name", type=str, default="resnet20")
    p.add_argument("--pretrained_path", type=str, default=None)
    p.add_argument("--reg_weight", type=float, default=0.005)

    # shrink ‑ normalized energy fraction
    p.add_argument(
        "--shrink_energy_frac",
        type=float,
        default=0.005,
        help="Fraction of squared‐singular‐value energy to remove by zeroing smallest singular values",
    )

    # bookkeeping
    p.add_argument(
        "--shrink_every", type=int, default=250, help="Shrink every N optimizer steps"
    )
    p.add_argument("--save_path", type=str, default="cifar10_resnet20_finetuned.pth")
    p.add_argument("--reg_state_path", type=str, default="regularizer_state.pth")
    p.add_argument("--log_path", type=str, required=True)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
cifar10_mean = (0.4914, 0.4822, 0.4464)
cifar10_std = (0.2470, 0.2435, 0.2616)


def main():
    args = parse_args()
    seed_everything(0)

    # data transforms
    tf_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    tf_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    # datasets & loaders
    ds_train = datasets.CIFAR10("data", train=True, download=True, transform=tf_train)
    ds_val = datasets.CIFAR10("data", train=False, download=True, transform=tf_val)
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # model, loss, optimizer, scheduler, regularizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        args.model_name, num_classes=10, pretrained_path=args.pretrained_path
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    regularizer = Regularizer(model)

    # hooks to capture pre‑activation inputs
    batch_acts = {}

    def _hook(m, inp, _out):
        x = inp[0] if isinstance(inp, tuple) else inp
        batch_acts[m] = x.detach()

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.register_forward_hook(_hook)

    # training loop
    log, step_idx = [], 0
    for epoch in range(args.epochs):
        model.train()
        tot_loss = tot_reg = 0.0

        for xb, yb in tqdm(dl_train, desc=f"Epoch {epoch+1}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)
            reg = regularizer.loss()
            (loss + args.reg_weight * reg).backward()
            optimizer.step()

            step_idx += 1
            if step_idx % args.shrink_every == 0:
                shrink_model_batch(
                    model, batch_acts, energy_frac=args.shrink_energy_frac
                )
                batch_acts.clear()

            tot_loss += loss.item() * xb.size(0)
            tot_reg += reg.item() * xb.size(0)

        tot_loss /= len(dl_train.dataset)
        tot_reg /= len(dl_train.dataset)
        scheduler.step()

        # validation
        model.eval()
        val_loss = correct = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()

        val_loss /= len(dl_val.dataset)
        acc = correct / len(dl_val.dataset)
        lr = optimizer.param_groups[0]["lr"]

        log.append(
            dict(
                epoch=epoch + 1,
                train_loss=tot_loss,
                reg_loss=tot_reg,
                val_loss=val_loss,
                accuracy=acc,
                learning_rate=lr,
            )
        )

        print(
            f"E{epoch+1:03d} | train={tot_loss:.4f} reg={tot_reg:.4f} "
            f"val={val_loss:.4f} acc={acc:.4f} lr={lr:.3e}"
        )

        torch.save(model.state_dict(), args.save_path)
        torch.save(regularizer.state_dict(), args.reg_state_path)

    with open(args.log_path, "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
