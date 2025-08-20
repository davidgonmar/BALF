#!/usr/bin/env python3
"""
Finetune a previously compressed CIFAR‑10 model *without* any regularisation,
***now with learning‑rate warm‑up support***.

Checkpoint discovery walks all sub‑folders of --models_dir, so the layout
created by run_low_rank_family.py works out of the box:

    compressed_models/reg0.004_params/resnet20_reg0.004_ratio0.30_params.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from utils import seed_everything

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ------------------------------------------------------------------ helpers
def find_checkpoint(
    models_dir: Path,
    model_name: str,
    reg_strength: float,
    ratio: float,
    metric: str,
) -> Path:
    """Recursively search *models_dir* for the expected checkpoint name."""
    pattern = f"reg{reg_strength:.3f}_{metric}/{model_name}_reg{reg_strength:.3f}_ratio{ratio:.2f}_{metric}.pt"
    matches = list(models_dir.rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{pattern}' inside '{models_dir}'. "
            "Make sure you've run run_low_rank_family.py first."
        )
    if len(matches) > 1:
        print(f"[warning] multiple matches found, using the first: {matches[0]}")
    return matches[0]


def evaluate(model, dl, device, criterion):
    """Return (loss, accuracy) on *dl* without gradient tracking."""
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
    loss /= len(dl.dataset)
    acc = correct / len(dl.dataset)
    return loss, acc


# ------------------------------------------------------------------ CLI
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--ratio", type=float, required=True, help="Compression ratio, e.g. 0.30"
    )
    p.add_argument("--metric", choices=["params", "flops"], required=True)
    p.add_argument("--reg_strength", type=float, required=True)
    p.add_argument(
        "--models_dir",
        default="compressed_models",
        help="Root folder that holds the per‑reg sub‑directories.",
    )
    p.add_argument("--model_name", default="resnet20")
    # optimisation hyper‑params
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--step_size", type=int, default=50)
    p.add_argument("--gamma", type=float, default=0.1)
    # warm‑up specific
    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=1,
        help="Number of warm‑up epochs before the main scheduler",
    )
    p.add_argument(
        "--warmup_start_factor",
        type=float,
        default=0.1,
        help="Initial LR multiplier at the beginning of warm‑up (start LR = warmup_start_factor * lr)",
    )
    # output
    p.add_argument("--save_path", required=True)
    p.add_argument("--log_path", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ------------------------------------------------------------------ main
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    # -------- locate compressed checkpoint -----------------------------------
    models_root = Path(args.models_dir)
    ckpt_path = find_checkpoint(
        models_root,
        args.model_name,
        args.reg_strength,
        args.ratio,
        args.metric,
    )
    print(f"✓ Found compressed model: {ckpt_path}")

    # -------- load model ------------------------------------------------------
    pkg = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = pkg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # -------- data ------------------------------------------------------------
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train_ds = datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_tf
    )
    val_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=val_tf)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # -------- optimiser -------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # -------- scheduler with warm‑up ------------------------------------------
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimiser,
        start_factor=args.warmup_start_factor,
        total_iters=args.warmup_epochs,
    )
    main_scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser,
        step_size=args.step_size,
        gamma=args.gamma,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimiser,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[args.warmup_epochs],
    )

    # -------- initial evaluation (epoch 0) ------------------------------------
    init_val_loss, init_acc = evaluate(model, val_dl, device, criterion)
    log = [
        {
            "epoch": 0,
            "train_loss": None,
            "val_loss": init_val_loss,
            "accuracy": init_acc,
            "learning_rate": optimiser.param_groups[0]["lr"],
        }
    ]
    print(f"Epoch 0 | val {init_val_loss:.4f} | acc {init_acc:.4f}")
    print(model)
    # -------- training loop ---------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_dl, desc=f"[{epoch}/{args.epochs}]"):
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_dl.dataset)
        scheduler.step()
        lr_now = optimiser.param_groups[0]["lr"]

        # --- validation
        val_loss, acc = evaluate(model, val_dl, device, criterion)

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": acc,
            "learning_rate": lr_now,
        }
        log.append(entry)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"acc {acc:.4f} | lr {lr_now:.6f}"
        )

        torch.save(model, args.save_path)

    with open(args.log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"✓ Finetuning complete — model saved to {args.save_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
