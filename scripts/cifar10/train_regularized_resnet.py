import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.models import load_model
from lib.utils import cifar10_mean, cifar10_std, AverageMeter, seed_everything
from lib.factorization.training import SingularValsRegularizer, BatchWhiteningShrinker


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--pretrained-path", type=str, default=None)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument(
        "--model", type=str, default="resnet20", choices=["resnet20", "resnet56"]
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--milestones", type=int, nargs="+", default=[20, 28])
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--reg-weight", type=float, default=0.003)
    p.add_argument("--shrink-energy-frac", type=float, default=0.005)
    p.add_argument("--shrink-every", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    valset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=transform_val
    )
    valloader = DataLoader(
        valset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = load_model(args.model, pretrained_path=args.pretrained_path).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )
    regularizer = SingularValsRegularizer(model)

    shrinker = BatchWhiteningShrinker(
        model,
        energy_frac=args.shrink_energy_frac,
        every=args.shrink_every,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "model.pth"
    epochs_path = out_dir / "epochs.json"
    config_path = out_dir / "config.json"

    # Save run configuration
    with config_path.open("w") as f:
        json.dump(vars(args), f, indent=2)

    epochs_log = []

    for epoch in range(args.epochs):
        model.train()
        train_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()

        for inputs, targets in tqdm(
            trainloader, desc=f"Epoch {epoch+1:03d} [Train]", leave=False
        ):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # shrinker.set_data(inputs)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            reg_loss = regularizer.loss()
            (loss + args.reg_weight * reg_loss).backward()
            optimizer.step()
            # shrinker.step()
            bs = targets.size(0)
            train_loss_meter.update(loss.item(), bs)
            reg_loss_meter.update(reg_loss.item(), bs)

        model.eval()
        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()
        with torch.no_grad():
            for inputs, targets in tqdm(
                valloader, desc=f"Epoch {epoch+1:03d} [Val]", leave=False
            ):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                correct = (outputs.argmax(1) == targets).sum().item()
                bs = targets.size(0)
                val_loss_meter.update(loss.item(), bs)
                val_acc_meter.update(correct / bs * 100.0, bs)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:03d} | LR {lr_now:.5f} | "
            f"Train Loss {train_loss_meter.avg:.4f} | Reg Loss {reg_loss_meter.avg:.4f} | "
            f"Val Loss {val_loss_meter.avg:.4f} | Val Acc {val_acc_meter.avg:.2f}%"
        )

        epochs_log.append(
            {
                "epoch": epoch + 1,
                "lr": lr_now,
                "train_loss": float(train_loss_meter.avg),
                "reg_loss": float(reg_loss_meter.avg),
                "val_loss": float(val_loss_meter.avg),
                "val_acc": float(val_acc_meter.avg),
            }
        )

        scheduler.step()
        torch.save(model.state_dict(), str(weights_path))

    with epochs_path.open("w") as f:
        json.dump(epochs_log, f, indent=2)


if __name__ == "__main__":
    main()
