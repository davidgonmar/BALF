#!/usr/bin/env python3
import argparse
import os
import random
import tarfile
from pathlib import Path
from tqdm import tqdm


def sample_from_tar(train_tar_path, out_dir, k=1024, seed=0):
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(train_tar_path, "r") as train_tar:
        classes = [
            m for m in train_tar.getmembers() if m.isfile() and m.name.endswith(".tar")
        ]
        if not classes:
            raise ValueError("no class archives found")
        n_classes = len(classes)
        if k < n_classes:
            raise ValueError(f"k={k} < number of classes ({n_classes})")
        rng.shuffle(classes)
        base = k // n_classes
        rem = k % n_classes
        extras_idx = set(rng.sample(range(n_classes), rem)) if rem > 0 else set()
        selected_names_by_class = {}
        total_available = 0
        for idx, cm in enumerate(tqdm(classes, desc="Selecting")):
            f = train_tar.extractfile(cm)
            if f is None:
                selected_names_by_class[cm] = set()
                continue
            target = base + (1 if idx in extras_idx else 0)
            picks = []
            seen = 0
            with tarfile.open(fileobj=f, mode="r") as class_tar:
                for m in class_tar:
                    if not m.isfile():
                        continue
                    total_available += 1
                    seen += 1
                    if len(picks) < target:
                        picks.append(m.name)
                    else:
                        j = rng.randrange(seen)
                        if j < target:
                            picks[j] = m.name
            selected_names_by_class[cm] = set(picks)
        total_selected = sum(len(v) for v in selected_names_by_class.values())
        if total_selected < k:
            raise ValueError(
                f"requested k={k} but only {total_selected} could be sampled from {total_available} images"
            )
        for cm, names in tqdm(selected_names_by_class.items(), desc="Extracting"):
            if not names:
                continue
            cls_name = cm.name.replace(".tar", "")
            dst = out_dir / cls_name
            dst.mkdir(parents=True, exist_ok=True)
            f = train_tar.extractfile(cm)
            if f is None:
                continue
            with tarfile.open(fileobj=f, mode="r") as class_tar:
                remaining = set(names)
                for m in class_tar:
                    if not m.isfile():
                        continue
                    if m.name in remaining:
                        im = class_tar.extractfile(m)
                        if im:
                            out_path = dst / os.path.basename(m.name)
                            with open(out_path, "wb") as g:
                                g.write(im.read())
                            remaining.remove(m.name)
                    if not remaining:
                        break
    print(f"✅ Extracted {k} images into {out_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Distill calibration set from ImageNet tar files."
    )
    ap.add_argument(
        "--train-tar", type=str, required=True, help="Path to ILSVRC2012_img_train.tar"
    )
    ap.add_argument(
        "--out", type=str, required=True, help="Destination folder for calibration set"
    )
    ap.add_argument(
        "--k",
        type=int,
        default=1024,
        help="Total number of images to select (balanced across classes)",
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()
    sample_from_tar(args.train_tar, args.out, k=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
