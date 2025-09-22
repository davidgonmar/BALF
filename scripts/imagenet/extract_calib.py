"""
Script used to extract a random uniform sample of k images from the
ImageNet tar files. We use this to extract a calibration subset from
the train set for activation-aware low-rank factorization.
"""

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

    # Pass 1: count total images per class
    counts = []
    with tarfile.open(train_tar_path, "r") as train_tar:
        class_members = [
            m for m in train_tar.getmembers() if m.isfile() and m.name.endswith(".tar")
        ]
        if not class_members:
            raise ValueError("no class archives found")

        for cm in tqdm(class_members, desc="Counting"):
            f = train_tar.extractfile(cm)
            if f is None:
                counts.append((cm, 0))
                continue
            n = 0
            with tarfile.open(fileobj=f, mode="r") as class_tar:
                for m in class_tar:
                    if m.isfile():
                        n += 1
            counts.append((cm, n))

    total = sum(n for _, n in counts)
    if total < k:
        raise ValueError(f"requested k={k} but only {total} available")

    # Pick random global indices
    chosen_indices = set(rng.sample(range(total), k))

    # Map global indices -> per-class local indices
    selected_by_class = {}
    offset = 0
    for cm, n in counts:
        local = []
        for i in range(n):
            if offset + i in chosen_indices:
                local.append(i)
        if local:
            selected_by_class[cm] = set(local)
        offset += n

    # Pass 2: extract selected files
    with tarfile.open(train_tar_path, "r") as train_tar:
        for cm, local_indices in tqdm(selected_by_class.items(), desc="Extracting"):
            f = train_tar.extractfile(cm)
            if f is None:
                continue
            cls_name = cm.name.replace(".tar", "")
            dst = out_dir / cls_name
            dst.mkdir(parents=True, exist_ok=True)

            with tarfile.open(fileobj=f, mode="r") as class_tar:
                for idx, m in enumerate(class_tar):
                    if not m.isfile():
                        continue
                    if idx in local_indices:
                        im = class_tar.extractfile(m)
                        if im:
                            out_path = dst / os.path.basename(m.name)
                            with open(out_path, "wb") as g:
                                g.write(im.read())

    print(f"Extracted {k} random images into {out_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Sample random images from ImageNet tar files."
    )
    ap.add_argument(
        "--train-tar", type=str, required=True, help="Path to ILSVRC2012_img_train.tar"
    )
    ap.add_argument(
        "--out", type=str, required=True, help="Destination folder for random sample"
    )
    ap.add_argument(
        "--k", type=int, default=16384, help="Total number of images to select"
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()
    sample_from_tar(args.train_tar, args.out, k=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
