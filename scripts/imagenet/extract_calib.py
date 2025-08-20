#!/usr/bin/env python3
"""
Make a calibration set directly from ImageNet tar files
- Works with ILSVRC2012_img_train.tar (class tars inside)
- Picks 1 per class, then fills randomly until k
"""

import argparse
import os
import random
import tarfile
from pathlib import Path
from tqdm import tqdm


def sample_from_tar(train_tar_path, out_dir, k=1024, seed=0):
    random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = []

    # 1. Open top-level train tar
    with tarfile.open(train_tar_path, "r") as train_tar:
        class_members = [
            m for m in train_tar.getmembers() if m.isfile() and m.name.endswith(".tar")
        ]
        n_classes = len(class_members)

        if k < n_classes:
            raise ValueError(f"k={k} is smaller than number of classes ({n_classes})")

        # 2. One per class
        for cm in tqdm(class_members, desc="Choosing 1 per class"):
            class_file = train_tar.extractfile(cm)
            with tarfile.open(fileobj=class_file, mode="r") as class_tar:
                imgs = [m for m in class_tar.getmembers() if m.isfile()]
                choice = random.choice(imgs)
                selected.append((cm.name.replace(".tar", ""), choice, cm))

        # 3. Collect all candidates for random fill
        all_candidates = []
        for cm in tqdm(class_members, desc="Collecting all candidates"):
            class_file = train_tar.extractfile(cm)
            with tarfile.open(fileobj=class_file, mode="r") as class_tar:
                imgs = [m for m in class_tar.getmembers() if m.isfile()]
                for im in imgs:
                    all_candidates.append((cm.name.replace(".tar", ""), im, cm))

        # 4. Fill the rest
        remaining_needed = k - len(selected)
        extras = random.sample(all_candidates, remaining_needed)
        selected.extend(extras)

        # 5. Extract only selected files
        for class_name, im_member, cm in tqdm(selected, desc="Extracting selected"):
            dst_class = out_dir / class_name
            dst_class.mkdir(parents=True, exist_ok=True)

            class_file = train_tar.extractfile(cm)
            with tarfile.open(fileobj=class_file, mode="r") as class_tar:
                im_data = class_tar.extractfile(im_member)
                if im_data:
                    out_path = dst_class / os.path.basename(im_member.name)
                    with open(out_path, "wb") as f:
                        f.write(im_data.read())

    print(f"✅ Extracted {len(selected)} images into {out_dir}")


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
        help="Number of images to select (>= number of classes)",
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()

    sample_from_tar(args.train_tar, args.out, k=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
