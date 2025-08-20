#!/usr/bin/env python3
"""
imagenet_val_to_imagefolder_from_meta.py

Make an ImageFolder layout for the **ImageNet validation** set using ONLY:
  --val-tar:    ILSVRC2012_img_val.tar
  --devkit-tar: ILSVRC2012_devkit_t12.tar.gz

We read:
  - data/ILSVRC2012_validation_ground_truth.txt (50k labels, 1..1000)
  - data/meta.mat or data/meta_clsloc.mat (maps ILSVRC2012_ID -> WNID)

Output:
  <out>/val/<wnid>/*.JPEG

Requires: tqdm, scipy
  pip install tqdm scipy
"""

import argparse
import os
import tarfile
from pathlib import Path
from tqdm import tqdm
from scipy.io import loadmat
import io

IMAGE_EXTS = {".jpeg", ".jpg", ".JPEG", ".JPG"}


def _is_image(name: str) -> bool:
    return Path(name).suffix in IMAGE_EXTS


def _read_member_text(tf: tarfile.TarFile, rel_candidates):
    """Return decoded text from the first member whose name ends with any candidate path."""
    names = tf.getnames()
    for rel in rel_candidates:
        m = tf.getmember(rel) if rel in names else None
        if m is None:
            matches = [x for x in tf.getmembers() if x.name.endswith(rel)]
            m = matches[0] if matches else None
        if m:
            f = tf.extractfile(m)
            if f:
                return f.read().decode("utf-8", errors="ignore")
    return None


def _read_member_bytes(tf: tarfile.TarFile, rel_candidates):
    """Return raw bytes from the first member whose name ends with any candidate path."""
    names = tf.getnames()
    for rel in rel_candidates:
        m = tf.getmember(rel) if rel in names else None
        if m is None:
            matches = [x for x in tf.getmembers() if x.name.endswith(rel)]
            m = matches[0] if matches else None
        if m:
            f = tf.extractfile(m)
            if f:
                return f.read()
    return None


def _build_label_to_wnid_from_meta(devkit_tar_path: Path):
    """
    Returns dict: {label_id (1..1000): 'nXXXXXXXX'}
    Uses devkit's meta.mat or meta_clsloc.mat, which contains 'synsets' with fields:
      - 'ILSVRC2012_ID' (1..1000)
      - 'WNID' (e.g., 'n01440764')
    """
    with tarfile.open(devkit_tar_path, "r:*") as dk:
        meta_bytes = _read_member_bytes(
            dk,
            [
                "data/meta.mat",
                "ILSVRC2012_devkit_t12/data/meta.mat",
                "data/meta_clsloc.mat",
                "ILSVRC2012_devkit_t12/data/meta_clsloc.mat",
            ],
        )
        if meta_bytes is None:
            raise RuntimeError(
                "Could not find meta.mat/meta_clsloc.mat in the devkit tar."
            )

        # loadmat can read from a file-like object; wrap the bytes
        mat = loadmat(io.BytesIO(meta_bytes), squeeze_me=True, struct_as_record=False)
        if "synsets" not in mat:
            raise RuntimeError("Devkit MAT missing 'synsets' struct.")

        synsets = mat["synsets"]  # array-like of structs
        label_to_wnid = {}

        # Handle both 1D arrays and single objects
        syn_list = (
            synsets
            if isinstance(synsets, (list, tuple))
            else synsets.flatten() if hasattr(synsets, "flatten") else [synsets]
        )
        for s in syn_list:
            # Field access differs slightly depending on scipy/mat version
            try:
                ilsvrc_id = int(getattr(s, "ILSVRC2012_ID"))
            except Exception:
                ilsvrc_id = int(
                    s.ILSVRC2012_ID[()]
                    if hasattr(s, "ILSVRC2012_ID")
                    else s["ILSVRC2012_ID"]
                )
            try:
                wnid = str(getattr(s, "WNID"))
            except Exception:
                wnid = str(s.WNID[()] if hasattr(s, "WNID") else s["WNID"])
            # Some metas include many synsets; we only keep ones with 1..1000 IDs
            if 1 <= ilsvrc_id <= 1000:
                label_to_wnid[ilsvrc_id] = wnid

        if len(label_to_wnid) != 1000:
            # Some metas have more entries; ensure we have 1000 class IDs mapped
            missing = [i for i in range(1, 1001) if i not in label_to_wnid]
            raise RuntimeError(
                f"Could not map all 1000 class IDs from meta.mat. Missing: {missing[:5]}..."
            )

        return label_to_wnid


def _read_val_labels(devkit_tar_path: Path):
    """Returns list of 50k integers (1..1000), order corresponds to val images sorted by filename."""
    with tarfile.open(devkit_tar_path, "r:*") as dk:
        gt_txt = _read_member_text(
            dk,
            [
                "data/ILSVRC2012_validation_ground_truth.txt",
                "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
            ],
        )
        if gt_txt is None:
            raise RuntimeError(
                "Missing data/ILSVRC2012_validation_ground_truth.txt in the devkit."
            )
        return [int(x.strip()) for x in gt_txt.splitlines() if x.strip()]


def build_val_imagefolder(val_tar_path: Path, devkit_tar_path: Path, out_root: Path):
    out_val = out_root
    out_val.mkdir(parents=True, exist_ok=True)

    # Build mappings from devkit
    label_to_wnid = _build_label_to_wnid_from_meta(devkit_tar_path)
    val_labels = _read_val_labels(devkit_tar_path)

    with tarfile.open(val_tar_path, "r:*") as vt:
        members = [m for m in vt.getmembers() if m.isfile() and _is_image(m.name)]
        members.sort(key=lambda m: m.name)  # align with ground-truth order

        if len(members) != len(val_labels):
            raise RuntimeError(
                f"Validation count mismatch: {len(members)} images vs {len(val_labels)} labels. "
                "Use the official ILSVRC2012_img_val.tar + matching devkit."
            )

        for i, m in enumerate(tqdm(members, desc="Writing val ImageFolder")):
            label = val_labels[i]  # 1..1000
            wnid = label_to_wnid[label]  # 'nXXXXXXXX'
            class_dir = out_val / wnid
            class_dir.mkdir(parents=True, exist_ok=True)

            src = vt.extractfile(m)
            if src is None:
                continue
            out_path = class_dir / os.path.basename(m.name)
            with open(out_path, "wb") as g:
                g.write(src.read())


def main():
    ap = argparse.ArgumentParser(
        description="Build ImageFolder for ImageNet **validation** using devkit meta.mat."
    )
    ap.add_argument("--val-tar", required=True, help="Path to ILSVRC2012_img_val.tar")
    ap.add_argument(
        "--devkit-tar", required=True, help="Path to ILSVRC2012_devkit_t12.tar.gz"
    )
    ap.add_argument(
        "--out", required=True, help="Output root (creates <out>/val/<wnid>/...)"
    )
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    build_val_imagefolder(Path(args.val_tar), Path(args.devkit_tar), out_root)
    print(f"✅ Done: ImageFolder at {out_root}")


if __name__ == "__main__":
    main()
