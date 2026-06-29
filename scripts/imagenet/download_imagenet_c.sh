#!/usr/bin/env bash
set -euo pipefail

IMAGENETC_ROOT="${IMAGENETC_ROOT:?IMAGENETC_ROOT not set}"
SCRATCH_DIR="/tmp/imagenet-c-download"

mkdir -p "${IMAGENETC_ROOT}" "${SCRATCH_DIR}"

download_verify_extract() {
  local name="$1"
  local expected_md5="$2"
  local archive="${SCRATCH_DIR}/${name}.tar"
  local url="https://zenodo.org/records/2235448/files/${name}.tar?download=1"

  echo "=== Downloading ${name}.tar ==="
  wget -c --progress=dot:giga -O "${archive}" "${url}"
  echo "${expected_md5}  ${archive}" | md5sum -c -

  echo "=== Extracting ${name}.tar into ${IMAGENETC_ROOT} ==="
  tar -xf "${archive}" -C "${IMAGENETC_ROOT}"
  rm -f "${archive}"
}

download_verify_extract blur 2d8e81fdd8e07fef67b9334fa635e45c
download_verify_extract digital 89157860d7b10d5797849337ca2e5c03
download_verify_extract extra d492dfba5fc162d8ec2c3cd8ee672984
download_verify_extract noise e80562d7f6c3f8834afb1ecf27252745
download_verify_extract weather 33ffea4db4d93fe4a428c40a6ce0c25d

touch "${IMAGENETC_ROOT}/.complete"
echo "ImageNet-C download, verification, and extraction complete."
