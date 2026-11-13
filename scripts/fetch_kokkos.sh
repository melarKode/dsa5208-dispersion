#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
revision=175257a51ff29a0059ec48bcd233ee096b2c0438
mkdir -p external
if [[ ! -d external/kokkos ]]; then
  git clone --depth 1 --branch 4.5.01 https://github.com/kokkos/kokkos.git external/kokkos
fi
actual="$(git -C external/kokkos rev-parse HEAD)"
[[ "$actual" == "$revision" ]] || { echo "Unexpected Kokkos revision: $actual" >&2; exit 1; }
[[ -z "$(git -C external/kokkos status --porcelain)" ]] || { echo "Kokkos has local changes" >&2; exit 1; }
