#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-serial}"
args=(-DCMAKE_BUILD_TYPE=Release)
if [[ "$(uname -s)" == Darwin ]]; then
  sdk="$(xcrun --show-sdk-path)"
  if [[ -f "$sdk/usr/include/c++/v1/array" ]]; then
    args+=("-DCMAKE_CXX_FLAGS=-isystem $sdk/usr/include/c++/v1")
  fi
fi
case "$mode" in
 serial) ;;
 mpi) args+=(-DDISPERSION_BUILD_PARALLEL=ON -DKokkos_ENABLE_OPENMP=OFF) ;;
 sanitize) args+=(-DDISPERSION_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug) ;;
 openmp) args=(-DCMAKE_BUILD_TYPE=Release -DDISPERSION_BUILD_PARALLEL=ON -DKokkos_ENABLE_OPENMP=ON "-DCMAKE_CXX_COMPILER=${DISPERSION_GXX:-g++-16}") ;;
 *) echo "Usage: $0 serial|mpi|sanitize|openmp" >&2; exit 1 ;;
esac
cmake -S . -B "build/$mode" "${args[@]}"
cmake --build "build/$mode" --parallel 4
ctest --test-dir "build/$mode" --output-on-failure
