#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -n "${PBS_JOBID:-}" ]] || { echo "Build in a PBS compute allocation" >&2; exit 1; }
source scripts/vanda_env.sh
mode="${1:-cpu}"
args=(-DCMAKE_BUILD_TYPE=Release -DDISPERSION_BUILD_PARALLEL=ON)
case "$mode" in
 cpu) args+=(-DCMAKE_CXX_COMPILER=mpicxx -DKokkos_ENABLE_OPENMP=ON) ;;
 gpu)
  module load cuda12.4/toolkit/12.4.1
  export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v g++)"
  args+=("-DCMAKE_CXX_COMPILER=$PWD/external/kokkos/bin/nvcc_wrapper" "-DMPI_CXX_COMPILER=$(command -v mpicxx)" -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE86=ON -DKokkos_ENABLE_OPENMP=OFF)
  ;;
 *) echo "Usage: $0 cpu|gpu" >&2; exit 1 ;;
esac
cmake -S . -B "build/vanda-$mode" "${args[@]}"
cmake --build "build/vanda-$mode" --parallel "${BUILD_JOBS:-4}"
ctest --test-dir "build/vanda-$mode" --output-on-failure
