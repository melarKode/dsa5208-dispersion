#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${PBS_JOBID:-}" ]]; then
  echo "Run this script inside a PBS compute allocation." >&2
  exit 1
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
module load GCC/13.3.0 OpenMPI/5.0.3-GCC-13.3.0 CMake/3.29.3-GCCcore-13.3.0
# Make the PBS launcher plugin available before starting mpirun.
export LD_LIBRARY_PATH="/opt/pbs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p logs
exec > >(tee "logs/smoke-${PBS_JOBID}.log") 2>&1
date -Is
hostname
module list
if command -v git >/dev/null 2>&1; then
  git -C external/kokkos rev-parse HEAD
else
  echo "Git unavailable; dependency revision not recorded (requested tag: 4.5.01)."
fi
cmake -S . -B build/vanda-cpu -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx \
  -DDISPERSION_BUILD_PARALLEL=ON -DKokkos_ENABLE_OPENMP=ON
cmake --build build/vanda-cpu --parallel 4
export OMP_NUM_THREADS=2
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=false
mpirun -np 1 --map-by slot:PE=2 --bind-to core --report-bindings \
  -x LD_LIBRARY_PATH -x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND -x OMP_DYNAMIC \
  build/vanda-cpu/mpi_kokkos_smoke
mpirun -np 2 --map-by slot:PE=2 --bind-to core --report-bindings \
  -x LD_LIBRARY_PATH -x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND -x OMP_DYNAMIC \
  build/vanda-cpu/mpi_kokkos_smoke
