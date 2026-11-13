#!/usr/bin/env bash
module load GCC/13.3.0 OpenMPI/5.0.3-GCC-13.3.0 CMake/3.29.3-GCCcore-13.3.0
export LD_LIBRARY_PATH="/opt/pbs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=false
