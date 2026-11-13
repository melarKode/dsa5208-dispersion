#!/usr/bin/env bash
# Site initialization is not compatible with strict shell options on every node.
atlas_saved_flags=$-
set +eu
source /app1/ebenv
if ! type module >/dev/null 2>&1; then
  module() {
    local atlas_module_code
    atlas_module_code=$(/app1/common/Utils/gnu/modules-3.2.10/CentOS-6.3/Modules/3.2.10/bin/modulecmd bash "$@") || return
    eval "$atlas_module_code"
  }
fi
case "$atlas_saved_flags" in *e*) set -e ;; esac
case "$atlas_saved_flags" in *u*) set -u ;; esac
unset atlas_saved_flags
module load GCC/13.3.0 OpenMPI/5.0.3-GCC-13.3.0 CMake/3.29.3-GCCcore-13.3.0 || return 1
command -v mpicxx cmake >/dev/null || return 1
atlas_fabric_lib="${EBROOTLIBFABRIC:?libfabric module missing}/lib64"
[[ -f "$atlas_fabric_lib/libfabric.so.1" ]] || atlas_fabric_lib="$EBROOTLIBFABRIC/lib"
[[ -f "$atlas_fabric_lib/libfabric.so.1" ]] || return 1
export LD_LIBRARY_PATH="$atlas_fabric_lib:/opt/pbs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$atlas_fabric_lib/libfabric.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
unset atlas_fabric_lib
export OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=false MPLBACKEND=Agg
