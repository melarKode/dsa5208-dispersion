#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/atlas_env.sh
module load Python/3.12.3-GCCcore-13.3.0
if [[ ! -x .venv-atlas/bin/python ]]; then python -m venv .venv-atlas; fi
.venv-atlas/bin/python -m pip install --only-binary=:all: \
  numpy==1.26.4 h5py==3.11.0 matplotlib==3.8.4
.venv-atlas/bin/python -m pip check
.venv-atlas/bin/python -c 'import numpy, h5py, matplotlib; print("Atlas Python environment ready")'
