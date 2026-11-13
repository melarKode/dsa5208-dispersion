#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
module load Python/3.12.3-GCCcore-13.3.0
if [[ ! -x .venv-vanda/bin/python ]]; then python -m venv .venv-vanda; fi
.venv-vanda/bin/python -m pip install -r requirements.txt
.venv-vanda/bin/python -c 'import numpy, h5py, matplotlib; print("Vanda Python environment ready")'
