#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p output
archive="output/dispersion-source.tar.gz"
tar --exclude='__pycache__' --exclude='*.pyc' -czf "$archive" \
  CMakeLists.txt README.md requirements.txt .gitignore .clang-format include src tests configs scripts docs data/metadata
printf '%s\n' "$archive"
