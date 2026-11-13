#!/usr/bin/env bash
set -euo pipefail
if [[ "$(uname -s)" == Darwin ]]; then
  echo "Run this on the cluster login node; no large download on the Mac." >&2
  exit 1
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$root/data/raw"
cd "$root/data/raw"
name=interpolated_cases.h5
expected=f31c3ebad74affc026016c04069d14b7
if [[ -f "$name" ]] && printf '%s  %s\n' "$expected" "$name" | md5sum --check --status; then
  echo "Verified existing $name"; exit 0
fi
if [[ -f "$name" ]]; then echo "Existing $name failed checksum; move it aside before retrying." >&2; exit 1; fi
curl --fail --location --retry 5 --retry-delay 5 --connect-timeout 30 --continue-at - \
  'https://zenodo.org/api/records/21674773/files/interpolated_cases.h5/content' -o "$name.partial"
printf '%s  %s\n' "$expected" "$name.partial" | md5sum --check
mv "$name.partial" "$name"
