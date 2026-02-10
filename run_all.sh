#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
  "$ROOT/houston.sh"
  "$ROOT/Urban.sh"
  "$ROOT/Trento.sh"
)

for script in "${scripts[@]}"; do
  if [ ! -f "$script" ]; then
    echo "[ERROR] Missing script: $script" >&2
    exit 1
  fi

  echo "[RUN ] $script"
  bash "$script"
  echo "[DONE] $script"
done

echo "All scripts completed."
