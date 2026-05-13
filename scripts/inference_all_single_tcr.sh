#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DEVICE="${DEVICE:-cuda:0}"

for cfg in configs/single_tcr/*_model.yml; do
    name="$(basename "$cfg" .yml)"
    echo "===== Inference: $name ====="
    python scripts/inference_test.py --config "$cfg" --device "$DEVICE"
done
