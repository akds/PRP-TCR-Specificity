#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DEVICE="${DEVICE:-cuda:0}"
PANEL="${PANEL:-SBWB}"

for cfg in configs/single_tcr/*_model.yml; do
    name="$(basename "$cfg" _model.yml)"
    echo "===== Proteome inference: $name (panel=$PANEL) ====="
    python scripts/inference_proteome.py \
        --config "$cfg" \
        --device "$DEVICE" \
        --tcr_id "$name" \
        --panel "$PANEL"
done
