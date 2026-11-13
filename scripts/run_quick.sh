#!/usr/bin/env bash
# Chạy thử nhanh (~1-2 phút): chỉ cold-start, 1 seed, trên ModCloth. Ghi vào 04-results/quick (không ghi đè kết quả chính).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 COLUMNS=120
mkdir -p results/quick
"$PY" -m sizerec experiment data/modcloth_final_data.json.gz --quick -o results/quick/modcloth 2>&1 \
  | tee results/quick/log_quick_modcloth.txt
