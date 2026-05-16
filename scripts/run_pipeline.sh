#!/usr/bin/env bash
# End-to-end pipeline: preprocess -> split -> train -> evaluate -> export TFLite.
# Usage: ./scripts/run_pipeline.sh [DATA_ROOT]
# DATA_ROOT defaults to ./Project/Annotated\ Data
set -euo pipefail

DATA_ROOT="${1:-Project/Annotated Data}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: dataset not found at: $DATA_ROOT" >&2
  echo "Pass the path to the MobiAct 'Annotated Data' folder as arg 1." >&2
  exit 1
fi

# shellcheck source=/dev/null
[ -f .venv/bin/activate ] && source .venv/bin/activate

echo "==> 1/5 preprocess"
python scripts/preprocess.py --data-root "$DATA_ROOT"

echo "==> 2/5 subject-disjoint split + train-only scaler"
python scripts/split_and_scale.py

echo "==> 3/5 train"
python scripts/train.py

echo "==> 4/5 evaluate (threshold tuned on val, reported on test)"
python scripts/evaluate.py

echo "==> 5/5 export TFLite + deploy config"
python scripts/export_tflite.py

echo "==> copying artifacts into SafeMotion app assets (if present)"
APP_ASSETS="SafeMotion-main/app/src/main/assets"
if [ -d "$APP_ASSETS" ]; then
  cp models/v2_clean/model.tflite        "$APP_ASSETS/model.tflite"
  cp models/v2_clean/deploy_config.json  "$APP_ASSETS/deploy_config.json"
  echo "    -> $APP_ASSETS/model.tflite"
  echo "    -> $APP_ASSETS/deploy_config.json"
else
  echo "    (skipped — $APP_ASSETS not present)"
fi

echo "Done."
