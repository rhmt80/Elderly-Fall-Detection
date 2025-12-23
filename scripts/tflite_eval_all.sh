#!/usr/bin/env bash
set -euo pipefail

OUTDIR="evaluation_outputs"
mkdir -p "$OUTDIR"
NPZ="processed/processed_data.npz"
BATCH=128

MODELS=(
  "models/tflite/model_float16.tflite"
  "models/tflite/model_dynamic_range.tflite"
  "models/tflite/model_int8.tflite"
)

for m in "${MODELS[@]}"; do
  if [ ! -f "$m" ]; then
    echo "Skipping missing model: $m"
    continue
  fi
  name=$(basename "$m" .tflite)
  out="$OUTDIR/${name}.log"
  echo "Running eval for $m -> $out"
  python scripts/tflite_eval.py --tflite "$m" --npz "$NPZ" --batch $BATCH 2>&1 | tee "$out"
  echo "--- finished $name ---"
done

echo
echo "Summary of outputs in $OUTDIR:"
ls -lh "$OUTDIR" || true
echo "You can inspect logs with: tail -n 200 evaluation_outputs/<model>.log"
