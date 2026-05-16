# Elderly Fall Detection — ML Pipeline

Edge-AI fall detection model for smartphones. This repository contains the **ML pipeline** (preprocessing, training, evaluation, TFLite export) that produces the on-device classifier used by the **SafeMotion** Android app.

The exported `model.tflite` is the same model loaded by the app's `TFLiteRunner` and gated by the app's 4-stage runtime pipeline (impact → free-fall → ML → stillness).

## Overview

- **Task:** binary classification of 2-second IMU windows — *fall* vs. *ADL*
- **Input:** `(100, 6)` float32 — accelerometer x/y/z + gyroscope x/y/z at 50 Hz, standardized
- **Output:** sigmoid probability ∈ [0, 1]
- **Dataset:** [MobiAct](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/) — 67 subjects, 20 activity classes, ~3.3k recordings
- **Model:** 1D-CNN, ~140k params, 557 KB TFLite (float32) / 286 KB (float16)
- **Splits:** **subject-disjoint** 46 train / 10 val / 11 test, stratified by per-subject fall fraction

## Test Set Performance (honest, single seed)

Threshold + smoothing tuned **on val only**, then frozen and reported once on test.

| Metric | Value |
|---|---|
| ROC-AUC | **0.956** |
| PR-AUC | **0.557** |
| F1 | **0.625** |
| Precision | 0.695 |
| Recall | 0.567 |
| Test windows | 50,908 (2.7% falls) |
| Threshold | 0.45 |
| Smoothing | majority-of-5 (≥2 positives) |

> **Why these numbers are lower than the prior version.** A previous iteration of this repo reported F1 ≈ 0.86 / ROC-AUC ≈ 0.965. Those numbers were inflated by four leaks: (1) `StandardScaler` fit on val + test before the split, (2) random window-level re-split inside the trainer that put overlapping windows from the same subject in train and test, (3) decision threshold and smoothing tuned on the test set, (4) coarse folder-based labels that called every window in a fall recording a fall, including pre/post-fall ADL frames. After fixing all four, the numbers above are what the model can actually do on unseen subjects. See [MODEL_CARD.md](MODEL_CARD.md).

> **Why this is still good enough to ship.** The Android app uses the ML score as one of four gates (impact peak, free-fall dip, ML, post-impact stillness). The runtime pipeline absorbs the lions' share of the false positives the model lets through — particularly stair-climb (`SCH`) and chair-sit-in (`CSI`), which look like falls in a 2-s window but fail the post-impact stillness gate.

Per-subject recall ranges 0.33 – 0.80 across the 11 held-out subjects (real variance the prior leak hid). Hardest false-positive activities: `CHU` (chair-up, 11.1% FPR), `CSI` (car-step-in, 9.9%), `SCH` (stair-climb, 8.6%). Full breakdown in [evaluation_outputs/v2_clean/metrics.json](evaluation_outputs/v2_clean/metrics.json).

## Repository Layout

```
.
├── model.tflite                       Final exported model (deployed in SafeMotion)
├── MODEL_CARD.md                      Model card: data, metrics, limitations
├── Project/Annotated Data/            MobiAct annotated CSVs (raw input)
├── processed/
│   ├── windows_raw.npz                Unscaled windows + subject/activity/recording arrays
│   ├── dataset.npz                    Subject-disjoint, scaled splits + metadata
│   └── scaler.pkl                     Train-only StandardScaler
├── models/
│   └── v2_clean/                      best_model.keras, history.json, deploy_config.json,
│                                      model.tflite, model_float16.tflite, parity.json
├── evaluation_outputs/v2_clean/       metrics.json, curves.png, test_preds.npz
└── scripts/
    ├── preprocess.py                  Raw CSV -> windows_raw.npz (per-sample fall labels)
    ├── split_and_scale.py             Subject-disjoint split + train-only scaler
    ├── train.py                       Conv-only model w/ focal loss + sensor augmentation
    ├── evaluate.py                    Tune on val, report once on test (+ per-subject, per-activity)
    ├── export_tflite.py               Keras -> TFLite + parity check + deploy_config.json
    └── run_pipeline.sh                End-to-end driver
```

## Pipeline

```bash
./scripts/run_pipeline.sh "Project/Annotated Data"
```

That runs all five stages end-to-end. Manually:

```bash
python scripts/preprocess.py --data-root "Project/Annotated Data"   # 1) windows + per-sample labels
python scripts/split_and_scale.py                                    # 2) subject-disjoint split + train-only scaler
python scripts/train.py                                              # 3) train (focal loss, sensor aug, val PR-AUC early stop)
python scripts/evaluate.py                                           # 4) pick threshold on val, report once on test
python scripts/export_tflite.py                                      # 5) TFLite + parity + deploy_config.json
```

### Key methodology choices

- **Per-sample labels.** A window is labeled fall iff any sample inside it carries a MobiAct fall code (`FOL/FKL/BSC/SDL`). The original pipeline labeled every window in a fall recording as a fall, including the pre-fall standing and post-fall ADL frames — that's noisy supervision.
- **Subject-disjoint split.** Subjects are split *before* anything else; overlapping windows can no longer leak across splits because they share a subject. Splits are stratified by per-subject fall fraction so each split has comparable class balance.
- **Train-only scaler.** `StandardScaler` is fit on train only and applied to val/test.
- **Focal loss only.** No `class_weight` — the original pipeline stacked focal loss with class weights, which double-counts and (with `alpha=0.25`) actually pulls in opposite directions.
- **Sensor-domain augmentation.** Gaussian noise (std 0.02 on standardized signal) + small random Z-axis rotation (±8°) applied independently to the accel and gyro frames, simulating phone-pose drift.
- **Val-only threshold + smoothing tuning.** Threshold ∈ [0.05, 0.95] × smoothing ∈ {(1,1), (3,1), (3,2), (5,2), (5,3)} grid-searched on val; the (threshold, window, min_pos) tuple that maximizes val F1 is frozen and used once on test.
- **PR-AUC for early stopping.** ROC-AUC is misleading on a 2.7% positive class; we early-stop on `val_pr_auc`.
- **Keras↔TFLite parity check.** Max abs diff over 500 test samples logged to `models/v2_clean/parity.json`. Current build: 1.8e-7.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins TensorFlow 2.21 and the data-science stack. Tested on macOS Apple Silicon with Python 3.13.

The MobiAct annotated CSVs are expected at `Project/Annotated Data/<ACTIVITY>/<ACTIVITY>_<subject>_<trial>_annotated.csv` (the layout MobiAct ships). Pass a different root path as the first arg to `run_pipeline.sh` if needed.

## Future improvements

- Multi-seed runs / `StratifiedGroupKFold` for confidence intervals.
- Probability calibration (isotonic on val) so the deployed threshold is more interpretable.
- A small in-the-wild collection from elderly users to fine-tune.
- Per-recording (not per-window) evaluation: a missed window is fine if a neighboring one fires.

## Android integration

`run_pipeline.sh` copies `model.tflite` and `deploy_config.json` into `SafeMotion-main/app/src/main/assets/`. The app's `TFLiteRunner`:

1. Loads `model.tflite` and `deploy_config.json` from assets.
2. Standardizes every incoming sensor window with the train-time `scaler_mean` / `scaler_scale` before inference (must match training, otherwise probabilities are meaningless).
3. Exposes the val-tuned `threshold` so the foreground service uses whatever the model was tuned for, not a hardcoded value.

If you retrain, the new `model.tflite` and `deploy_config.json` are dropped in together — no source-code change needed in the app.

## Related

This repo is the model side of a two-part project:

- **This repo** — produces `model.tflite` and `deploy_config.json`.
- **SafeMotion Android app** ([SafeMotion-main/](SafeMotion-main/)) — wraps the model in a 4-gate runtime pipeline (impact peak → free-fall dip → ML → post-impact stillness) with SMS + location alerts.

## License

Research / educational use. The MobiAct dataset is subject to its own license.
