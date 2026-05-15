# Elderly Fall Detection — ML Pipeline

Edge-AI fall detection model for smartphones. This repository contains the **machine learning pipeline** (data preprocessing, model training, evaluation, and TFLite conversion) that produces the on-device classifier used by the **SafeMotion** Android app.

The exported `model.tflite` is the same model loaded by the app's `TFLiteRunner` and gated by the app's 4-stage validation pipeline (impact → free-fall → ML → stillness).

## Overview

- **Task:** Binary classification of 2-second sensor windows — *fall* vs. *not fall*
- **Input:** 100 timesteps × 6 channels (accelerometer x/y/z + gyroscope x/y/z) sampled at 50 Hz
- **Output:** Single sigmoid probability ∈ [0, 1]
- **Dataset:** [MobiAct](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/) — annotated falls + activities of daily living (ADLs)
- **Deployed model:** Conv-only 1D-CNN (~95 KB TFLite, runs in real-time on-device)

## Test Set Performance

| Metric | Value |
|---|---|
| ROC-AUC | **0.965** |
| F1 (best threshold) | **0.859** |
| Precision | 0.836 |
| Recall | 0.883 |
| Best threshold | 0.35 |
| Test windows | 32,535 |

Full threshold sweep and confusion matrix in [`evaluation_outputs/`](evaluation_outputs/).

> The Android app uses a stricter on-device threshold (0.92) because the ML score is only one of four gates — the surrounding physics-based gates (free-fall dip, impact peak, post-impact stillness) absorb most of the false positives, so the on-device threshold is tuned for precision rather than recall.

## Repository Layout

```
.
├── model.tflite                     Final exported model (deployed in SafeMotion)
├── processed/
│   ├── processed_data.npz           Windowed + scaled MobiAct samples
│   └── scaler.pkl                   StandardScaler fit on training data
├── scripts/
│   ├── preprocess_mobiact.py        Raw CSV loader / class label mapping
│   ├── preprocess_mobiact_full.py   Window extraction + scaling + split
│   ├── resplit_processed.py         Subject-disjoint train/val/test split
│   ├── train_cnn_lstm.py            Trains CNN+LSTM and Conv-only models
│   ├── threshold_sweep.py           Sweeps decision threshold on val set
│   ├── threshold_sweep_conv_only.py Same, restricted to conv-only model
│   ├── sweep_majority_window.py     Post-hoc majority-vote smoothing
│   ├── sweep_with_smoothing.py      Threshold × smoothing joint sweep
│   ├── apply_threshold.py           Applies chosen threshold to test set
│   ├── evaluate_best.py             Final eval + confusion matrix
│   ├── convert_to_tflite.py         Keras → TFLite (float16 / dyn-range / int8)
│   ├── tflite_eval.py               Evaluates a .tflite file on test set
│   ├── tflite_eval_all.sh           Runs tflite_eval over all variants
│   └── save_deploy_config.py        Writes threshold + smoothing config JSON
├── evaluation_outputs/              Test metrics, confusion matrix, sweeps
└── requirements.txt
```

## Pipeline

### 1. Preprocess MobiAct

Splits each annotated CSV into 2-second windows (100 samples) with 50% overlap, labels each window by the activity class (`FALL_CLASSES` → 1, ADLs → 0), fits a `StandardScaler` on training data, and produces a subject-aware split so the same person never appears in both train and test.

```bash
python scripts/preprocess_mobiact_full.py
python scripts/resplit_processed.py        # subject-disjoint train/val/test
```

Outputs `processed/processed_data.npz` (`X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`) and `processed/scaler.pkl`.

### 2. Train

Two architectures are defined in [`scripts/train_cnn_lstm.py`](scripts/train_cnn_lstm.py):

- **`build_cnn_lstm`** — Conv1D × 3 → LSTM(64) → Dense. Higher accuracy but uses TensorList ops that require `Select TF Ops` to convert to TFLite (larger runtime, harder to ship).
- **`build_conv_only`** — Conv1D × 4 → GlobalAveragePooling → Dense. Slightly lower ceiling but converts cleanly to a stock TFLite interpreter — this is the **deployed** model.

Loss: focal loss (γ=2.0, α=0.25) to handle class imbalance, with class weights also applied.

```bash
python scripts/train_cnn_lstm.py --model conv_only --epochs 50
```

### 3. Threshold tuning

The model outputs raw sigmoid probabilities. The decision threshold is tuned on the validation set, optionally combined with majority-window smoothing across consecutive predictions to suppress isolated false positives.

```bash
python scripts/threshold_sweep_conv_only.py
python scripts/sweep_with_smoothing.py
```

### 4. Evaluate

```bash
python scripts/evaluate_best.py
```

Writes [`evaluation_outputs/test_metrics_best.json`](evaluation_outputs/test_metrics_best.json), `confusion_matrix_best.png`, and `test_preds_best.npz`.

### 5. Convert to TFLite

```bash
python scripts/convert_to_tflite.py
```

Three variants are produced (float16, dynamic-range, int8). Each is benchmarked with [`tflite_eval.py`](scripts/tflite_eval.py) — see [`evaluation_outputs/model_*.log`](evaluation_outputs/) for size / accuracy trade-offs. The selected variant is exported as `model.tflite` and dropped into the Android app's `app/src/main/assets/` folder.

### 6. Deploy config

```bash
python scripts/save_deploy_config.py
```

Writes a small JSON consumed by the Android app describing the chosen threshold and smoothing settings.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# TensorFlow / Keras are not pinned in requirements.txt — install a version
# compatible with your platform (the conversion script targets TF 2.x / Keras 3):
pip install "tensorflow>=2.15" "keras>=3.0"
```

Place the MobiAct annotated dataset at the path referenced by `DATA_ROOT` in [`scripts/preprocess_mobiact.py`](scripts/preprocess_mobiact.py) before running preprocessing.

## Related

This repo is the model side of a two-part project:

- **ML pipeline** (this repo) — trains the classifier and exports `model.tflite`.
- **SafeMotion Android app** — consumes `model.tflite` and wraps it in a runtime 4-gate detection pipeline (impact → free-fall → ML → stillness), with SMS + location alerts to a configured caretaker.

## Tech Stack

- Python 3, TensorFlow / Keras 3
- scikit-learn, NumPy, pandas, SciPy
- TensorFlow Lite (deployment target)
- MobiAct dataset

## License

Research / educational use. The MobiAct dataset is subject to its own license terms.
