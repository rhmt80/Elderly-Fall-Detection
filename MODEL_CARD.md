# Model Card — SafeMotion fall detector

## Summary
- **Task:** binary classification of 2-second IMU windows — fall vs. ADL
- **Architecture:** 1D-CNN (~140k params, 5 conv blocks + GAP + dense)
- **Input:** shape `(100, 6)` float32, 50 Hz, channels `[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]`, standardized using `processed/scaler.pkl`
- **Output:** sigmoid probability ∈ [0, 1]
- **Training data:** MobiAct annotated CSVs, 67 subjects, 329,663 windows (50% overlap), fall ratio 2.7%
- **Splits:** subject-disjoint (46 train / 10 val / 11 test), stratified by per-subject fall fraction

## Test metrics (subject-disjoint, threshold + smoothing chosen on val only)

| | value |
|---|---|
| precision | 0.695 |
| recall | 0.567 |
| F1 | 0.625 |
| ROC-AUC | 0.956 |
| PR-AUC | 0.557 |
| n test windows | 50,908 |
| n test falls | 1,556 |
| selected threshold | 0.45 |
| smoothing | majority-of-5 (≥2 positives) |

A previous version of this repo reported F1 ≈ 0.86 / ROC-AUC ≈ 0.965. Those numbers were inflated by:
1. StandardScaler fit on all data including val/test before splitting.
2. Window-level random splits that put overlapping windows from the same subject in both train and test.
3. Threshold and smoothing tuned on the test set.
4. Folder-based labels (every window in a fall recording labeled fall, including pre/post-fall ADL frames).

The numbers above are after fixing all four.

## Per-subject recall on test
Recall varies from 0.33 to 0.80 across the 11 held-out subjects, which is meaningful real-world variance the prior splits hid.

## Hardest false positives (per-activity FPR on test)
- `CHU` chair-up (11.1%)
- `CSI` car-step-in (9.9%)
- `SCH` stair-climb (8.6%)

These activities contain transient downward acceleration that genuinely resembles a fall onset — fixing them is what the Android app's 4-gate runtime pipeline (impact peak + free-fall dip + ML + post-impact stillness) is designed to do, since stationarity afterward is the discriminator the windowed model can't see.

## Calibration
Model is uncalibrated; threshold was selected by F1 on val (0.45) and is loaded by the Android app from `assets/deploy_config.json` so model and runtime stay in sync. The ML score is one of four gates in the runtime pipeline — surrounding physics gates (impact peak, free-fall dip, post-impact stillness) absorb the FPs the per-window precision-recall trade lets through.

## Limitations
- Trained on MobiAct only — no in-the-wild data, no elderly-specific data.
- Phone-pose: a single Z-axis rotation augmentation was applied; arbitrary 3D pose changes are not covered.
- 50 Hz only; sample rates significantly different from training will degrade.
- Deploys without per-user calibration.

## Files
- `model.tflite` — float32 deployment artifact (Keras↔TFLite max abs diff 1.8e-7 over 500 samples)
- `models/v2_clean/best_model.keras` — Keras source
- `models/v2_clean/deploy_config.json` — threshold, smoothing, scaler stats
- `processed/scaler.pkl` — train-only StandardScaler
- `evaluation_outputs/v2_clean/metrics.json` — full metrics including per-subject and per-activity breakdowns

## Reproduction
```bash
./scripts/run_pipeline.sh "Project/Annotated Data"
```
Seed 42, single split. For confidence intervals consider multi-seed or `StratifiedGroupKFold`.
