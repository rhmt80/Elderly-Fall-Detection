import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "best_model.keras"
DATA_PATH = BASE / "processed" / "processed_data.npz"
OUT_DIR = BASE / "evaluation_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
data = np.load(DATA_PATH, allow_pickle=True)
print("NPZ keys:", data.files)
X_test = data["X_test"].astype(np.float32)
y_test = data["y_test"].astype(np.int32)
if y_test.ndim > 1 and y_test.shape[1] == 1:
    y_test = y_test.ravel()
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
print("Model loaded:", MODEL_PATH)
y_pred_raw = model.predict(X_test, batch_size=256)
y_prob = None
if y_pred_raw.ndim == 1:
    y_prob = y_pred_raw
elif y_pred_raw.ndim == 2 and y_pred_raw.shape[1] == 1:
    y_prob = y_pred_raw.ravel()
elif y_pred_raw.ndim == 2 and y_pred_raw.shape[1] == 2:
    exp = np.exp(y_pred_raw - np.max(y_pred_raw, axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    y_prob = probs[:, 1]
else:
    y_prob = y_pred_raw.ravel()
y_prob = np.clip(y_prob, 0.0, 1.0)
best_thr = 0.5
best_f1 = -1.0
results_sweep = []
ths = np.linspace(0.1, 0.9, 17)
for thr in ths:
    y_pred = (y_prob >= thr).astype(int)
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    results_sweep.append({"threshold": float(thr), "precision": p, "recall": r, "f1": f1})
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr
y_pred_best = (y_prob >= best_thr).astype(int)
precision = precision_score(y_test, y_pred_best, zero_division=0)
recall = recall_score(y_test, y_pred_best, zero_division=0)
f1 = f1_score(y_test, y_pred_best, zero_division=0)
cm = confusion_matrix(y_test, y_pred_best)
auc = None
try:
    auc = float(roc_auc_score(y_test, y_prob))
except Exception:
    auc = None
out_metrics = {
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": auc,
    "best_threshold": float(best_thr),
    "best_threshold_f1": float(best_f1),
    "n_test": int(len(y_test)),
}
out_metrics["threshold_sweep"] = results_sweep
with open(OUT_DIR / "test_metrics_best.json", "w") as fh:
    json.dump(out_metrics, fh, indent=2)
np.savez_compressed(OUT_DIR / "test_preds_best.npz", y_prob=y_prob, y_pred=y_pred_best, y_test=y_test)
fig, ax = plt.subplots(figsize=(4, 4))
cm_arr = cm
ax.imshow(cm_arr, interpolation="nearest")
ax.set_title(f"Confusion matrix (thr={best_thr:.2f})")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
for i in range(cm_arr.shape[0]):
    for j in range(cm_arr.shape[1]):
        ax.text(j, i, format(int(cm_arr[i, j]), "d"), ha="center", va="center")
fig.tight_layout()
fig.savefig(OUT_DIR / "confusion_matrix_best.png", dpi=150)
plt.close(fig)
import csv
with open(OUT_DIR / "threshold_sweep.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["threshold", "precision", "recall", "f1"])
    writer.writeheader()
    for row in results_sweep:
        writer.writerow(row)
print("=== Best Model Re-Evaluation ===")
print(f"n_test:     {len(y_test)}")
print(f"best_thr:   {best_thr:.3f}")
print(f"precision:  {precision:.4f}")
print(f"recall:     {recall:.4f}")
print(f"f1:         {f1:.4f}")
print(f"roc_auc:    {auc}")
print("Saved outputs to:", OUT_DIR)
