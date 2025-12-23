# scripts/threshold_sweep_conv_only.py
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

data = np.load("../models/exp_conv_only/test_preds.npz")
y_test = data["y_test"]
y_prob = data["y_prob"]

thresholds = [i/100 for i in range(20,91,2)]
print("threshold, precision, recall, f1")
best = None
for t in thresholds:
    preds = (y_prob >= t).astype(int)
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    print(f"{t:.2f}, {p:.4f}, {r:.4f}, {f1:.4f}")
    if best is None or f1 > best[0]:
        best = (f1, t, p, r)
print("\nBest by F1:", best)