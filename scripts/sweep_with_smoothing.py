# scripts/sweep_with_smoothing.py
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

data = np.load("/Users/rehmatsinghchawla/Desktop/Project/models/exp_bce/test_preds.npz")
y_test = data["y_test"]
y_prob = data["y_prob"]

thresholds = [i/100 for i in range(20,91,2)]
ks = [1, 2, 3]  # k=1 means no smoothing, k=2 => require 2-in-row, etc.

best = None
print("k, threshold, precision, recall, f1")
for k in ks:
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        if k > 1:
            smoothed = (np.convolve(preds, np.ones(k, dtype=int), mode='same') >= k).astype(int)
        else:
            smoothed = preds
        p = precision_score(y_test, smoothed, zero_division=0)
        r = recall_score(y_test, smoothed, zero_division=0)
        f1 = f1_score(y_test, smoothed, zero_division=0)
        print(f"{k}, {t:.2f}, {p:.4f}, {r:.4f}, {f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, k, t, p, r)

print("\nBest by F1 overall:", best)