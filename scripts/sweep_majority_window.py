# scripts/sweep_majority_window.py
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

data = np.load("/Users/rehmatsinghchawla/Desktop/Project/models/exp_bce/test_preds.npz")
y_test = data["y_test"]
y_prob = data["y_prob"]

thresholds = [i/100 for i in range(40,81,2)]   # narrower range around good region
windows = [1, 3, 5]   # sliding window sizes (odd numbers preferred)
best = None
print("w, m, threshold, precision, recall, f1")
for w in windows:
    for m in range(1, w+1):   # require m positives in window of size w
        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            # compute majority-in-window voting (centered sliding window)
            votes = []
            pad = (w-1)//2
            padded = np.pad(preds, pad, mode='constant', constant_values=0)
            for i in range(len(preds)):
                window = padded[i:i+w]
                votes.append(1 if window.sum() >= m else 0)
            votes = np.array(votes)
            p = precision_score(y_test, votes, zero_division=0)
            r = recall_score(y_test, votes, zero_division=0)
            f1 = f1_score(y_test, votes, zero_division=0)
            print(f"{w}, {m}, {t:.2f}, {p:.4f}, {r:.4f}, {f1:.4f}")
            if best is None or f1 > best[0]:
                best = (f1, w, m, t, p, r)

print("\nBest by F1 overall:", best)