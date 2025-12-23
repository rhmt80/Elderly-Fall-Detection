import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

data = np.load("/Users/rehmatsinghchawla/Desktop/Project/models/exp_bce/test_preds.npz")
y_test = data["y_test"]
y_prob = data["y_prob"]

t = 0.74
preds = (y_prob >= t).astype(int)

p = precision_score(y_test, preds, zero_division=0)
r = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)
cm = confusion_matrix(y_test, preds)

print(f"Threshold {t:.2f} -> precision: {p:.4f}, recall: {r:.4f}, f1: {f1:.4f}")
print("Confusion matrix:\n", cm)