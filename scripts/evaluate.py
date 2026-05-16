"""
Evaluate a trained model honestly.

Step 1: tune threshold (and optional majority-window smoothing) on VAL only.
Step 2: report a single set of test metrics at the frozen threshold.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve, roc_curve,
    confusion_matrix, precision_score, recall_score, f1_score,
)


def majority_smooth(preds, window, min_pos):
    if window <= 1:
        return preds
    out = np.zeros_like(preds)
    half = window // 2
    for i in range(len(preds)):
        lo = max(0, i - half)
        hi = min(len(preds), i + half + 1)
        out[i] = 1 if preds[lo:hi].sum() >= min_pos else 0
    return out


def best_threshold_on_val(y_val, p_val, smoothing_options):
    grid = np.linspace(0.05, 0.95, 19)
    best = None
    for win, mp in smoothing_options:
        for t in grid:
            preds = (p_val >= t).astype(np.int8)
            preds = majority_smooth(preds, win, mp)
            f1 = f1_score(y_val, preds, zero_division=0)
            if best is None or f1 > best["f1"]:
                best = {"threshold": float(t), "window": int(win), "min_pos": int(mp), "f1": float(f1)}
    return best


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--model", default=str(root / "models" / "v2_clean" / "best_model.keras"))
    ap.add_argument("--data", default=str(root / "processed" / "dataset.npz"))
    ap.add_argument("--out-dir", default=str(root / "evaluation_outputs" / "v2_clean"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data, allow_pickle=False)
    X_val, y_val = data["X_val"].astype(np.float32), data["y_val"].astype(np.int8)
    X_test, y_test = data["X_test"].astype(np.float32), data["y_test"].astype(np.int8)
    subj_test = data["subject_test"]
    act_test = data["activity_test"]

    model = tf.keras.models.load_model(args.model, compile=False)
    p_val = model.predict(X_val, batch_size=512, verbose=0).ravel()
    p_test = model.predict(X_test, batch_size=512, verbose=0).ravel()

    smoothing_options = [(1, 1), (3, 1), (3, 2), (5, 2), (5, 3)]
    best = best_threshold_on_val(y_val, p_val, smoothing_options)
    print("Best (on VAL):", best)

    test_preds = (p_test >= best["threshold"]).astype(np.int8)
    test_preds = majority_smooth(test_preds, best["window"], best["min_pos"])

    cm = confusion_matrix(y_test, test_preds)
    metrics = {
        "selected_on_val": best,
        "test": {
            "precision": float(precision_score(y_test, test_preds, zero_division=0)),
            "recall":    float(recall_score(y_test, test_preds, zero_division=0)),
            "f1":        float(f1_score(y_test, test_preds, zero_division=0)),
            "pr_auc":    float(average_precision_score(y_test, p_test)),
            "roc_auc":   float(roc_auc_score(y_test, p_test)),
            "n":         int(len(y_test)),
            "n_positive":int(int(y_test.sum())),
            "confusion_matrix": cm.tolist(),
        },
    }

    per_subject = {}
    for s in np.unique(subj_test):
        m = subj_test == s
        if y_test[m].sum() > 0:
            per_subject[int(s)] = {
                "n": int(m.sum()),
                "n_falls": int(y_test[m].sum()),
                "recall": float(recall_score(y_test[m], test_preds[m], zero_division=0)),
            }
    metrics["per_subject_recall"] = per_subject

    per_activity_fpr = {}
    neg_mask = y_test == 0
    for a in np.unique(act_test[neg_mask]):
        m = (act_test == a) & neg_mask
        if m.sum() > 0:
            per_activity_fpr[str(a)] = {
                "n": int(m.sum()),
                "fpr": float(test_preds[m].mean()),
            }
    metrics["per_activity_fpr"] = per_activity_fpr

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(out_dir / "test_preds.npz", y_test=y_test, p_test=p_test, pred=test_preds)

    prec, rec, _ = precision_recall_curve(y_test, p_test)
    fpr, tpr, _ = roc_curve(y_test, p_test)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].plot(rec, prec); axs[0].set_xlabel("recall"); axs[0].set_ylabel("precision")
    axs[0].set_title(f"PR (AP={metrics['test']['pr_auc']:.3f})")
    axs[1].plot(fpr, tpr); axs[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axs[1].set_xlabel("FPR"); axs[1].set_ylabel("TPR")
    axs[1].set_title(f"ROC (AUC={metrics['test']['roc_auc']:.3f})")
    axs[2].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axs[2].text(j, i, str(cm[i, j]), ha="center", va="center")
    axs[2].set_xticks([0, 1]); axs[2].set_yticks([0, 1])
    axs[2].set_xticklabels(["ADL", "fall"]); axs[2].set_yticklabels(["ADL", "fall"])
    axs[2].set_xlabel("predicted"); axs[2].set_ylabel("true")
    axs[2].set_title("Confusion (test)")
    plt.tight_layout(); plt.savefig(out_dir / "curves.png", dpi=120); plt.close()

    print(json.dumps(metrics["test"], indent=2))
    print("Per-subject recall:")
    for s, v in sorted(per_subject.items()):
        print(f"  subject {s}: n={v['n']:>5}  falls={v['n_falls']:>4}  recall={v['recall']:.3f}")
    print("Per-activity FPR:")
    for a, v in sorted(per_activity_fpr.items(), key=lambda kv: -kv[1]["fpr"]):
        print(f"  {a}: n={v['n']:>5}  FPR={v['fpr']:.4f}")


if __name__ == "__main__":
    main()
