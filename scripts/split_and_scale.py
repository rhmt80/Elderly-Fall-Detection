"""
Subject-disjoint train/val/test split + train-only StandardScaler.

Reads processed/windows_raw.npz, writes processed/dataset.npz with train/val/test
splits and processed/scaler.pkl. Subjects never overlap across splits, eliminating
both subject leakage and overlapping-window leakage (since overlapping windows
share a subject).

Stratification is by per-subject fall count bucket so each split has a similar
fall/ADL ratio. We do this at the subject level rather than the window level.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def stratify_key(per_subject_fall_frac: np.ndarray) -> np.ndarray:
    return np.array([
        "high" if f > 0.4 else "mid" if f > 0.05 else "low"
        for f in per_subject_fall_frac
    ])


def split_subjects(subjects, fall_fracs, val_frac, test_frac, seed):
    keys = stratify_key(fall_fracs)
    rest_size = val_frac + test_frac
    try:
        train, rest = train_test_split(subjects, test_size=rest_size, random_state=seed, stratify=keys)
        rest_keys = keys[np.isin(subjects, rest)]
        val, test = train_test_split(rest, test_size=test_frac / rest_size, random_state=seed, stratify=rest_keys)
    except ValueError as e:
        print(f"Stratified split failed ({e}); falling back to random.")
        train, rest = train_test_split(subjects, test_size=rest_size, random_state=seed)
        val, test = train_test_split(rest, test_size=test_frac / rest_size, random_state=seed)
    return set(train.tolist()), set(val.tolist()), set(test.tolist())


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--in", dest="inp", default=str(root / "processed" / "windows_raw.npz"))
    ap.add_argument("--out-npz", default=str(root / "processed" / "dataset.npz"))
    ap.add_argument("--out-scaler", default=str(root / "processed" / "scaler.pkl"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = np.load(args.inp, allow_pickle=False)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int8)
    subject = data["subject"].astype(np.int32)
    activity = data["activity"]
    recording = data["recording"]

    unique_subjects = np.unique(subject)
    fall_frac_per_subject = np.array([y[subject == s].mean() for s in unique_subjects])
    print(f"Total subjects: {len(unique_subjects)}  windows: {len(X)}  fall_ratio: {y.mean():.4f}")

    train_set, val_set, test_set = split_subjects(
        unique_subjects, fall_frac_per_subject, args.val_frac, args.test_frac, args.seed
    )
    train_idx = np.array([s in train_set for s in subject])
    val_idx = np.array([s in val_set for s in subject])
    test_idx = np.array([s in test_set for s in subject])

    assert train_idx.sum() + val_idx.sum() + test_idx.sum() == len(X)
    assert not (set(subject[train_idx].tolist()) & set(subject[val_idx].tolist()))
    assert not (set(subject[train_idx].tolist()) & set(subject[test_idx].tolist()))
    assert not (set(subject[val_idx].tolist()) & set(subject[test_idx].tolist()))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))

    def apply(a):
        flat = a.reshape(-1, a.shape[-1])
        return scaler.transform(flat).reshape(a.shape).astype(np.float32)

    X_train_s = apply(X_train)
    X_val_s = apply(X_val)
    X_test_s = apply(X_test)

    joblib.dump(scaler, args.out_scaler)
    np.savez_compressed(
        args.out_npz,
        X_train=X_train_s, y_train=y_train,
        X_val=X_val_s, y_val=y_val,
        X_test=X_test_s, y_test=y_test,
        subject_train=subject[train_idx], subject_val=subject[val_idx], subject_test=subject[test_idx],
        activity_train=activity[train_idx], activity_val=activity[val_idx], activity_test=activity[test_idx],
        recording_train=recording[train_idx], recording_val=recording[val_idx], recording_test=recording[test_idx],
    )

    summary = {
        "train": {"n": int(len(X_train)), "fall_ratio": float(y_train.mean()), "subjects": sorted(train_set)},
        "val":   {"n": int(len(X_val)),   "fall_ratio": float(y_val.mean()),   "subjects": sorted(val_set)},
        "test":  {"n": int(len(X_test)),  "fall_ratio": float(y_test.mean()),  "subjects": sorted(test_set)},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    summary_path = Path(args.out_npz).with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {args.out_npz} and {args.out_scaler}")
    for split in ("train", "val", "test"):
        s = summary[split]
        print(f"  {split}: n={s['n']:>7}  fall={s['fall_ratio']:.4f}  subjects={len(s['subjects'])}")


if __name__ == "__main__":
    main()
