"""
Build raw (unscaled) windowed dataset from MobiAct annotated CSVs.

Outputs processed/windows_raw.npz with arrays:
    X        (N, 100, 6)  float32  acc_xyz + gyro_xyz
    y        (N,)         int8     1 = fall window, 0 = ADL window
    subject  (N,)         int16    subject id parsed from filename
    activity (N,)         <U3      activity folder name
    recording(N,)         <U64     filename stem (for grouping windows of one recording)

Per-window label: a window is labeled fall (1) iff any sample inside it carries
a fall label (FOL/FKL/BSC/SDL). This is more honest than folder-based labels
because fall recordings contain pre/post-fall ADL frames.

Subject ID parsed strictly: filename FOL_<subj>_<trial>_annotated.csv -> int(subj).
Files that don't parse are skipped (with a count).
"""
from __future__ import annotations
import argparse
import os
import re
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd

FALL_CLASSES = ("FOL", "FKL", "BSC", "SDL")
SAMPLE_RATE = 50
WINDOW_SEC = 2
WINDOW_SIZE = WINDOW_SEC * SAMPLE_RATE  # 100
STEP = WINDOW_SIZE // 2                 # 50% overlap
SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
FNAME_RE = re.compile(r"^([A-Z]{3})_(\d+)_(\d+)_annotated\.csv$")


def iter_files(data_root: Path):
    for activity_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for csv in sorted(activity_dir.glob("*_annotated.csv")):
            m = FNAME_RE.match(csv.name)
            if not m:
                yield csv, activity_dir.name, None, None
                continue
            activity, subj, trial = m.group(1), int(m.group(2)), int(m.group(3))
            yield csv, activity, subj, trial


def windows_from_recording(arr: np.ndarray, sample_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = arr.shape[0]
    if n < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE, arr.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int8)
    starts = np.arange(0, n - WINDOW_SIZE + 1, STEP)
    X = np.stack([arr[s:s + WINDOW_SIZE] for s in starts]).astype(np.float32)
    fall_mask = np.isin(sample_labels, FALL_CLASSES)
    y = np.array([fall_mask[s:s + WINDOW_SIZE].any() for s in starts], dtype=np.int8)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path to MobiAct 'Annotated Data' folder")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "processed" / "windows_raw.npz"))
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    Xs, ys, subs, acts, recs = [], [], [], [], []
    skipped_unparsed = 0
    skipped_short = 0
    files_processed = 0

    for csv_path, activity, subj, trial in iter_files(data_root):
        if subj is None:
            skipped_unparsed += 1
            continue
        try:
            df = pd.read_csv(csv_path, usecols=SENSOR_COLS + ["label"])
        except (ValueError, KeyError):
            df = pd.read_csv(csv_path)
            cols = [c for c in df.columns if c.strip().lower() in SENSOR_COLS]
            if len(cols) < 6 or "label" not in df.columns:
                skipped_unparsed += 1
                continue
            df = df[cols + ["label"]]

        arr = df[SENSOR_COLS].to_numpy(dtype=np.float32)
        labels = df["label"].astype(str).to_numpy()
        if arr.shape[0] < WINDOW_SIZE:
            skipped_short += 1
            continue

        Xw, yw = windows_from_recording(arr, labels)
        if Xw.shape[0] == 0:
            continue
        rec_id = csv_path.stem
        Xs.append(Xw)
        ys.append(yw)
        subs.append(np.full(Xw.shape[0], subj, dtype=np.int16))
        acts.append(np.full(Xw.shape[0], activity, dtype="<U3"))
        recs.append(np.full(Xw.shape[0], rec_id, dtype="<U64"))
        files_processed += 1
        if files_processed % 200 == 0:
            print(f"  processed {files_processed} files, {sum(x.shape[0] for x in Xs)} windows so far")

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    subject = np.concatenate(subs, axis=0)
    activity = np.concatenate(acts, axis=0)
    recording = np.concatenate(recs, axis=0)

    print(f"Files processed: {files_processed}  unparsed: {skipped_unparsed}  too short: {skipped_short}")
    print(f"Windows: {X.shape}, fall ratio: {y.mean():.4f}")
    print(f"Subjects: {len(np.unique(subject))}  activities: {sorted(np.unique(activity).tolist())}")

    np.savez_compressed(out_path, X=X, y=y, subject=subject, activity=activity, recording=recording)
    print(f"Saved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
