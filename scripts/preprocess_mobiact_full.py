# scripts/preprocess_mobiact_full.py
import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import sys, os
sys.path.append(os.path.dirname(__file__))
from preprocess_mobiact import DATA_ROOT, FALL_CLASSES, load_all_data
from glob import glob

SAMPLE_RATE = 50
WINDOW_SEC = 2
WINDOW_SIZE = WINDOW_SEC * SAMPLE_RATE      # 100 samples
STEP = WINDOW_SIZE // 2                    # 50 (50% overlap)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

def extract_subject_id(filepath):
    # filename examples: CHU_6_4_annotated.csv -> subject id is token 2 (0-based)
    fname = os.path.basename(filepath)
    parts = fname.split('_')
    # small guard
    if len(parts) >= 3:
        return parts[2]
    return "unknown"

def create_windows_from_file_array(arr):
    # arr shape (N, 6)
    Xw = []
    for start in range(0, arr.shape[0] - WINDOW_SIZE + 1, STEP):
        win = arr[start:start + WINDOW_SIZE]
        if win.shape[0] == WINDOW_SIZE:
            Xw.append(win)
    return np.array(Xw)          # (num_windows, 100, 6)

def build_all_windows():
    # We will read file list from Annotated Data and create windows + subject ids
    files = glob(os.path.join(DATA_ROOT, "*", "*_annotated.csv"))
    X_list, y_list, subj_list = [], [], []
    for fpath in files:
        # determine label by folder name
        folder = os.path.basename(os.path.dirname(fpath))
        label = 1 if folder in FALL_CLASSES else 0

        df = __import__("pandas").read_csv(fpath)
        # attempt to pick columns if present
        cols = [c for c in df.columns if c.strip().lower() in ('acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z')]
        if len(cols) < 6:
            # fallback: assume order x,y,z columns for acc then gyro by position
            arr = df.select_dtypes(include=[float, int]).values
            # if more than 6 columns, try first 6 numeric columns
            arr = arr[:, :6]
        else:
            arr = df[cols].values

        if arr.shape[0] < WINDOW_SIZE:
            continue    # skip very short files
        wins = create_windows_from_file_array(arr)
        X_list.append(wins)
        y_list.extend([label] * wins.shape[0])
        subj = extract_subject_id(os.path.basename(fpath))
        subj_list.extend([subj] * wins.shape[0])

    if not X_list:
        raise RuntimeError("No windows created. Check data and column parsing.")
    X = np.vstack(X_list)                     # (total_windows, 100, 6)
    y = np.array(y_list)
    subj = np.array(subj_list)
    return X, y, subj

def normalize_and_save(X, y, subj):
    # reshape for scaler
    n_windows, T, C = X.shape
    X2 = X.reshape(-1, C)   # (n_windows*T, C)
    scaler = StandardScaler().fit(X2)
    X_scaled = scaler.transform(X2).reshape(n_windows, T, C)
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
    print("Saved scaler to", os.path.join(OUT_DIR, "scaler.pkl"))

    # subjectwise split: group by unique subject ids
    unique_subj = np.unique(subj)
    train_subj, temp_subj = train_test_split(unique_subj, test_size=0.30, random_state=42)
    val_subj, test_subj = train_test_split(temp_subj, test_size=0.5, random_state=42)

    def select_by_subj(arr, subj_arr, chosen):
        mask = np.isin(subj_arr, chosen)
        return arr[mask], subj_arr[mask], y[mask]

    X_train, subj_train, y_train = select_by_subj(X_scaled, subj, train_subj)
    X_val, subj_val, y_val = select_by_subj(X_scaled, subj, val_subj)
    X_test, subj_test, y_test = select_by_subj(X_scaled, subj, test_subj)

    np.savez(os.path.join(OUT_DIR, "processed_data.npz"),
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)
    print("Saved processed_data.npz with shapes:",
          "X_train", X_train.shape, "X_val", X_val.shape, "X_test", X_test.shape)
    # Print class balance
    def bal(v, name):
        print(f"{name}: total={len(v)}, falls={v.sum()}, nonfalls={len(v)-v.sum()}, fall%={(v.sum()/len(v))*100:.2f}%")
    bal(y_train, "Train")
    bal(y_val, "Val")
    bal(y_test, "Test")

if __name__ == "__main__":
    print("Building windows (this may take a minute)...")
    X, y, subj = build_all_windows()
    print("Total windows:", X.shape)
    normalize_and_save(X, y, subj)