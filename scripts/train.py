"""
Train conv-only fall-detection model on the subject-disjoint splits.

No re-splitting (subjects stay disjoint).
Focal loss only (no double-counting via class_weight).
Light sensor-domain augmentation: Gaussian noise + small Z-axis rotation.
Selects best epoch by val PR-AUC.
"""
from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def focal_loss(gamma=2.0, alpha=0.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    return loss


def build_model(input_shape, dropout=0.3, width=96):
    inp = layers.Input(shape=input_shape, name="input_window")
    x = layers.Conv1D(width, 7, padding="same", activation="relu")(inp)
    x = layers.Conv1D(width, 5, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(width, 3, padding="same", activation="relu")(x)
    x = layers.Conv1D(width, 3, padding="same", activation="relu", dilation_rate=2)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(width, 3, padding="same", activation="relu", dilation_rate=4)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    return models.Model(inp, out, name="conv_falldet")


def make_aug_dataset(X, y, batch_size, noise_std, rot_deg, seed):
    rng = tf.random.Generator.from_seed(seed)

    def aug(x, label):
        rot = rng.uniform([2], -rot_deg, rot_deg) * (np.pi / 180.0)
        cs = tf.cos(rot); sn = tf.sin(rot); z = tf.zeros_like(cs[0])
        Ra = tf.stack([
            tf.stack([cs[0], -sn[0], z]),
            tf.stack([sn[0],  cs[0], z]),
            tf.stack([z,      z,     tf.ones_like(z)]),
        ])
        Rg = tf.stack([
            tf.stack([cs[1], -sn[1], z]),
            tf.stack([sn[1],  cs[1], z]),
            tf.stack([z,      z,     tf.ones_like(z)]),
        ])
        acc = tf.matmul(x[:, 0:3], Ra)
        gyr = tf.matmul(x[:, 3:6], Rg)
        x = tf.concat([acc, gyr], axis=1)
        x = x + rng.normal(tf.shape(x), stddev=noise_std)
        return x, label

    ds = tf.data.Dataset.from_tensor_slices((X, y.astype(np.float32)))
    ds = ds.shuffle(8192, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--data", default=str(root / "processed" / "dataset.npz"))
    ap.add_argument("--out-dir", default=str(root / "models" / "v2_clean"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--focal-alpha", type=float, default=0.75)
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--rot-deg", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-aug", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data, allow_pickle=False)
    X_train = data["X_train"].astype(np.float32); y_train = data["y_train"].astype(np.int8)
    X_val = data["X_val"].astype(np.float32); y_val = data["y_val"].astype(np.int8)
    print("Train:", X_train.shape, "fall=", y_train.mean())
    print("Val:  ", X_val.shape, "fall=", y_val.mean())

    if args.no_aug:
        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train.astype(np.float32)))
            .shuffle(8192, seed=args.seed)
            .batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        )
    else:
        train_ds = make_aug_dataset(X_train, y_train, args.batch_size, args.noise_std, args.rot_deg, args.seed)

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val.astype(np.float32)))
        .batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    )

    model = build_model(X_train.shape[1:], dropout=args.dropout, width=args.width)
    model.compile(
        optimizer=optimizers.Adam(args.lr),
        loss=focal_loss(gamma=2.0, alpha=args.focal_alpha),
        metrics=[
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.BinaryAccuracy(name="acc"),
        ],
    )
    model.summary()
    best_path = str(out_dir / "best_model.keras")
    cb_list = [
        callbacks.ModelCheckpoint(best_path, monitor="val_pr_auc", mode="max", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=args.patience, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1, min_lr=1e-6),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cb_list, verbose=2)
    with open(out_dir / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved best model to {best_path}")


if __name__ == "__main__":
    main()
