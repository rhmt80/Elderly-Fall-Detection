import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, losses, optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# -------------------------
# Defaults
# -------------------------
DEFAULT_DATA_PATH = "/Users/rehmatsinghchawla/Desktop/Project/processed/processed_data.npz"
DEFAULT_OUT_DIR = "../models"
SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# -------------------------
# Utilities
# -------------------------

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for binary classification. Returns a callable loss.
    Usage: model.compile(..., loss=focal_loss(gamma=2., alpha=0.25))
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # per-sample binary crossentropy (not reduced)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        # p_t = model's estimated probability of the true class
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)
    return loss

def set_seed(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_npz(path):
    print(f"Loading data from {path} ...")
    data = np.load(path, allow_pickle=True)
    # common keys
    if "X" in data and "y" in data:
        X = data["X"]
        y = data["y"]
    elif "x" in data and "y" in data:
        X = data["x"]
        y = data["y"]
    else:
        keys = list(data.keys())
        X = data[keys[0]]
        y = data[keys[1]]
    print("Loaded shapes:", X.shape, y.shape)
    return X.astype(np.float32), y.astype(np.int32)

# -------------------------
# Models
# -------------------------

def build_cnn_lstm(input_shape, dropout_rate=0.3):
    """Original CNN + LSTM model (may require Select TF Ops to convert to TFLite)."""
    inp = layers.Input(shape=input_shape, name="input_window")

    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inp)
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.LSTM(64, return_sequences=False, name="lstm")(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="cnn_lstm_falldet")
    return model


def build_conv_only(input_shape, dropout_rate=0.3):
    """Conv-only model that avoids TensorArray/TensorList ops and converts cleanly to TFLite.
    Keeps temporal receptive field via extra conv + global pooling.
    """
    inp = layers.Input(shape=input_shape, name="input_window")

    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)

    # dilated conv to increase receptive field without recurrent ops
    x = layers.Conv1D(64, 3, padding="same", activation="relu", dilation_rate=2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="conv_only_falldet")
    return model

# -------------------------
# Callbacks / metrics
# -------------------------
class ValMetricsCallback(callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred_prob = self.model.predict(X_val, batch_size=256, verbose=0)
        y_pred = (y_pred_prob.ravel() >= 0.5).astype(int)
        p = precision_score(y_val, y_pred, zero_division=0)
        r = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        logs = logs or {}
        logs["val_precision"] = p
        logs["val_recall"] = r
        logs["val_f1"] = f1
        print(f" — val_precision: {p:.4f}  val_recall: {r:.4f}  val_f1: {f1:.4f}")

# -------------------------
# TFLite conversion helpers
# -------------------------

def representative_data_gen(X, num_samples=100):
    """Yield representative samples for post-training quantization.
    Expects X shape (N, T, C).
    """
    # Shuffle and pick samples
    idx = np.random.RandomState(SEED).permutation(len(X))[:num_samples]
    for i in idx:
        # Must yield a list/tuple of input arrays matching signature
        yield [np.expand_dims(X[i], axis=0)]


def try_tflite_conversion(saved_model_dir, out_dir, try_select_ops=True, float16_quant=True, int8=False, X_rep=None):
    os.makedirs(out_dir, exist_ok=True)
    tflite_path = os.path.join(out_dir, "model.tflite")
    print("Attempting TFLite conversion for SavedModel:", saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # Default: enable builtins
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.experimental_new_converter = True

    if float16_quant:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    if int8:
        if X_rep is None:
            raise ValueError("X_rep must be provided for int8 quantization representative dataset")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_data_gen(X_rep, num_samples=300)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    try:
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"Successfully wrote {tflite_path} (%.3f MB)" % size_mb)
        return tflite_path
    except Exception as e:
        print("TFLite conversion failed:", e)
        return None

# -------------------------
# Training loop
# -------------------------

def train(args):
    set_seed(args.seed)
    X, y = load_npz(args.data)

    if X.ndim != 3:
        raise ValueError("X must be 3D: (N, timesteps, channels)")
    input_shape = X.shape[1:]
    print("Input shape (timesteps, channels):", input_shape)

    # Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=args.seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=args.val_split/(1-args.test_split),
        random_state=args.seed, stratify=y_train_val)

    print("Split shapes:")
    print("  train:", X_train.shape, np.sum(y_train), "positive labels")
    print("  val:  ", X_val.shape, np.sum(y_val), "positive labels")
    print("  test: ", X_test.shape, np.sum(y_test), "positive labels")

    cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {int(i): float(w) for i, w in enumerate(cw)}
    print("Class weights:", class_weights)

    # Build model
    if args.model_type == "cnn_lstm":
        model = build_cnn_lstm(input_shape, dropout_rate=args.dropout)
    else:
        model = build_conv_only(input_shape, dropout_rate=args.dropout)

    model.summary()

        # choose loss
    if args.loss == "focal":
        chosen_loss = focal_loss(gamma=2.0, alpha=0.25)
    else:
        chosen_loss = losses.BinaryCrossentropy()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss=chosen_loss,
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    )

    os.makedirs(args.out_dir, exist_ok=True)
    # prefer native Keras format
    best_path = os.path.join(args.out_dir, "best_model_savedmodel")
    final_keras = None
    saved_model_dir = os.path.join(args.out_dir, "saved_model")

    cb_list = [
        callbacks.ModelCheckpoint(
            best_path,
            monitor="val_loss",
            save_best_only=True,
            save_format="tf",
            verbose=1
        ),
        callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        callbacks.TensorBoard(log_dir=os.path.join(args.out_dir, "tb_logs")),
        ValMetricsCallback(validation_data=(X_val, y_val))
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=2
    )

    # Save Keras & SavedModel
    # model.save(final_keras)
    # print("Saved final Keras model to", final_keras)

    tf.saved_model.save(model, saved_model_dir)
    print("Saved SavedModel to", saved_model_dir)

    # Evaluate on test set
    y_test_prob = model.predict(X_test, batch_size=256).ravel()
    np.savez(os.path.join(args.out_dir, "test_preds.npz"), y_test=y_test, y_prob=y_test_prob)
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    p = precision_score(y_test, y_test_pred, zero_division=0)
    r = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_test_pred)
    print("Test Precision: %.4f  Recall: %.4f  F1: %.4f" % (p, r, f1))
    print("Confusion matrix:\n", cm)

    metrics_out = {
        "test_precision": float(p),
        "test_recall": float(r),
        "test_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "history": {k: [float(x) for x in v] for k, v in history.history.items()}
    }
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("Saved test metrics to", os.path.join(args.out_dir, "test_metrics.json"))

    # Optional TFLite conversion
    if args.try_tflite:
        # try float16 with Select TF Ops as fallback
        tflite_path = try_tflite_conversion(saved_model_dir, args.out_dir,
                                           try_select_ops=True, float16_quant=True,
                                           int8=args.int8, X_rep=(X_train if args.int8 else None))
        if tflite_path is None:
            print("TFLite conversion did not succeed. Consider switching to --model_type conv_only and re-training.")

    print("Training complete.")
    return model, saved_model_dir

# -------------------------
# Threshold sweep utility
# -------------------------

def threshold_sweep(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.2, 0.9, 15)
    rows = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        rows.append((t, p, r, f1))
    print("threshold, precision, recall, f1")
    for r in rows:
        print("%.3f, %.4f, %.4f, %.4f" % r)
    return rows

# -------------------------
# Argparse
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Path to processed .npz with X and y")
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--val_split", type=float, default=VALIDATION_SPLIT)
    p.add_argument("--test_split", type=float, default=TEST_SPLIT)
    p.add_argument("--model_type", type=str, choices=["cnn_lstm", "conv_only"], default="conv_only",
                   help="Model architecture to use. Use conv_only if you need a TFLite-friendly model.")
    p.add_argument("--try_tflite", action="store_true", help="Attempt TFLite conversion after training")
    p.add_argument("--int8", action="store_true", help="Attempt int8 quantization (requires representative dataset and may fail for LSTM models)")
    p.add_argument("--loss", type=str, choices=["bce", "focal"], default="bce",
                   help="Loss to use: 'bce' for BinaryCrossentropy (default) or 'focal' for focal loss")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    trained_model, saved_model_dir = train(args)
