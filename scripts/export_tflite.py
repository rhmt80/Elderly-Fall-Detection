"""
Export trained Keras model to TFLite, with parity check + threshold/scaler config.

Outputs to <out-dir>:
  model.tflite              float32 TFLite (default — matches Android's TFLiteRunner)
  model_float16.tflite      float16-weight quant (smaller, similar accuracy)
  parity.json               max abs diff between Keras and TFLite predictions
  deploy_config.json        threshold + smoothing + scaler stats for the Android app
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
import joblib


def representative_gen(X, n=300, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[:n]
    for i in idx:
        yield [X[int(i):int(i) + 1].astype(np.float32)]


def convert(model, out_path, float16=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_bytes = converter.convert()
    Path(out_path).write_bytes(tflite_bytes)
    return len(tflite_bytes)


def tflite_predict(tflite_path, X):
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    preds = np.zeros(len(X), dtype=np.float32)
    for i in range(len(X)):
        interp.set_tensor(inp["index"], X[i:i + 1].astype(np.float32))
        interp.invoke()
        preds[i] = interp.get_tensor(out["index"]).ravel()[0]
    return preds


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--model", default=str(root / "models" / "v2_clean" / "best_model.keras"))
    ap.add_argument("--data", default=str(root / "processed" / "dataset.npz"))
    ap.add_argument("--scaler", default=str(root / "processed" / "scaler.pkl"))
    ap.add_argument("--metrics", default=str(root / "evaluation_outputs" / "v2_clean" / "metrics.json"))
    ap.add_argument("--out-dir", default=str(root / "models" / "v2_clean"))
    ap.add_argument("--copy-to", default=str(root / "model.tflite"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.model, compile=False)
    data = np.load(args.data, allow_pickle=False)
    X_test = data["X_test"].astype(np.float32)
    X_train = data["X_train"].astype(np.float32)

    fp32_path = out_dir / "model.tflite"
    fp16_path = out_dir / "model_float16.tflite"
    sz32 = convert(model, fp32_path, float16=False)
    sz16 = convert(model, fp16_path, float16=True)
    print(f"float32 tflite: {sz32/1024:.1f} KB  -> {fp32_path}")
    print(f"float16 tflite: {sz16/1024:.1f} KB  -> {fp16_path}")

    sample = X_test[:500]
    keras_preds = model.predict(sample, batch_size=128, verbose=0).ravel()
    tflite_preds = tflite_predict(fp32_path, sample)
    parity = {
        "n": int(len(sample)),
        "max_abs_diff": float(np.max(np.abs(keras_preds - tflite_preds))),
        "mean_abs_diff": float(np.mean(np.abs(keras_preds - tflite_preds))),
        "corr": float(np.corrcoef(keras_preds, tflite_preds)[0, 1]),
    }
    with open(out_dir / "parity.json", "w") as f:
        json.dump(parity, f, indent=2)
    print("parity:", parity)

    with open(args.metrics) as f:
        metrics = json.load(f)
    sel = metrics["selected_on_val"]
    scaler = joblib.load(args.scaler)
    deploy = {
        "input_shape": [int(X_test.shape[1]), int(X_test.shape[2])],
        "channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "sample_rate_hz": 50,
        "threshold": sel["threshold"],
        "smoothing": {"window": sel["window"], "min_pos": sel["min_pos"]},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "test_metrics": metrics["test"],
    }
    with open(out_dir / "deploy_config.json", "w") as f:
        json.dump(deploy, f, indent=2)
    print("deploy_config saved to", out_dir / "deploy_config.json")

    if args.copy_to:
        shutil.copy(fp32_path, args.copy_to)
        print(f"copied {fp32_path} -> {args.copy_to}")


if __name__ == "__main__":
    main()
