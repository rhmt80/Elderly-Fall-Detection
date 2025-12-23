#!/usr/bin/env python3
"""
tflite_eval.py
Evaluate a .tflite model on X_test and compute metrics + latency.
Automatically uses LiteRT (ai_edge_litert) if available, then tflite_runtime, then tf.lite.
Usage:
  python scripts/tflite_eval.py --tflite models/tflite/model_float16.tflite --npz processed/processed_data.npz
"""
import argparse
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Runtime detection: try LiteRT -> tflite_runtime -> tf.lite
Interpreter = None
try:
    # LiteRT import path (python package name: ai-edge-litert -> import ai_edge_litert)
    from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter
    Interpreter = LiteRTInterpreter
    print("Using LiteRT Python runtime (ai_edge_litert).")
except Exception:
    try:
        import tflite_runtime.interpreter as tflite_rt
        Interpreter = tflite_rt.Interpreter
        print("Using tflite_runtime.")
    except Exception:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("Using tf.lite.Interpreter (full TensorFlow).")


def load_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if "X_test" not in d.files or "y_test" not in d.files:
        raise ValueError(f"NPZ must contain X_test and y_test. Found keys: {d.files}")
    X_test = d["X_test"].astype(np.float32)
    y_test = d["y_test"].astype(np.int32)
    if y_test.ndim > 1 and y_test.shape[1] == 1:
        y_test = y_test.ravel()
    return X_test, y_test


def run_tflite(model_path: Path, X: np.ndarray, batch_size=64):
    """
    Run a TFLite/LiteRT model on data X and return (probs, mean_latency_s, median_latency_s).
    This version attempts to resize the interpreter input tensor to the desired batch size.
    If resize is unsupported or fails, it falls back to per-sample inference.
    """
    interp = Interpreter(model_path=str(model_path))

    # Try to allocate once with original shapes
    try:
        interp.allocate_tensors()
    except Exception:
        # some runtimes require no-op here; continue
        pass

    # get input / output details (may change after resize)
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    input_index = input_details[0]["index"]
    input_shape = list(input_details[0]["shape"])

    # detect quantization on input
    input_dtype = np.dtype(input_details[0]["dtype"])
    input_is_int = input_dtype == np.int8
    in_scale, in_zero = None, None
    if input_is_int:
        in_scale, in_zero = input_details[0].get("quantization", (None, None))

    n = len(X)

    # prefer to run with the requested batch_size if the runtime supports resizing
    desired_batch = min(batch_size, n)
    can_use_batch = True
    try:
        # Only resize if the model's current batch dim differs
        if input_shape[0] != desired_batch:
            interp.resize_tensor_input(input_index, [desired_batch] + input_shape[1:])
            interp.allocate_tensors()
            # refresh details after resize
            input_details = interp.get_input_details()
            output_details = interp.get_output_details()
            input_index = input_details[0]["index"]
            input_shape = list(input_details[0]["shape"])
    except Exception:
        # Runtime didn't allow resize; will fall back to batch=1 loop
        can_use_batch = False

    all_probs = []
    latencies = []

    if can_use_batch:
        # iterate in chunks of desired_batch
        for i in range(0, n, desired_batch):
            batch = X[i:i+desired_batch]
            # quantize input if needed
            if input_is_int:
                if in_scale is None or in_scale == 0:
                    raise RuntimeError("Input quantization scale is invalid.")
                batch_q = (batch / in_scale + in_zero).round().astype(np.int8)
                in_data = batch_q
            else:
                in_data = batch.astype(input_details[0]["dtype"])

            # ensure correct shape (TFLite sometimes requires full batch dim)
            if in_data.shape[0] != input_shape[0]:
                # pad/truncate batch dimension to match interpreter input shape
                if in_data.shape[0] < input_shape[0]:
                    pad_count = input_shape[0] - in_data.shape[0]
                    pad_shape = (pad_count,) + tuple(in_data.shape[1:])
                    pad_arr = np.zeros(pad_shape, dtype=in_data.dtype)
                    in_data = np.concatenate([in_data, pad_arr], axis=0)
                else:
                    in_data = in_data[: input_shape[0]]

            t0 = time.time()
            interp.set_tensor(input_index, in_data)
            interp.invoke()
            t1 = time.time()

            # compute per-sample latency excluding padded samples
            real_batch_size = min(desired_batch, n - i)
            latencies.append((t1 - t0) / float(real_batch_size))

            out = interp.get_tensor(output_details[0]["index"])
            # dequantize if output is int8
            if np.dtype(output_details[0]["dtype"]) == np.int8:
                out_scale, out_zero = output_details[0].get("quantization", (None, None))
                out = (out.astype(np.float32) - out_zero) * out_scale

            out = np.array(out).reshape(input_shape[0], -1)
            # take only the first real_batch_size rows (ignore padded rows)
            out = out[:real_batch_size]

            if out.shape[1] == 1:
                probs = out.ravel()
            elif out.shape[1] == 2:
                exp = np.exp(out - np.max(out, axis=1, keepdims=True))
                probs = (exp / exp.sum(axis=1, keepdims=True))[:, 1]
            else:
                probs = out.ravel()

            all_probs.append(probs)
    else:
        # fallback: run per-sample inference (batch=1)
        # re-allocate interpreter with batch=1 if possible
        try:
            interp.resize_tensor_input(input_index, [1] + input_shape[1:])
            interp.allocate_tensors()
            input_details = interp.get_input_details()
            output_details = interp.get_output_details()
            input_index = input_details[0]["index"]
        except Exception:
            pass

        for i in range(n):
            sample = X[i:i+1]
            if input_is_int:
                if in_scale is None or in_scale == 0:
                    raise RuntimeError("Input quantization scale is invalid.")
                sample_q = (sample / in_scale + in_zero).round().astype(np.int8)
                in_data = sample_q
            else:
                in_data = sample.astype(input_details[0]["dtype"])

            t0 = time.time()
            interp.set_tensor(input_index, in_data)
            interp.invoke()
            t1 = time.time()
            latencies.append((t1 - t0) / 1.0)

            out = interp.get_tensor(output_details[0]["index"]) 
            if np.dtype(output_details[0]["dtype"]) == np.int8:
                out_scale, out_zero = output_details[0].get("quantization", (None, None))
                out = (out.astype(np.float32) - out_zero) * out_scale

            out = np.array(out).reshape(1, -1)
            if out.shape[1] == 1:
                probs = out.ravel()
            elif out.shape[1] == 2:
                exp = np.exp(out - np.max(out, axis=1, keepdims=True))
                probs = (exp / exp.sum(axis=1, keepdims=True))[:, 1]
            else:
                probs = out.ravel()
            all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)[:n]
    return probs, float(np.mean(latencies)), float(np.median(latencies))


def threshold_sweep_best(y_test, y_prob):
    ths = np.linspace(0.1, 0.9, 17)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in ths:
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tflite", required=True, help="Path to .tflite model")
    p.add_argument("--npz", default="processed/processed_data.npz")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--threshold", type=float, default=None, help="If not set, sweep to pick best by F1")
    args = p.parse_args()

    model_path = Path(args.tflite)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    X_test, y_test = load_npz(args.npz)
    print("Loaded X_test:", X_test.shape, "y_test:", y_test.shape)

    y_prob, mean_lat, med_lat = run_tflite(model_path, X_test, batch_size=args.batch)
    print(f"Latency per sample (s): mean={mean_lat:.6f}, median={med_lat:.6f}")

    if args.threshold is None:
        thr, best_f1 = threshold_sweep_best(y_test, y_prob)
        print("Best threshold by F1:", thr, "F1:", best_f1)
    else:
        thr = args.threshold

    y_pred = (y_prob >= thr).astype(int)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = None
    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = None
    cm = confusion_matrix(y_test, y_pred)

    print("TFLite eval summary (thr=%.3f): precision=%.4f recall=%.4f f1=%.4f auc=%s" % (thr, prec, rec, f1, auc))
    print("Confusion matrix:\n", cm)

if __name__ == "__main__":
    main()