# """
# convert_to_tflite.py
# Converts a Keras model to TFLite variants:
#  - float16 quantized (weights in float16)
#  - dynamic-range quantization (safe, small)
#  - int8 (post-training quant, uses representative dataset)
# Outputs saved under models/tflite/
# Usage:
#   python scripts/convert_to_tflite.py --keras /path/to/best_model.keras --npz processed/processed_data.npz
# """
# import argparse
# from pathlib import Path
# import numpy as np
# import tensorflow as tf

# def representative_dataset_generator(npz_path, batch_key="X_train", n_samples=500):
#     data = np.load(npz_path, allow_pickle=True)
#     if batch_key not in data.files:
#         raise ValueError(f"{batch_key} not found in {npz_path}. Keys: {data.files}")
#     X = data[batch_key]
#     rng = np.random.default_rng(42)
#     idx = rng.permutation(len(X))[:min(n_samples, len(X))]
#     for i in idx:
#         sample = X[int(i)].astype(np.float32)
#         # yield sample with batch dim
#         yield [np.expand_dims(sample, axis=0)]

# def convert(model_path: Path, npz_path: Path, out_dir: Path, repr_key="X_train"):
#     out_dir.mkdir(parents=True, exist_ok=True)
#     print("Loading Keras model:", model_path)
#     model = tf.keras.models.load_model(str(model_path), compile=False)

#     # 1) float16
#     conv = tf.lite.TFLiteConverter.from_keras_model(model)
#     conv.optimizations = [tf.lite.Optimize.DEFAULT]
#     conv.target_spec.supported_types = [tf.float16]
#     tflite_float16 = conv.convert()
#     (out_dir / "model_float16.tflite").write_bytes(tflite_float16)
#     print("Saved float16 ->", out_dir / "model_float16.tflite", "bytes:", len(tflite_float16))

#     # 2) dynamic-range
#     conv = tf.lite.TFLiteConverter.from_keras_model(model)
#     conv.optimizations = [tf.lite.Optimize.DEFAULT]
#     tflite_dyn = conv.convert()
#     (out_dir / "model_dynamic_range.tflite").write_bytes(tflite_dyn)
#     print("Saved dynamic-range ->", out_dir / "model_dynamic_range.tflite", "bytes:", len(tflite_dyn))

#     # 3) int8 (post-training quant)
#     try:
#         conv = tf.lite.TFLiteConverter.from_keras_model(model)
#         conv.optimizations = [tf.lite.Optimize.DEFAULT]
#         conv.representative_dataset = lambda: representative_dataset_generator(npz_path, batch_key=repr_key, n_samples=500)
#         # Request full integer ops
#         conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#         # If you want full integer IO, set below. If this causes unsupported op errors, remove these two lines.
#         conv.inference_input_type = tf.int8
#         conv.inference_output_type = tf.int8
#         tflite_int8 = conv.convert()
#         (out_dir / "model_int8.tflite").write_bytes(tflite_int8)
#         print("Saved int8 ->", out_dir / "model_int8.tflite", "bytes:", len(tflite_int8))
#     except Exception as e:
#         print("INT8 conversion failed:", e)
#         print("Hint: try changing repr_key to X_val or using fewer representative samples, or omit full int8 IO to keep float IO.")

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--keras", default="models/best_model.keras", help="Path to .keras model")
#     p.add_argument("--npz", default="processed/processed_data.npz", help="NPZ with train/val/test splits")
#     p.add_argument("--out", default="models/tflite", help="Output directory")
#     p.add_argument("--repr_key", default="X_train", help="Key in NPZ for representative dataset (X_train/X_val)")
#     args = p.parse_args()
#     convert(Path(args.keras), Path(args.npz), Path(args.out), repr_key=args.repr_key)

import tensorflow as tf
import keras
import tempfile
import os

print("TF:", tf.__version__)
print("Keras:", keras.__version__)

# Load Keras-3 model correctly
model = keras.models.load_model(
    "/Users/rehmatsinghchawla/Desktop/models/exp_conv_only/best_model.keras",
    compile=False,
    safe_mode=False
)

# Export to SavedModel
tmp_dir = tempfile.mkdtemp()
saved_model_path = os.path.join(tmp_dir, "saved_model")
model.export(saved_model_path)

print("SavedModel exported to:", saved_model_path)

# Convert to Android-compatible TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.experimental_new_converter = False
converter.optimizations = []

tflite_model = converter.convert()

with open("model_android_compatible.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Android-compatible TFLite model generated")