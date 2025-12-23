# scripts/save_deploy_config.py
import json, os
cfg = {"threshold": 0.74, "smoothing": {"method": "majority_window", "window_size": 3, "min_positives": 1}}
out = "../models/exp_bce/deploy_config.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(cfg, f, indent=2)
print("Saved", out)
