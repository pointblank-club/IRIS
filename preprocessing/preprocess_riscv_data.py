import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# === CONFIG ===
INPUT_FILE = "tools/training_data/training_data.json"
OUT_JSON = "tools/training_data/training_data_flat.json"
OUT_CSV = "tools/training_data/training_data_flat.csv"

# === LOAD DATA ===
with open(INPUT_FILE, "r") as f:
    raw = json.load(f)

data = raw["data"]
flat_samples = []

for entry in data:
    feats = entry["features"]
    seq = entry["pass_sequence"]
    runtime = entry["execution_time"]
    bin_size = entry["binary_size"]

    flat_samples.append({
        "program": entry["program"],
        "program_features": feats,
        "pass_sequence": seq,
        "runtime": runtime,
        "binary_size": bin_size
    })

# === FLATTEN FOR XGBOOST (tabular) ===
# Extract unique passes to one-hot encode
all_passes = sorted({p for s in [d["pass_sequence"] for d in flat_samples] for p in s})
pass_to_idx = {p: i for i, p in enumerate(all_passes)}

rows = []
for sample in flat_samples:
    row = {}
    # Add program features
    for k, v in sample["program_features"].items():
        row[k] = v
    # One-hot encode pass sequence
    for p in all_passes:
        row[f"pass_{p}"] = 1 if p in sample["pass_sequence"] else 0
    row["runtime"] = sample["runtime"]
    row["binary_size"] = sample["binary_size"]
    rows.append(row)

df = pd.DataFrame(rows)

# === NORMALIZE PROGRAM FEATURES (for ML models) ===
feature_cols = [c for c in df.columns if c.startswith("total_") or c.startswith("num_") or c.endswith("_ratio")]
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# === SAVE OUTPUTS ===
# Flat JSON for Transformer / PyTorch Dataset
with open(OUT_JSON, "w") as f:
    json.dump(flat_samples, f, indent=2)

# CSV for XGBoost
df.to_csv(OUT_CSV, index=False)

print(f"[+] Saved Transformer-ready JSON -> {OUT_JSON}")
print(f"[+] Saved XGBoost-ready CSV -> {OUT_CSV}")
print(f"[+] {len(flat_samples)} samples, {len(all_passes)} unique passes")
