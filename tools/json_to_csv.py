import pandas as pd
import json
from pathlib import Path

# Resolve paths relative to this script's directory
SCRIPT_DIR = Path(__file__).resolve().parent
json_file_path = SCRIPT_DIR / "training_data" / "training_data.json"
csv_file_path = SCRIPT_DIR / "training_data" / "train.csv"

# Load JSON file
if not json_file_path.exists():
    raise FileNotFoundError(
        f"Training JSON not found at: {json_file_path}.\n"
        f"Tip: Run this script from anywhere; paths are resolved relative to {SCRIPT_DIR}"
    )

with open(json_file_path, "r") as f:
    data = json.load(f)

# Extract the 'data' array from JSON
training_data = data["data"]

# Group data by program
grouped_data = {}

for entry in training_data:
    program = entry["program"]

    if program not in grouped_data:
        # Initialize with features (same for all sequences of this program)
        grouped_data[program] = {
            "program": program,
            "features": entry["features"],  # Keep as dict for now
            # 'seq_ids': [],
            "sequences": [],
            "sequence_lengths": [],
            "execution_times": [],
            "binary_sizes": [],
        }

    # Append sequence-specific data
    # grouped_data[program]['seq_ids'].append(entry['sequence_id'])
    grouped_data[program]["sequences"].append(
        entry["pass_sequence"]
    )  # Keep as list for nested structure
    grouped_data[program]["sequence_lengths"].append(entry["sequence_length"])
    grouped_data[program]["execution_times"].append(entry["execution_time"])
    grouped_data[program]["binary_sizes"].append(entry["binary_size"])

# Convert to list of dictionaries for DataFrame
rows = []
for program, prog_data in sorted(grouped_data.items()):
    row = {
        "program": prog_data["program"],
        # 'num_sequences': len(prog_data['seq_ids'])
    }

    # Flatten features with 'features_' prefix
    for feat_name, feat_value in prog_data["features"].items():
        row[f"features_{feat_name}"] = feat_value

    # Add aggregated arrays as string representations
    # row['seq_ids'] = str(prog_data['seq_ids'])
    row["sequences"] = str(prog_data["sequences"])
    row["sequence_lengths"] = str(prog_data["sequence_lengths"])
    row["execution_times"] = str(prog_data["execution_times"])
    row["binary_sizes"] = str(prog_data["binary_sizes"])

    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# Save to CSV
df.to_csv(csv_file_path, index=False)

print(f"Successfully converted JSON to grouped CSV!")
print(f"Total programs: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nSample row structure:")
print(f"  - program: program name")
print(f"  - num_sequences: count of sequences per program")
print(
    f"  - features_*: {len([c for c in df.columns if c.startswith('features_')])} feature columns"
)
print(f"  - seq_ids: list of sequence IDs")
print(f"  - sequences: list of pass sequences")
print(f"  - execution_times: list of execution times")
print(f"  - binary_sizes: list of binary sizes")
print(f"\nCSV saved to: {csv_file_path}")
