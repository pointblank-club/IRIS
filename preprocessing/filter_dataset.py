import json
import random
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter('ignore', FutureWarning)
random.seed(42)

INPUT_PATH = Path('tools/training_data/new_flattened_hybrid_data.json')
OUTPUT_PATH = Path('tools/training_data/filtered_hybrid_data.json')
TOP_K = 20
MID_SAMPLES = 2
MIN_HW_COUNT = 5

with INPUT_PATH.open('r', encoding='utf-8') as f:
    raw = json.load(f)

data = raw if isinstance(raw, list) else raw.get('data', [])
if not isinstance(data, list):
    raise TypeError('Expected a list of samples in the dataset.')

full_df = pd.DataFrame(data)
if 'program' not in full_df.columns:
    full_df['program'] = full_df.get('program_id', 'unknown_program')

machine_cols = [col for col in full_df.columns if col.startswith('machine_')]
if machine_cols:
    def make_signature(row):
        parts = []
        for col in machine_cols:
            val = row.get(col)
            if val in (None, '', False):
                continue
            parts.append(f"{col}={val}")
        return '|'.join(parts) if parts else 'default'
    full_df['hardware_signature'] = full_df.apply(make_signature, axis=1)
else:
    full_df['hardware_signature'] = 'default'

full_df = full_df.dropna(subset=['execution_time'])
full_df['execution_time'] = pd.to_numeric(full_df['execution_time'], errors='coerce')
full_df = full_df.dropna(subset=['execution_time'])

full_df = full_df.reset_index(drop=True)

group_cols = ['program', 'hardware_signature']
counts_before = full_df.groupby(group_cols).size()
exec_stats_before = full_df.groupby(group_cols)['execution_time'].agg(['min', 'max', 'mean'])
print(f"Before filtering -> total samples: {len(full_df)}")
print(f"Before filtering -> avg sequences per program-hw: {counts_before.mean():.2f}")
print(f"Before filtering -> mean(min exec_time): {exec_stats_before['min'].mean():.6f}")
print(f"Before filtering -> mean(max exec_time): {exec_stats_before['max'].mean():.6f}")
print(f"Before filtering -> execution_time mean: {full_df['execution_time'].mean():.6f}")
print(f"Before filtering -> execution_time variance: {full_df['execution_time'].var():.6f}")

filtered_parts = []
for _, group_df in full_df.groupby(group_cols):
    group_sorted = group_df.sort_values('execution_time', ascending=True)
    if group_sorted.empty:
        continue
    top_subset = group_sorted.head(TOP_K)
    if len(group_sorted) > 2 * TOP_K:
        mid_pool = group_sorted.iloc[TOP_K:-TOP_K]
    else:
        mid_pool = group_sorted.iloc[TOP_K:]
    if len(mid_pool) > 0:
        mid_subset = mid_pool.sample(n=min(MID_SAMPLES, len(mid_pool)), random_state=42)
    else:
        mid_subset = pd.DataFrame(columns=group_sorted.columns)
    combined = pd.concat([top_subset, mid_subset], ignore_index=True)
    threshold = group_sorted['execution_time'].quantile(0.9)
    combined = combined[combined['execution_time'] <= threshold]
    combined = combined.drop_duplicates(subset='pass_sequence')
    filtered_parts.append(combined)

if filtered_parts:
    filtered_df = pd.concat(filtered_parts, ignore_index=True)
else:
    filtered_df = pd.DataFrame(columns=full_df.columns)

hw_counts = Counter(filtered_df['hardware_signature'])
all_hw_counts = Counter(full_df['hardware_signature'])
reinforce_parts = []
for hw, count in hw_counts.items():
    if count < MIN_HW_COUNT:
        needed = min(MIN_HW_COUNT - count, all_hw_counts[hw])
        candidates = full_df[full_df['hardware_signature'] == hw].sort_values('execution_time').head(needed)
        reinforce_parts.append(candidates)
if reinforce_parts:
    filtered_df = pd.concat([filtered_df] + reinforce_parts, ignore_index=True)

filtered_df = filtered_df.drop_duplicates(subset=['program', 'hardware_signature', 'pass_sequence'])

counts_after = filtered_df.groupby(group_cols).size()
if not counts_after.empty:
    exec_stats_after = filtered_df.groupby(group_cols)['execution_time'].agg(['min', 'max', 'mean'])
    print(f"After filtering -> total samples: {len(filtered_df)}")
    print(f"After filtering -> avg sequences per program-hw: {counts_after.mean():.2f}")
    print(f"After filtering -> mean(min exec_time): {exec_stats_after['min'].mean():.6f}")
    print(f"After filtering -> mean(max exec_time): {exec_stats_after['max'].mean():.6f}")
    print(f"After filtering -> execution_time mean: {filtered_df['execution_time'].mean():.6f}")
    print(f"After filtering -> execution_time variance: {filtered_df['execution_time'].var():.6f}")
else:
    print('After filtering -> no samples remain.')

filtered_df = filtered_df.drop(columns=['hardware_signature'])
filtered_records = filtered_df.to_dict(orient='records')
with OUTPUT_PATH.open('w', encoding='utf-8') as f:
    json.dump(filtered_records, f, indent=2)

print(f"Filtered dataset saved to {OUTPUT_PATH}")
