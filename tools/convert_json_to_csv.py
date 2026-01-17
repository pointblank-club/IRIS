#!/usr/bin/env python3
"""
Convert Hybrid JSON Training Data to CSV Format
Groups all sequences for each program into a single row with aggregated lists.
"""

import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def convert_hybrid_json_to_csv(json_path: str, csv_path: str):
    """
    Convert hybrid training data JSON to CSV format.
    Groups all sequences per program into aggregated lists.
    """
    print(f"Loading JSON data from: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    # Group data by program
    program_data = defaultdict(
        lambda: {
            "features": None,
            "sequences": [],
            "sequence_lengths": [],
            "machine_configs": [],
            "execution_times": [],
            "binary_sizes": [],
        }
    )

    for point in data["data"]:
        program = point["program"]

        # Store features (same for all sequences of a program)
        if program_data[program]["features"] is None:
            program_data[program]["features"] = point["features"]

        # Aggregate sequences
        program_data[program]["sequences"].append(point["ir_passes"])
        program_data[program]["sequence_lengths"].append(point["ir_pass_count"])
        program_data[program]["machine_configs"].append(point.get("machine_config", {}))
        program_data[program]["execution_times"].append(point["execution_time"])
        program_data[program]["binary_sizes"].append(point["binary_size"])

    print(f"Grouped data for {len(program_data)} unique programs")

    # Prepare CSV headers
    feature_keys = sorted(list(data["data"][0]["features"].keys()))
    headers = ["program"]
    headers.extend([f"features_{key}" for key in feature_keys])
    headers.extend(
        [
            "sequences",
            "sequence_lengths",
            "machine_configs",
            "execution_times",
            "binary_sizes",
            "num_sequences",
            "avg_execution_time",
            "min_execution_time",
            "max_execution_time",
            "avg_binary_size",
        ]
    )

    # Write CSV
    print(f"Writing CSV to: {csv_path}")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for program in sorted(program_data.keys()):
            pdata = program_data[program]

            row = [program]

            # Add features
            for key in feature_keys:
                row.append(pdata["features"][key])

            # Add aggregated sequence data
            row.append(json.dumps(pdata["sequences"]))
            row.append(json.dumps(pdata["sequence_lengths"]))
            row.append(json.dumps(pdata["machine_configs"]))
            row.append(json.dumps(pdata["execution_times"]))
            row.append(json.dumps(pdata["binary_sizes"]))

            # Add statistics
            row.append(len(pdata["sequences"]))
            row.append(sum(pdata["execution_times"]) / len(pdata["execution_times"]))
            row.append(min(pdata["execution_times"]))
            row.append(max(pdata["execution_times"]))
            row.append(sum(pdata["binary_sizes"]) / len(pdata["binary_sizes"]))

            writer.writerow(row)

    print(f"✓ Successfully converted {len(program_data)} programs to CSV")
    print(
        f"  Total sequences: {sum(len(p['sequences']) for p in program_data.values())}"
    )


def convert_flat_json_to_csv(json_path: str, csv_path: str):
    """
    Convert JSON to flat CSV (one row per sequence).
    """
    print(f"Loading JSON data from: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    if not data["data"]:
        print("No data points found!")
        return

    # Prepare headers
    feature_keys = sorted(list(data["data"][0]["features"].keys()))
    headers = ["program", "sequence_id"]
    headers.extend([f"features_{key}" for key in feature_keys])
    headers.extend(
        [
            "ir_passes",
            "ir_pass_count",
            "machine_config",
            "machine_flag_count",
            "execution_time",
            "binary_size",
        ]
    )

    # Write CSV
    print(f"Writing flat CSV to: {csv_path}")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for point in data["data"]:
            row = [point["program"], point["sequence_id"]]

            # Add features
            for key in feature_keys:
                row.append(point["features"][key])

            # Add pass and machine config data
            row.append(json.dumps(point["ir_passes"]))
            row.append(point["ir_pass_count"])
            row.append(json.dumps(point.get("machine_config", {})))
            row.append(point.get("machine_flag_count", 0))
            row.append(point["execution_time"])
            row.append(point["binary_size"])

            writer.writerow(row)

    print(f"✓ Successfully converted {len(data['data'])} data points to flat CSV")


def convert_baselines_to_csv(json_path: str, csv_path: str):
    """
    Convert baselines JSON to CSV format.
    """
    print(f"Loading baselines from: {json_path}")
    with open(json_path) as f:
        baselines = json.load(f)

    print(f"Writing baselines CSV to: {csv_path}")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Headers
        headers = [
            "program",
            "O0_time",
            "O0_size",
            "O0_runs",
            "O1_time",
            "O1_size",
            "O1_runs",
            "O2_time",
            "O2_size",
            "O2_runs",
            "O3_time",
            "O3_size",
            "O3_runs",
        ]
        writer.writerow(headers)

        for program in sorted(baselines.keys()):
            baseline = baselines[program]
            row = [program]

            for level in ["-O0", "-O1", "-O2", "-O3"]:
                if level in baseline:
                    row.extend(
                        [
                            baseline[level]["execution_time"],
                            baseline[level]["binary_size"],
                            baseline[level].get("num_runs", 1),
                        ]
                    )
                else:
                    row.extend([None, None, None])

            writer.writerow(row)

    print(f"✓ Successfully converted {len(baselines)} program baselines to CSV")


def main():
    parser = argparse.ArgumentParser(
        description="Convert hybrid JSON training data to CSV format"
    )
    parser.add_argument(
        "--input",
        default="training_data/training_data_hybrid.json",
        help="Input JSON file path",
    )
    parser.add_argument(
        "--output",
        default="training_data/training_data_hybrid.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--format",
        choices=["grouped", "flat"],
        default="grouped",
        help="Output format: 'grouped' (one row per program) or 'flat' (one row per sequence)",
    )
    parser.add_argument("--baselines", help="Also convert baselines JSON to CSV")
    parser.add_argument(
        "--baselines-output",
        default="training_data/baselines.csv",
        help="Output CSV file for baselines",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("JSON TO CSV CONVERTER")
    print("=" * 70)

    # Convert main training data
    if args.format == "grouped":
        convert_hybrid_json_to_csv(args.input, args.output)
    else:
        convert_flat_json_to_csv(args.input, args.output)

    # Convert baselines if specified
    if args.baselines:
        convert_baselines_to_csv(args.baselines, args.baselines_output)

    print("\n" + "=" * 70)
    print("Conversion complete!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  Training data: {args.output}")
    if args.baselines:
        print(f"  Baselines: {args.baselines_output}")


if __name__ == "__main__":
    main()
