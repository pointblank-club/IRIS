#!/usr/bin/env python3
"""
Analyze feature correlations to check for bias in training data.
High correlations indicate potential bias issues.
"""

import json
import argparse
from pathlib import Path
import math


def load_training_data(data_file):
    """Load training data from JSON."""
    with open(data_file) as f:
        return json.load(f)


def compute_correlation(x_values, y_values):
    """Compute Pearson correlation coefficient between two feature vectors."""
    n = len(x_values)
    if n == 0:
        return 0.0

    # Calculate means
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n

    # Calculate correlation
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))

    sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
    sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)

    denominator = math.sqrt(sum_sq_x * sum_sq_y)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def analyze_correlations(dataset, threshold=0.8):
    """
    Analyze feature correlations in the dataset.

    Args:
        dataset: Training dataset dictionary
        threshold: Correlation threshold for flagging (0.8 = 80% correlated)
    """
    data_points = dataset["data"]

    if not data_points:
        print("No data points found!")
        return

    # Get all feature names
    feature_names = list(data_points[0]["features"].keys())

    # Collect feature vectors
    feature_vectors = {}
    for name in feature_names:
        feature_vectors[name] = [point["features"][name] for point in data_points]

    # Compute correlations
    high_correlations = []

    print("=" * 80)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 80)
    print(
        f"Analyzing {len(feature_names)} features across {len(data_points)} data points"
    )
    print(
        f"Correlation threshold: {threshold:.2f} (anything above indicates bias risk)"
    )
    print()

    for i, name1 in enumerate(feature_names):
        for name2 in feature_names[i + 1 :]:
            corr = compute_correlation(feature_vectors[name1], feature_vectors[name2])

            if abs(corr) >= threshold:
                high_correlations.append((name1, name2, corr))

    # Report results
    if high_correlations:
        print(f"⚠️  Found {len(high_correlations)} highly correlated feature pairs:")
        print()

        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        for name1, name2, corr in high_correlations:
            print(f"  {name1:30s} ↔ {name2:30s}  |  r = {corr:+.3f}")

        print()
        print("⚠️  WARNING: High correlations can cause ML bias!")
        print("   - Model can't distinguish independent effects")
        print("   - May need more diverse programs or feature engineering")
    else:
        print(f"✓ No highly correlated features found (threshold = {threshold:.2f})")
        print("  Your features are sufficiently independent for unbiased ML training!")

    print()
    print("=" * 80)

    # Show feature diversity statistics
    print("\nFEATURE DIVERSITY STATISTICS")
    print("=" * 80)

    for name in sorted(feature_names)[:10]:  # Show first 10
        values = feature_vectors[name]
        mean_val = sum(values) / len(values)
        std_val = math.sqrt(sum((v - mean_val) ** 2 for v in values) / len(values))
        min_val = min(values)
        max_val = max(values)

        # Coefficient of variation (std/mean) - high is good for diversity
        cv = (std_val / mean_val * 100) if mean_val != 0 else 0

        print(
            f"{name:30s}  |  min={min_val:8.2f}  max={max_val:8.2f}  "
            f"mean={mean_val:8.2f}  std={std_val:8.2f}  CV={cv:6.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature correlations to detect bias"
    )
    parser.add_argument("data_file", help="Training data JSON file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Correlation threshold for flagging (default: 0.8)",
    )

    args = parser.parse_args()

    # Load data
    dataset = load_training_data(args.data_file)

    # Analyze
    analyze_correlations(dataset, threshold=args.threshold)


if __name__ == "__main__":
    main()
