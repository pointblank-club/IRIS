#!/usr/bin/env python3
"""
Debug script to analyze why transformer predictions are identical
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from services.llvm_optimization_service import LLVMOptimizationService
import numpy as np

# Sample C codes with different characteristics
test_codes = {
    "simple_loop": """
#include <stdio.h>
int main() {
    int sum = 0;
    for(int i = 0; i < 100; i++) {
        sum += i;
    }
    return sum;
}
""",
    "nested_loops": """
#include <stdio.h>
int main() {
    int sum = 0;
    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            sum += i * j;
        }
    }
    return sum;
}
""",
    "complex_computation": """
#include <stdio.h>
#include <math.h>
int main() {
    double result = 0.0;
    for(int i = 1; i < 1000; i++) {
        result += sqrt((double)i) / i;
    }
    return (int)result;
}
""",
    "function_calls": """
#include <stdio.h>
int factorial(int n) {
    if(n <= 1) return 1;
    return n * factorial(n-1);
}
int main() {
    int sum = 0;
    for(int i = 1; i <= 10; i++) {
        sum += factorial(i);
    }
    return sum;
}
""",
    "array_operations": """
#include <stdio.h>
int main() {
    int arr[100];
    for(int i = 0; i < 100; i++) {
        arr[i] = i * 2;
    }
    int sum = 0;
    for(int i = 0; i < 100; i++) {
        sum += arr[i];
    }
    return sum;
}
""",
}


def main():
    print("=" * 80)
    print("Transformer Model Prediction Diagnostic")
    print("=" * 80)

    # Initialize service
    service = LLVMOptimizationService(target_arch="riscv64")

    # Check if model is loaded
    if service.transformer_model is None:
        print("\n❌ ERROR: Transformer model is NOT loaded!")
        print("   This explains why predictions are identical - using fallback passes")
        return
    else:
        print("\n✓ Transformer model is loaded")

    print("\n" + "=" * 80)
    print("Testing predictions for different programs")
    print("=" * 80)

    all_predictions = []
    all_features = []

    for test_name, code in test_codes.items():
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"{'='*60}")

        # Extract features
        success, features, error = service.extract_features_from_c(code)

        if not success:
            print(f"❌ Feature extraction failed: {error}")
            continue

        print(f"✓ Extracted {len(features)} features")

        # Show some key features
        key_features = [
            "num_basic_blocks",
            "num_instructions",
            "num_loops",
            "num_function_calls",
            "num_branches",
        ]
        print("\nKey features:")
        for feat in key_features:
            value = features.get(feat, features.get(f"feature_{feat}", "N/A"))
            print(f"  {feat}: {value}")

        all_features.append(features)

        # Predict passes
        for opt_level in ["O_0", "O_1", "O_2", "O_3"]:
            success, passes, error = service.predict_passes_with_transformer(
                features, opt_level=opt_level, beam_size=5
            )

            if not success:
                print(f"  ❌ Prediction failed for {opt_level}: {error}")
                continue

            print(f"\n  {opt_level}: {len(passes)} passes")
            print(f"    {', '.join(passes[:10])}{'...' if len(passes) > 10 else ''}")
            all_predictions.append((test_name, opt_level, passes))

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Check if all predictions are identical
    unique_predictions = set()
    for test_name, opt_level, passes in all_predictions:
        unique_predictions.add(tuple(passes))

    print(f"\nTotal predictions made: {len(all_predictions)}")
    print(f"Unique predictions: {len(unique_predictions)}")

    if len(unique_predictions) == 1:
        print("\n⚠️  WARNING: ALL PREDICTIONS ARE IDENTICAL!")
        print("    This indicates a problem with the model:")
        print("    - Model may not be properly trained")
        print("    - Features might be scaled to similar values")
        print("    - Model might be ignoring input features")
        print("\n    Predicted sequence:")
        for passes in unique_predictions:
            print(f"      {', '.join(passes[:15])}{'...' if len(passes) > 15 else ''}")
    elif len(unique_predictions) < len(all_predictions) * 0.3:
        print(
            f"\n⚠️  WARNING: Very few unique predictions ({len(unique_predictions)}/{len(all_predictions)})"
        )
        print("    Model may be undertrained or overfitted")
    else:
        print(
            f"\n✓ Good diversity in predictions ({len(unique_predictions)}/{len(all_predictions)})"
        )

    # Feature diversity analysis
    if len(all_features) > 1:
        print("\n" + "-" * 80)
        print("Feature Diversity Analysis")
        print("-" * 80)

        # Compare first two programs
        feat1 = all_features[0]
        feat2 = all_features[1]

        differences = 0
        for key in feat1.keys():
            val1 = feat1.get(key, 0)
            val2 = feat2.get(key, 0)
            if val1 != val2:
                differences += 1

        similarity_pct = (1 - differences / len(feat1)) * 100
        print(f"Feature similarity between first two programs: {similarity_pct:.1f}%")

        if similarity_pct > 80:
            print("⚠️  Features are very similar - might explain identical predictions")
        else:
            print("✓ Features show good diversity")

    print("\n" + "=" * 80)
    print("Recommendation:")
    if len(unique_predictions) <= 2:
        print("  The model is not generating diverse predictions.")
        print("  Solutions:")
        print("  1. Retrain the model with more diverse training data")
        print("  2. Check if feature extraction is working correctly")
        print("  3. Verify that scaled features maintain diversity")
        print("  4. Increase beam_size or use sampling-based generation")
    print("=" * 80)


if __name__ == "__main__":
    main()
