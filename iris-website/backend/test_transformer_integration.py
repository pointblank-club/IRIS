#!/usr/bin/env python3
"""
Test script for transformer model integration with LLVM optimization service.
"""

import requests
import json
from pathlib import Path

# API endpoint
BASE_URL = "http://localhost:5001/api/llvm"

# Test C code samples
TEST_CODE_SIMPLE = """
#include <stdio.h>

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main() {
    int result = factorial(10);
    printf("Factorial of 10 is %d\n", result);
    return 0;
}
"""

TEST_CODE_LOOPS = """
#include <stdio.h>

void bubble_sort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

int main() {
    int arr[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(arr)/sizeof(arr[0]);
    bubble_sort(arr, n);
    
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    return 0;
}
"""

TEST_CODE_MATRIX = """
#include <stdio.h>

#define N 100

void matrix_multiply(int A[N][N], int B[N][N], int C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    static int A[N][N], B[N][N], C[N][N];
    
    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }
    }
    
    matrix_multiply(A, B, C);
    
    // Print a sample result
    printf("C[0][0] = %d\n", C[0][0]);
    return 0;
}
"""


def test_feature_extraction():
    """Test feature extraction endpoint."""
    print("\n=== Testing Feature Extraction ===")

    response = requests.post(
        f"{BASE_URL}/features",
        json={"code": TEST_CODE_SIMPLE, "target_arch": "riscv64"},
        timeout=10,
    )

    if response.status_code == 200:
        data = response.json()
        if data["success"]:
            print(f"✓ Features extracted: {data['feature_count']} features")
            # Print first few features
            features = data["features"]
            feature_names = list(features.keys())[:5]
            for name in feature_names:
                print(f"  - {name}: {features[name]}")
            print(f"  ... and {len(features) - 5} more features")
            return True
        else:
            print(f"✗ Feature extraction failed: {data.get('error', 'Unknown error')}")
    else:
        print(f"✗ HTTP error {response.status_code}")
    return False


def test_transformer_prediction():
    """Test optimization with transformer-predicted passes."""
    print("\n=== Testing Transformer Pass Prediction ===")

    test_cases = [
        ("Simple Recursive", TEST_CODE_SIMPLE, "O_0"),
        ("Loop Optimization", TEST_CODE_LOOPS, "O_2"),
        ("Matrix Computation", TEST_CODE_MATRIX, "O_3"),
    ]

    for name, code, opt_level in test_cases:
        print(f"\nTest: {name} with hint {opt_level}")

        # Test with transformer (no ir_passes provided)
        response = requests.post(
            f"{BASE_URL}/optimize",
            json={
                "code": code,
                "target_arch": "riscv64",
                "use_transformer": True,
                "opt_level_hint": opt_level,
            },
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                metrics = data["metrics"]
                print(
                    f"  ✓ Optimization successful using {data.get('passes_source', 'unknown')}"
                )
                print(f"    - Execution time: {metrics['execution_time_avg']:.6f}s")
                print(f"    - Binary size: {metrics['binary_size']} bytes")
                print(f"    - Pass count: {metrics.get('pass_count', 'N/A')}")
                if "ir_passes" in metrics:
                    print(f"    - First 5 passes: {metrics['ir_passes'][:5]}")
            else:
                print(f"  ✗ Optimization failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"  ✗ HTTP error {response.status_code}")


def test_manual_vs_transformer():
    """Compare manual passes with transformer-predicted passes."""
    print("\n=== Comparing Manual vs Transformer Passes ===")

    # Some common optimization passes
    manual_passes = [
        "mem2reg",
        "simplifycfg",
        "instcombine",
        "reassociate",
        "gvn",
        "licm",
        "loop-unroll",
        "dce",
        "adce",
    ]

    code = TEST_CODE_LOOPS

    # Test with manual passes
    print("\n1. Testing with manual passes...")
    response = requests.post(
        f"{BASE_URL}/optimize",
        json={"code": code, "ir_passes": manual_passes, "target_arch": "riscv64"},
        timeout=10,
    )

    manual_metrics = None
    if response.status_code == 200:
        data = response.json()
        if data["success"]:
            manual_metrics = data["metrics"]
            print(f"  ✓ Manual optimization:")
            print(f"    - Execution time: {manual_metrics['execution_time_avg']:.6f}s")
            print(f"    - Binary size: {manual_metrics['binary_size']} bytes")

    # Test with transformer
    print("\n2. Testing with transformer-predicted passes...")
    response = requests.post(
        f"{BASE_URL}/optimize",
        json={
            "code": code,
            "target_arch": "riscv64",
            "use_transformer": True,
            "opt_level_hint": "O_2",
        },
        timeout=10,
    )

    transformer_metrics = None
    if response.status_code == 200:
        data = response.json()
        if data["success"]:
            transformer_metrics = data["metrics"]
            print(f"  ✓ Transformer optimization:")
            print(
                f"    - Execution time: {transformer_metrics['execution_time_avg']:.6f}s"
            )
            print(f"    - Binary size: {transformer_metrics['binary_size']} bytes")
            print(
                f"    - Passes used: {len(transformer_metrics.get('ir_passes', []))} passes"
            )

    # Compare if both succeeded
    if manual_metrics and transformer_metrics:
        print("\n3. Comparison:")
        speedup = (
            manual_metrics["execution_time_avg"]
            / transformer_metrics["execution_time_avg"]
        )
        size_ratio = transformer_metrics["binary_size"] / manual_metrics["binary_size"]

        print(
            f"  - Speedup: {speedup:.2f}x {'(transformer faster)' if speedup > 1 else '(manual faster)'}"
        )
        print(
            f"  - Size ratio: {size_ratio:.2f}x {'(transformer smaller)' if size_ratio < 1 else '(manual smaller)'}"
        )


def test_compare_endpoint():
    """Test the compare endpoint with transformer."""
    print("\n=== Testing Compare Endpoint with Transformer ===")

    response = requests.post(
        f"{BASE_URL}/compare",
        json={
            "code": TEST_CODE_SIMPLE,
            "target_arch": "riscv64",
            "use_transformer": True,
            "opt_level_hint": "O_2",
        },
        timeout=10,
    )

    if response.status_code == 200:
        data = response.json()
        if data["success"]:
            print("✓ Comparison completed successfully")

            # Show ML optimization results
            if data.get("ml_optimization"):
                ml_opt = data["ml_optimization"]
                if ml_opt.get("execution_time_avg"):
                    print(f"\nML Optimization (Transformer):")
                    print(f"  - Execution time: {ml_opt['execution_time_avg']:.6f}s")
                    print(f"  - Binary size: {ml_opt['binary_size']} bytes")
                    print(f"  - Passes used: {ml_opt.get('pass_count', 'N/A')}")

            # Show standard optimization results
            if data.get("standard_optimizations"):
                print(f"\nStandard Optimizations:")
                for opt_level, metrics in data["standard_optimizations"].items():
                    if metrics.get("success"):
                        print(f"  {opt_level}:")
                        print(
                            f"    - Execution time: {metrics['execution_time_avg']:.6f}s"
                        )
                        print(f"    - Binary size: {metrics['binary_size']} bytes")

            # Show comparisons
            if data.get("comparison"):
                print(f"\nComparisons with Standard:")
                for opt_level, comp in data["comparison"].items():
                    if opt_level != "vs_best":
                        print(f"  vs {opt_level}:")
                        print(f"    - Speedup: {comp['speedup']:.2f}x")
                        print(f"    - Size reduction: {comp['size_reduction']:.2%}")

                if "vs_best" in data["comparison"]:
                    best = data["comparison"]["vs_best"]
                    print(f"\n  vs Best Standard ({best['best_standard']}):")
                    print(f"    - ML beats best: {best['ml_beats_best']}")
                    print(f"    - Speedup: {best['speedup_vs_best']:.2f}x")
        else:
            print(f"✗ Comparison failed: {data.get('error', 'Unknown error')}")
    else:
        print(f"✗ HTTP error {response.status_code}")


def main():
    """Run all tests."""
    print("=" * 80)
    print("Transformer Model Integration Tests")
    print("=" * 80)

    print("\nNote: Make sure the backend server is running on port 5000")
    print("Run: python3 app_simplified.py")

    try:
        # Check health first
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code != 200:
            print("\n✗ Backend server not responding. Please start it first.")
            return

        print("\n✓ Backend server is running")

        # Run tests
        test_feature_extraction()
        test_transformer_prediction()
        test_manual_vs_transformer()
        test_compare_endpoint()

        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)

    except requests.exceptions.ConnectionError:
        print("\n✗ Cannot connect to backend server. Please start it first:")
        print("  cd iris-website/backend")
        print("  python3 app_simplified.py")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")


if __name__ == "__main__":
    main()
