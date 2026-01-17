#!/usr/bin/env python3
"""
Test script for the simplified LLVM optimization API
"""

import requests
import json
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:5001/api/llvm"

# Sample C programs for testing
SAMPLE_PROGRAMS = {
    "simple_loop": """
#include <stdio.h>

int main() {
    int sum = 0;
    for (int i = 0; i < 1000; i++) {
        sum += i;
    }
    printf(\"Sum: %d\n\", sum);
    return 0;
}
""",
    "matrix_multiply": """
#include <stdio.h>

#define N 10

int main() {
    int a[N][N], b[N][N], c[N][N];
    
    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = i + j;
            b[i][j] = i - j;
            c[i][j] = 0;
        }
    }
    
    // Matrix multiplication
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    
    // Print result
    printf(\"Result: %d\n\", c[N-1][N-1]);
    return 0;
}
""",
    "fibonacci": """
#include <stdio.h>

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    int n = 20;
    int result = fibonacci(n);
    printf(\"Fibonacci(%d) = %d\n\", n, result);
    return 0;
}
""",
}

# Example ML-generated pass sequences
ML_PASS_SEQUENCES = [
    # Simple optimization sequence
    ["mem2reg", "simplifycfg", "instcombine"],
    # Loop optimization focused
    ["mem2reg", "loop-simplify", "loop-rotate", "licm", "loop-unroll", "simplifycfg"],
    # Aggressive optimization
    [
        "mem2reg",
        "gvn",
        "simplifycfg",
        "instcombine",
        "loop-simplify",
        "loop-rotate",
        "licm",
        "loop-unroll",
        "sccp",
        "dce",
        "simplifycfg",
    ],
]


def test_feature_extraction(program_name: str, code: str) -> Dict[str, Any]:
    """Test feature extraction endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing feature extraction for: {program_name}")
    print(f"{'='*60}")

    response = requests.post(
        f"{BASE_URL}/features",
        json={"code": code, "target_arch": "riscv64"},
        timeout=10,
    )

    if response.status_code == 200:
        data = response.json()
        if data["success"]:
            print(f"✓ Features extracted successfully")
            print(f"  Feature count: {data['feature_count']}")
            # Show first 5 features as example
            if data.get("features"):
                features = list(data["features"].items())[:5]
                print(f"  Sample features:")
                for key, value in features:
                    print(f"    - {key}: {value}")
        else:
            print(f"✗ Feature extraction failed: {data.get('error')}")
    else:
        print(f"✗ API error: {response.status_code}")

    return response.json() if response.status_code == 200 else None


def test_ml_optimization(program_name: str, code: str, passes: list) -> Dict[str, Any]:
    """Test ML optimization endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing ML optimization for: {program_name}")
    print(f"Passes: {', '.join(passes[:3])}{'...' if len(passes) > 3 else ''}")
    print(f"{'='*60}")

    response = requests.post(
        f"{BASE_URL}/optimize",
        json={"code": code, "ir_passes": passes, "target_arch": "riscv64"},
        timeout=10,
    )

    if response.status_code == 200:
        data = response.json()
        if data["success"]:
            metrics = data["metrics"]
            print(f"✓ ML optimization completed")
            print(f"  Execution time: {metrics['execution_time_avg']:.6f}s")
            print(f"  Binary size: {metrics['binary_size']} bytes")
            print(f"  Pass count: {metrics['pass_count']}")
            print(f"  Optimization time: {metrics['optimization_time']:.4f}s")
        else:
            print(f"✗ Optimization failed: {data.get('error')}")
    else:
        print(f"✗ API error: {response.status_code}")

    return response.json() if response.status_code == 200 else None


def test_standard_optimizations(program_name: str, code: str) -> Dict[str, Any]:
    """Test standard optimizations endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing standard optimizations for: {program_name}")
    print(f"{'='*60}")

    response = requests.post(
        f"{BASE_URL}/standard",
        json={
            "code": code,
            "opt_levels": ["-O0", "-O1", "-O2", "-O3"],
            "target_arch": "riscv64",
        },
        timeout=10,
    )

    if response.status_code == 200:
        data = response.json()
        if data["success"]:
            print(f"✓ Standard optimizations completed")
            for opt_level, metrics in data["results"].items():
                if metrics.get("success"):
                    print(f"  {opt_level}:")
                    print(f"    Execution time: {metrics['execution_time_avg']:.6f}s")
                    print(f"    Binary size: {metrics['binary_size']} bytes")
                else:
                    print(f"  {opt_level}: Failed - {metrics.get('error')}")
        else:
            print(f"✗ Standard optimization failed: {data.get('error')}")
    else:
        print(f"✗ API error: {response.status_code}")

    return response.json() if response.status_code == 200 else None


def test_comparison(program_name: str, code: str, passes: list) -> Dict[str, Any]:
    """Test comparison endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing ML vs Standard comparison for: {program_name}")
    print(f"{'='*60}")

    response = requests.post(
        f"{BASE_URL}/compare",
        json={"code": code, "ir_passes": passes, "target_arch": "riscv64"},
        timeout=10,
    )

    if response.status_code == 200:
        data = response.json()
        if data["success"]:
            print(f"✓ Comparison completed")

            # ML optimization results
            if data.get("ml_optimization") and data["ml_optimization"].get(
                "execution_time_avg"
            ):
                ml_metrics = data["ml_optimization"]
                print(f"\nML Optimization:")
                print(f"  Execution time: {ml_metrics['execution_time_avg']:.6f}s")
                print(f"  Binary size: {ml_metrics['binary_size']} bytes")

            # Comparison results
            if data.get("comparison"):
                print(f"\nComparison Results:")
                for opt_level, comp in data["comparison"].items():
                    if opt_level == "vs_best":
                        print(f"\n  Best Standard: {comp['best_standard']}")
                        print(f"  ML beats best: {comp['ml_beats_best']}")
                        print(f"  Speedup vs best: {comp['speedup_vs_best']:.2f}x")
                    else:
                        print(f"\n  vs {opt_level}:")
                        print(f"    Speedup: {comp['speedup']:.2f}x")
                        print(f"    Size reduction: {comp['size_reduction']*100:.1f}%")
                        print(f"    ML faster: {comp['ml_faster']}")
        else:
            print(f"✗ Comparison failed: {data.get('error')}")
    else:
        print(f"✗ API error: {response.status_code}")

    return response.json() if response.status_code == 200 else None


def main():
    """Run all tests."""
    print("=" * 80)
    print("IRIS LLVM Optimization API Test Suite")
    print("Target: RISC-V 64-bit")
    print("=" * 80)

    # Check API health
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ API Status: {data['status']}")
            print(f"  Service: {data['service']}")
            print(f"  Target: {data['target_arch']}")
        else:
            print(f"\n✗ API health check failed")
            return
    except requests.ConnectionError:
        print("\n✗ Could not connect to API. Is the server running?")
        print("  Run: python app_simplified.py")
        return

    # Test each sample program
    for program_name, code in SAMPLE_PROGRAMS.items():
        print(f"\n{'#'*80}")
        print(f"# Testing Program: {program_name}")
        print(f"{'#'*80}")

        # 1. Extract features
        features_result = test_feature_extraction(program_name, code)

        # 2. Test ML optimization with different pass sequences
        for i, passes in enumerate(ML_PASS_SEQUENCES[:2]):  # Test first 2 sequences
            print(f"\n--- ML Pass Sequence {i+1} ---")
            ml_result = test_ml_optimization(program_name, code, passes)

        # 3. Test standard optimizations
        std_result = test_standard_optimizations(program_name, code)

        # 4. Test comparison (using the first pass sequence)
        comparison_result = test_comparison(program_name, code, ML_PASS_SEQUENCES[0])

    print("\n" + "=" * 80)
    print("Test Suite Completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
