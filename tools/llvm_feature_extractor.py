#!/usr/bin/env python3
"""
LLVM-Based Feature Extractor
Uses LLVM's built-in analysis passes for accurate feature extraction.
"""

import re
import json
import subprocess
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict


class LLVMAnalysisExtractor:
    """Extract features using LLVM's built-in analysis passes."""

    def __init__(self):
        self.features = {}

    def extract_from_file(self, ir_file: str) -> Dict[str, Any]:
        """
        Extract features from LLVM IR using analysis passes.

        Args:
            ir_file: Path to .ll or .bc file

        Returns:
            Dictionary of features
        """
        ir_path = Path(ir_file)

        if not ir_path.exists():
            raise FileNotFoundError(f"IR file not found: {ir_file}")

        features = {}

        # Run different LLVM analysis passes
        features.update(self._extract_loop_features(ir_path))
        features.update(self._extract_basic_block_features(ir_path))
        features.update(self._extract_branch_features(ir_path))
        features.update(self._extract_instruction_counts(ir_path))
        features.update(self._extract_cost_model_features(ir_path))

        # Also get basic counts from IR text (still useful)
        features.update(self._extract_text_features(ir_path))

        return features

    def _run_opt_pass(self, ir_file: Path, pass_name: str) -> str:
        """Run an LLVM analysis pass and return output."""
        try:
            result = subprocess.run(
                ["opt", f"-passes=print<{pass_name}>", str(ir_file), "-disable-output"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Analysis output goes to stderr
            return result.stderr
        except Exception as e:
            print(f"Warning: Failed to run {pass_name}: {e}")
            return ""

    def _extract_loop_features(self, ir_file: Path) -> Dict[str, Any]:
        """Extract loop features using LLVM's loop analysis."""
        output = self._run_opt_pass(ir_file, "loops")

        features = {
            "num_loops": 0,
            "max_loop_depth": 0,
            "num_nested_loops": 0,
            "num_loop_headers": 0,
            "num_loop_latches": 0,
        }

        if not output:
            return features

        # Parse loop information
        # Format: "Loop at depth X containing: ..."
        loop_lines = re.findall(r"Loop at depth (\d+)", output)

        features["num_loops"] = len(loop_lines)

        if loop_lines:
            depths = [int(d) for d in loop_lines]
            features["max_loop_depth"] = max(depths)
            features["num_nested_loops"] = sum(1 for d in depths if d > 1)

        features["num_loop_headers"] = len(re.findall(r"<header>", output))
        features["num_loop_latches"] = len(re.findall(r"<latch>", output))
        features["num_loop_exits"] = len(re.findall(r"<exiting>", output))

        return features

    def _extract_basic_block_features(self, ir_file: Path) -> Dict[str, Any]:
        """Extract basic block frequency features."""
        output = self._run_opt_pass(ir_file, "block-freq")

        features = {
            "num_hot_blocks": 0,
            "num_cold_blocks": 0,
            "avg_block_frequency": 0.0,
        }

        if not output:
            return features

        # Parse block frequencies
        # Format: " - : float = X.X, int = Y"
        freq_matches = re.findall(r"float = ([\d.]+)", output)

        if freq_matches:
            frequencies = [float(f) for f in freq_matches]
            features["avg_block_frequency"] = sum(frequencies) / len(frequencies)
            features["num_hot_blocks"] = sum(1 for f in frequencies if f > 10.0)
            features["num_cold_blocks"] = sum(1 for f in frequencies if f < 2.0)

        return features

    def _extract_branch_features(self, ir_file: Path) -> Dict[str, Any]:
        """Extract branch probability features."""
        output = self._run_opt_pass(ir_file, "branch-prob")

        features = {
            "num_likely_branches": 0,
            "num_unlikely_branches": 0,
            "avg_branch_probability": 0.0,
        }

        if not output:
            return features

        # Parse branch probabilities
        # This is approximate - branch-prob output format varies
        prob_matches = re.findall(r"(\d+)%", output)

        if prob_matches:
            probabilities = [int(p) for p in prob_matches]
            features["avg_branch_probability"] = sum(probabilities) / len(probabilities)
            features["num_likely_branches"] = sum(1 for p in probabilities if p > 75)
            features["num_unlikely_branches"] = sum(1 for p in probabilities if p < 25)

        return features

    def _extract_cost_model_features(self, ir_file: Path) -> Dict[str, Any]:
        """Extract instruction cost estimates."""
        output = self._run_opt_pass(ir_file, "cost-model")

        features = {
            "total_instruction_cost": 0,
            "num_expensive_ops": 0,
            "num_cheap_ops": 0,
        }

        if not output:
            return features

        # Parse cost model output
        # Format varies but typically shows "Cost: X for instruction: ..."
        cost_matches = re.findall(r"Cost:\s*(\d+)", output)

        if cost_matches:
            costs = [int(c) for c in cost_matches]
            features["total_instruction_cost"] = sum(costs)
            features["num_expensive_ops"] = sum(1 for c in costs if c > 10)
            features["num_cheap_ops"] = sum(1 for c in costs if c <= 2)
            features["avg_instruction_cost"] = sum(costs) / len(costs) if costs else 0

        return features

    def _extract_instruction_counts(self, ir_file: Path) -> Dict[str, int]:
        """Count instructions by type from IR text."""
        # Read IR file
        if ir_file.suffix == ".bc":
            # Convert to .ll first
            ll_file = ir_file.with_suffix(".ll")
            subprocess.run(
                ["llvm-dis", str(ir_file), "-o", str(ll_file)],
                capture_output=True,
                timeout=10,
            )
            ir_text = ll_file.read_text()
        else:
            ir_text = ir_file.read_text()

        return {
            "num_load": len(re.findall(r"\sload\s+", ir_text)),
            "num_store": len(re.findall(r"\sstore\s+", ir_text)),
            "num_call": len(re.findall(r"\scall\s+", ir_text)),
            "num_br": len(re.findall(r"\sbr\s+", ir_text)),
            "num_add": len(re.findall(r"\sadd\s+", ir_text)),
            "num_mul": len(re.findall(r"\smul\s+", ir_text)),
            "num_icmp": len(re.findall(r"\sicmp\s+", ir_text)),
            "num_phi": len(re.findall(r"\sphi\s+", ir_text)),
            "num_getelementptr": len(re.findall(r"\sgetelementptr\s+", ir_text)),
            "num_alloca": len(re.findall(r"\salloca\s+", ir_text)),
        }

    def _extract_text_features(self, ir_file: Path) -> Dict[str, Any]:
        """Extract basic features from IR text."""
        if ir_file.suffix == ".bc":
            ll_file = ir_file.with_suffix(".ll")
            subprocess.run(
                ["llvm-dis", str(ir_file), "-o", str(ll_file)],
                capture_output=True,
                timeout=10,
            )
            ir_text = ll_file.read_text()
        else:
            ir_text = ir_file.read_text()

        lines = ir_text.split("\n")

        return {
            "total_lines": len(lines),
            "total_functions": len(re.findall(r"define\s+", ir_text)),
            "total_basic_blocks": len(re.findall(r"^[\w\d]+:", ir_text, re.MULTILINE)),
            "total_instructions": len(re.findall(r"^\s+%", ir_text, re.MULTILINE)),
        }


def main():
    """Test the LLVM-based feature extractor."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Extract features using LLVM analysis")
    parser.add_argument("ir_file", help="LLVM IR file (.ll or .bc)")
    parser.add_argument("-o", "--output", help="Output JSON file")

    args = parser.parse_args()

    extractor = LLVMAnalysisExtractor()

    try:
        features = extractor.extract_from_file(args.ir_file)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(features, f, indent=2)
            print(f"Features saved to {args.output}")
        else:
            print(json.dumps(features, indent=2))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
