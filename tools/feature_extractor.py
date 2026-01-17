#!/usr/bin/env python3
"""
Feature Extractor for LLVM IR
Extracts comprehensive features from LLVM intermediate representation for ML training.
"""

import re
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter


class LLVMFeatureExtractor:
    """Extract features from LLVM IR bitcode or text representation."""

    def __init__(self):
        self.features = {}

    def extract_from_file(self, ir_file: str) -> Dict[str, Any]:
        """
        Extract features from an LLVM IR file (.ll or .bc).

        Args:
            ir_file: Path to LLVM IR file

        Returns:
            Dictionary of extracted features
        """
        ir_path = Path(ir_file)

        if not ir_path.exists():
            raise FileNotFoundError(f"IR file not found: {ir_file}")

        # Convert .bc to .ll if needed
        if ir_path.suffix == ".bc":
            ll_file = self._convert_bc_to_ll(ir_file)
            ir_text = ll_file.read_text()
        else:
            ir_text = ir_path.read_text()

        return self.extract_from_text(ir_text)

    def _convert_bc_to_ll(self, bc_file: str) -> Path:
        """Convert LLVM bitcode to text format."""
        ll_file = Path(bc_file).with_suffix(".ll")
        try:
            subprocess.run(
                ["llvm-dis", bc_file, "-o", str(ll_file)],
                check=True,
                capture_output=True,
            )
            return ll_file
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert bitcode: {e.stderr.decode()}")

    def extract_from_text(self, ir_text: str) -> Dict[str, Any]:
        """
        Extract features from LLVM IR text.

        Returns a dictionary with ~50 features covering:
        - Function characteristics
        - Instruction counts
        - Control flow complexity
        - Memory operations
        - Loop characteristics
        - Data type usage
        """
        features = {}

        # Split into lines for analysis
        lines = ir_text.split("\n")

        # Extract basic counts
        features.update(self._extract_instruction_counts(ir_text))
        features.update(self._extract_function_features(ir_text))
        features.update(self._extract_control_flow_features(ir_text))
        features.update(self._extract_memory_features(ir_text))
        features.update(self._extract_arithmetic_features(ir_text))
        features.update(self._extract_loop_features(ir_text))
        features.update(self._extract_call_features(ir_text))
        features.update(self._extract_type_features(ir_text))
        features.update(self._extract_complexity_metrics(ir_text))

        # Add derived features
        features.update(self._compute_derived_features(features))

        return features

    def _extract_instruction_counts(self, ir_text: str) -> Dict[str, int]:
        """Count total instructions and basic blocks."""
        return {
            "total_instructions": len(re.findall(r"^\s+%", ir_text, re.MULTILINE)),
            "total_basic_blocks": len(re.findall(r"^[\w\d]+:", ir_text, re.MULTILINE)),
            "total_lines": len(ir_text.split("\n")),
        }

    def _extract_function_features(self, ir_text: str) -> Dict[str, int]:
        """Extract function-related features."""
        functions = re.findall(r"define\s+.*?@(\w+)", ir_text)
        declarations = re.findall(r"declare\s+.*?@(\w+)", ir_text)

        return {
            "num_functions": len(functions),
            "num_declarations": len(declarations),
            "total_function_calls": len(re.findall(r"call\s+", ir_text)),
        }

    def _extract_control_flow_features(self, ir_text: str) -> Dict[str, int]:
        """Extract control flow related features."""
        # Basic counts
        num_br = len(re.findall(r"\sbr\s+", ir_text))
        num_conditional_br = len(re.findall(r"\sbr\s+i1\s+%", ir_text))
        num_unconditional_br = len(re.findall(r"\sbr\s+label\s+%", ir_text))
        num_basic_blocks = len(re.findall(r"^[\w\d]+:", ir_text, re.MULTILINE))

        # Branch density: branches per basic block
        branch_density = num_br / max(1, num_basic_blocks)

        # Critical edges estimation:
        # A critical edge connects a block with multiple successors to a block with multiple predecessors
        # Approximate by looking at conditional branches leading to blocks with multiple incoming edges
        # Count PHI nodes with 2+ incoming values as proxy for blocks with multiple predecessors
        phi_with_multiple_preds = len(
            re.findall(r"phi\s+\w+.*?\[.*?,.*?\].*?\[", ir_text)
        )
        critical_edges = min(num_conditional_br, phi_with_multiple_preds)

        return {
            # Branch instructions
            "num_br": num_br,
            "num_conditional_br": num_conditional_br,
            "num_unconditional_br": num_unconditional_br,
            # Switch and select
            "num_switch": len(re.findall(r"\sswitch\s+", ir_text)),
            "num_select": len(re.findall(r"\sselect\s+", ir_text)),
            # Comparisons
            "num_icmp": len(re.findall(r"\sicmp\s+", ir_text)),
            "num_fcmp": len(re.findall(r"\sfcmp\s+", ir_text)),
            # Return instructions
            "num_ret": len(re.findall(r"\sret\s+", ir_text)),
            # PHI nodes (indicators of complex control flow)
            "num_phi": len(re.findall(r"\sphi\s+", ir_text)),
            # Derived control flow metrics
            "branch_density": branch_density,
            "critical_edges": critical_edges,
        }

    def _extract_memory_features(self, ir_text: str) -> Dict[str, int]:
        """Extract memory operation features."""
        # Count global variable accesses
        global_loads = len(re.findall(r"load\s+.*?@\w+", ir_text))
        global_stores = len(re.findall(r"store\s+.*?@\w+", ir_text))
        global_accesses = global_loads + global_stores

        # Estimate alias pairs by counting pointers with similar usage patterns
        # Count unique pointer variables
        ptr_vars = set(
            re.findall(
                r"%(\w+)\s*=\s*(?:alloca|getelementptr|load.*\*|bitcast.*\*)", ir_text
            )
        )
        # Estimate potential aliases as pairs of pointers used in same contexts
        # This is a rough heuristic: assume 10% of pointer pairs could alias
        num_ptrs = len(ptr_vars)
        alias_pairs = (num_ptrs * (num_ptrs - 1)) // 20 if num_ptrs > 1 else 0

        return {
            # Load/Store operations
            "num_load": len(re.findall(r"\sload\s+", ir_text)),
            "num_store": len(re.findall(r"\sstore\s+", ir_text)),
            # Memory allocation
            "num_alloca": len(re.findall(r"\salloca\s+", ir_text)),
            # Pointer operations
            "num_getelementptr": len(re.findall(r"\sgetelementptr\s+", ir_text)),
            "num_ptrtoint": len(re.findall(r"\sptrtoint\s+", ir_text)),
            "num_inttoptr": len(re.findall(r"\sinttoptr\s+", ir_text)),
            # Memory intrinsics
            "num_memcpy": len(re.findall(r"@llvm\.memcpy", ir_text)),
            "num_memset": len(re.findall(r"@llvm\.memset", ir_text)),
            "num_memmove": len(re.findall(r"@llvm\.memmove", ir_text)),
            # Data flow features
            "global_accesses": global_accesses,
            "alias_pairs": alias_pairs,
        }

    def _extract_arithmetic_features(self, ir_text: str) -> Dict[str, int]:
        """Extract arithmetic and logical operation features."""
        return {
            # Integer arithmetic
            "num_add": len(re.findall(r"\sadd\s+", ir_text)),
            "num_sub": len(re.findall(r"\ssub\s+", ir_text)),
            "num_mul": len(re.findall(r"\smul\s+", ir_text)),
            "num_div": len(re.findall(r"\s[us]div\s+", ir_text)),
            "num_rem": len(re.findall(r"\s[us]rem\s+", ir_text)),
            # Floating point arithmetic
            "num_fadd": len(re.findall(r"\sfadd\s+", ir_text)),
            "num_fsub": len(re.findall(r"\sfsub\s+", ir_text)),
            "num_fmul": len(re.findall(r"\sfmul\s+", ir_text)),
            "num_fdiv": len(re.findall(r"\sfdiv\s+", ir_text)),
            # Bitwise operations
            "num_and": len(re.findall(r"\sand\s+", ir_text)),
            "num_or": len(re.findall(r"\sor\s+", ir_text)),
            "num_xor": len(re.findall(r"\sxor\s+", ir_text)),
            "num_shl": len(re.findall(r"\sshl\s+", ir_text)),
            "num_shr": len(re.findall(r"\s[la]shr\s+", ir_text)),
        }

    def _extract_loop_features(self, ir_text: str) -> Dict[str, int]:
        """
        Extract loop-related features.
        Note: This is approximate as LLVM IR doesn't explicitly mark loops.
        We use heuristics based on back-edges (branches to earlier blocks).
        """
        # Find all basic block labels
        blocks = re.findall(r"^(\w+):", ir_text, re.MULTILINE)

        # Find all branch targets
        branch_targets = re.findall(r"label\s+%(\w+)", ir_text)

        # Count potential back-edges (branches to earlier blocks)
        # This is a heuristic for loop detection
        potential_loops = 0
        block_set = set(blocks)

        # Approximate loop count by counting backward branches
        for i, block in enumerate(blocks):
            # Check if there are branches to this block from later blocks
            # This is simplified; real loop detection is more complex
            pass

        # For now, use PHI nodes as a proxy for loops
        # (PHI nodes are commonly used in loop headers)
        num_phi = len(re.findall(r"\sphi\s+", ir_text))

        # Estimate loop depth by analyzing PHI nodes per function
        functions = re.findall(
            r"define\s+.*?\{(.*?)^\}", ir_text, re.MULTILINE | re.DOTALL
        )
        total_loop_depth = 0
        num_functions_with_loops = 0

        for func in functions:
            phi_in_func = len(re.findall(r"\sphi\s+", func))
            if phi_in_func > 0:
                # Approximate depth by PHI concentration
                num_functions_with_loops += 1
                total_loop_depth += min(phi_in_func, 5)  # Cap at depth 5

        avg_loop_depth = total_loop_depth / max(1, num_functions_with_loops)

        return {
            "estimated_loops": min(num_phi, len(blocks) // 3),  # Conservative estimate
            "num_back_edges": len([t for t in branch_targets if t in block_set]),
            "loop_depth_avg": avg_loop_depth,
        }

    def _extract_call_features(self, ir_text: str) -> Dict[str, int]:
        """Extract function call features."""
        # Direct calls
        direct_calls = len(re.findall(r"call\s+.*?@\w+", ir_text))

        # Indirect calls
        indirect_calls = len(re.findall(r"call\s+.*?\(", ir_text)) - direct_calls

        # Tail calls
        tail_calls = len(re.findall(r"tail\s+call", ir_text))

        # Inline candidates: small functions (heuristic based on instruction count)
        # Functions with fewer instructions are good inline candidates
        functions = re.findall(
            r"define\s+.*?\{(.*?)^\}", ir_text, re.MULTILINE | re.DOTALL
        )
        inline_candidates = 0
        for func in functions:
            inst_count = len(re.findall(r"^\s+%", func, re.MULTILINE))
            if (
                inst_count > 0 and inst_count < 30
            ):  # Small functions (< 30 instructions)
                inline_candidates += 1

        # Call graph edges: unique caller-callee relationships
        # Extract all function names
        defined_funcs = set(re.findall(r"define\s+.*?@(\w+)", ir_text))

        # Count unique call edges
        call_edges = set()
        for func_match in re.finditer(
            r"define\s+.*?@(\w+).*?\{(.*?)^\}", ir_text, re.MULTILINE | re.DOTALL
        ):
            caller = func_match.group(1)
            func_body = func_match.group(2)
            # Find all callees in this function
            callees = re.findall(r"call\s+.*?@(\w+)", func_body)
            for callee in callees:
                call_edges.add((caller, callee))

        call_graph_edges = len(call_edges)

        return {
            "num_direct_calls": direct_calls,
            "num_indirect_calls": max(0, indirect_calls),
            "num_tail_calls": tail_calls,
            "inline_candidates": inline_candidates,
            "call_graph_edges": call_graph_edges,
        }

    def _extract_type_features(self, ir_text: str) -> Dict[str, int]:
        """Extract data type usage features."""
        return {
            # Integer types
            "uses_i1": len(re.findall(r"\bi1\b", ir_text)),
            "uses_i8": len(re.findall(r"\bi8\b", ir_text)),
            "uses_i32": len(re.findall(r"\bi32\b", ir_text)),
            "uses_i64": len(re.findall(r"\bi64\b", ir_text)),
            # Floating point types
            "uses_float": len(re.findall(r"\bfloat\b", ir_text)),
            "uses_double": len(re.findall(r"\bdouble\b", ir_text)),
            # Pointer types
            "uses_ptr": len(re.findall(r"\*", ir_text)),
            # Array types
            "uses_array": len(re.findall(r"\[\d+\s+x\s+", ir_text)),
            # Vector types
            "uses_vector": len(re.findall(r"<\d+\s+x\s+", ir_text)),
        }

    def _extract_complexity_metrics(self, ir_text: str) -> Dict[str, float]:
        """Compute complexity metrics."""
        num_blocks = len(re.findall(r"^[\w\d]+:", ir_text, re.MULTILINE))
        num_branches = len(re.findall(r"\sbr\s+", ir_text))
        num_functions = len(re.findall(r"define\s+", ir_text))

        # Cyclomatic complexity approximation: M = E - N + 2P
        # E = edges (branches), N = nodes (blocks), P = connected components (functions)
        if num_functions > 0:
            cyclomatic = max(1, num_branches - num_blocks + 2 * num_functions)
        else:
            cyclomatic = 1

        return {
            "cyclomatic_complexity": cyclomatic,
            "avg_block_size": (
                len(re.findall(r"^\s+%", ir_text, re.MULTILINE)) / max(1, num_blocks)
            ),
        }

    def _compute_derived_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Compute derived features from basic counts."""
        derived = {}

        # Memory intensity
        total_mem_ops = features.get("num_load", 0) + features.get("num_store", 0)
        total_inst = features.get("total_instructions", 1)
        derived["memory_intensity"] = total_mem_ops / max(1, total_inst)

        # Branch intensity
        derived["branch_intensity"] = features.get("num_br", 0) / max(1, total_inst)

        # Arithmetic intensity (arith_ratio)
        total_arith = (
            features.get("num_add", 0)
            + features.get("num_sub", 0)
            + features.get("num_mul", 0)
            + features.get("num_div", 0)
        )
        derived["arithmetic_intensity"] = total_arith / max(1, total_inst)
        derived["arith_ratio"] = derived["arithmetic_intensity"]  # Alias for clarity

        # Logic ratio: logical operations per instruction
        total_logic = (
            features.get("num_and", 0)
            + features.get("num_or", 0)
            + features.get("num_xor", 0)
            + features.get("num_shl", 0)
            + features.get("num_shr", 0)
        )
        derived["logic_ratio"] = total_logic / max(1, total_inst)

        # Memory ratio (mem_ratio)
        derived["mem_ratio"] = derived["memory_intensity"]  # Alias for clarity

        # Vector ratio: vector operations per instruction
        vector_uses = features.get("uses_vector", 0)
        derived["vector_ratio"] = vector_uses / max(1, total_inst)

        # Floating point operations (combined count)
        floating_point_ops = (
            features.get("num_fadd", 0)
            + features.get("num_fsub", 0)
            + features.get("num_fmul", 0)
            + features.get("num_fdiv", 0)
        )
        derived["floating_point_ops"] = floating_point_ops
        derived["floating_point_ratio"] = floating_point_ops / max(1, total_inst)

        # Call intensity
        derived["call_intensity"] = features.get("total_function_calls", 0) / max(
            1, total_inst
        )

        # PHI ratio (indicates control flow complexity)
        derived["phi_ratio"] = features.get("num_phi", 0) / max(
            1, features.get("total_basic_blocks", 1)
        )

        # Load/store ratio
        num_load = features.get("num_load", 0)
        num_store = features.get("num_store", 0)
        if num_store > 0:
            derived["load_store_ratio"] = num_load / num_store
        else:
            derived["load_store_ratio"] = float(num_load) if num_load > 0 else 1.0

        return derived

    def get_feature_vector(self, features: Dict[str, Any]) -> List[float]:
        """Convert feature dictionary to a flat vector (for ML)."""
        # Define feature order
        feature_names = sorted(features.keys())
        return [float(features[name]) for name in feature_names]

    def get_feature_names(self, features: Dict[str, Any]) -> List[str]:
        """Get ordered list of feature names."""
        return sorted(features.keys())


def extract_features_from_c_source(
    c_file: str, output_bc: str = None, target_arch: str = "riscv64"
) -> Dict[str, Any]:
    """
    Compile C source to LLVM IR and extract features.

    Args:
        c_file: Path to C source file
        output_bc: Optional output path for bitcode file
        target_arch: Target architecture (riscv64, riscv32, or native)

    Returns:
        Dictionary of extracted features
    """
    clang_path = "clang"
    c_path = Path(c_file)

    if not c_path.exists():
        raise FileNotFoundError(f"C source file not found: {c_file}")

    # Generate bitcode
    if output_bc is None:
        output_bc = c_path.with_suffix(".bc")

    # Build clang command with target flags
    clang_cmd = [clang_path]
    if target_arch == "riscv64":
        clang_cmd.extend(["--target=riscv64-unknown-linux-gnu"])
    elif target_arch == "riscv32":
        clang_cmd.extend(["--target=riscv32-unknown-linux-gnu"])
    # else: native compilation (no target flag)

    clang_cmd.extend(["-O0", "-emit-llvm", "-c", str(c_path), "-o", str(output_bc)])

    try:
        subprocess.run(clang_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to compile C source: {e.stderr.decode()}")

    # Extract features
    extractor = LLVMFeatureExtractor()
    features = extractor.extract_from_file(str(output_bc))

    return features


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from LLVM IR for ML training"
    )
    parser.add_argument(
        "input", help="Input file (C source, LLVM IR .ll, or bitcode .bc)"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file for features (JSON format). If not specified, prints to stdout.",
    )
    parser.add_argument(
        "--vector",
        action="store_true",
        help="Output as feature vector instead of dictionary",
    )
    parser.add_argument(
        "--bc-output", help="Output path for generated bitcode (if input is C source)"
    )
    parser.add_argument(
        "--target-arch",
        choices=["riscv64", "riscv32", "native"],
        default="riscv64",
        help="Target architecture for C compilation (default: riscv64)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    # Determine input type and extract features
    if input_path.suffix == ".c":
        features = extract_features_from_c_source(
            str(input_path), args.bc_output, args.target_arch
        )
    elif input_path.suffix in [".ll", ".bc"]:
        extractor = LLVMFeatureExtractor()
        features = extractor.extract_from_file(str(input_path))
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    # Format output
    if args.vector:
        extractor = LLVMFeatureExtractor()
        output = {
            "feature_names": extractor.get_feature_names(features),
            "feature_vector": extractor.get_feature_vector(features),
            "program": input_path.stem,
        }
    else:
        output = {
            "program": input_path.stem,
            "features": features,
            "feature_count": len(features),
        }

    output_str = json.dumps(output, indent=2)

    # Write or print
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str)
        print(f"Extracted {len(features)} features to {args.output}")
    else:
        print(output_str)


if __name__ == "__main__":
    main()
