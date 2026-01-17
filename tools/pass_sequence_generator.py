#!/usr/bin/env python3
"""
Pass Sequence Generator for LLVM Optimization
Generates diverse, valid LLVM pass sequences for compiler optimization experiments.
"""

import random
import json
import argparse
from typing import List, Set
from itertools import combinations


class PassSequenceGenerator:
    """Generate diverse LLVM optimization pass sequences."""

    # Core LLVM optimization passes organized by category
    PASSES = {
        "mem": [
            "mem2reg",  # Promote memory to register
            "memcpyopt",  # Optimize memcpy calls
            "sroa",  # Scalar replacement of aggregates
        ],
        "cfg": [
            "simplifycfg",  # Simplify control flow graph
            "mergereturn",  # Merge multiple returns
            "lowerswitch",  # Lower switch instructions
            "break-crit-edges",  # Break critical edges
        ],
        "loop": [
            "loop-simplify",  # Canonicalize loops
            "loop-rotate",  # Rotate loops
            "loop-unroll",  # Unroll loops
            "loop-unroll-and-jam",  # Unroll and jam loops
            "licm",  # Loop invariant code motion
            "loop-deletion",  # Delete dead loops
            "loop-reduce",  # Loop strength reduction
            "loop-vectorize",  # Vectorize loops
            "indvars",  # Canonicalize induction variables
        ],
        "scalar": [
            "gvn",  # Global value numbering
            "sccp",  # Sparse conditional constant propagation
            "ipsccp",  # Interprocedural SCCP
            "dce",  # Dead code elimination
            "adce",  # Aggressive DCE
            "dse",  # Dead store elimination
            "early-cse",  # Early common subexpression elimination
            "reassociate",  # Reassociate expressions
            "instcombine",  # Combine instructions (includes constant propagation)
            "instsimplify",  # Simplify instructions
            "jump-threading",  # Thread jumps through blocks
            "correlated-propagation",  # Propagate correlated values
            "tailcallelim",  # Tail call elimination
            # Note: constprop removed - merged into instcombine in LLVM 18+
        ],
        "inline": [
            "inline",  # Function inlining
            "always-inline",  # Always inline functions marked as such
            "partial-inliner",  # Partial function inlining
        ],
        "interprocedural": [
            "globalopt",  # Global variable optimization
            "globaldce",  # Dead global elimination
            "argpromotion",  # Promote by-reference arguments
            "deadargelim",  # Dead argument elimination
            "function-attrs",  # Deduce function attributes (LLVM 18+ uses hyphen)
            # Note: ipconstprop removed - use ipsccp instead (already in scalar)
        ],
        "vectorize": [
            "slp-vectorizer",  # Superword-level parallelism vectorizer
        ],
        # RISC-V Hardware-Specific Passes (IR-level only)
        # Note: Many RISC-V passes are machine-level and run during llc, not opt
        "riscv_optimization": [
            "consthoist",  # Constant hoisting (RISC-V aware)
            "lower-constant-intrinsics",  # Lower constants for RISC-V
            "div-rem-pairs",  # Optimize div/rem pairs for RISC-V
        ],
        # Target-aware generic passes (tuned for RISC-V cost model)
        "target_aware": [
            "lower-expect",  # Lower expect intrinsics
            "strip-dead-prototypes",  # Remove unused function declarations
            "elim-avail-extern",  # Eliminate available externally
            "lower-matrix-intrinsics",  # Lower matrix operations
            "annotation-remarks",  # Add optimization remarks
        ],
    }

    # All passes flattened
    ALL_PASSES = [p for passes in PASSES.values() for p in passes]

    # Common pass combinations that work well together
    PASS_SYNERGIES = [
        # Generic synergies
        ["mem2reg", "simplifycfg", "instcombine"],
        ["gvn", "dce", "simplifycfg"],
        ["loop-simplify", "licm", "loop-unroll"],
        ["sroa", "early-cse", "instcombine"],
        ["inline", "functionattrs", "argpromotion"],
        ["sccp", "dce", "simplifycfg"],
        ["loop-rotate", "licm", "indvars"],
        ["reassociate", "gvn", "dce"],
        # RISC-V-specific synergies (IR-level only)
        ["consthoist", "lower-constant-intrinsics", "sroa"],
        ["div-rem-pairs", "instcombine", "reassociate"],
        ["lower-constant-intrinsics", "early-cse", "instcombine"],
        # Hybrid synergies (RISC-V-aware + generic)
        ["mem2reg", "consthoist", "instcombine"],
        ["loop-vectorize", "lower-constant-intrinsics", "slp-vectorizer"],
        ["inline", "lower-expect", "function-attrs"],
        ["strip-dead-prototypes", "globaldce", "elim-avail-extern"],
    ]

    def __init__(self, seed: int = None):
        """Initialize the generator with an optional seed."""
        if seed is not None:
            random.seed(seed)

    def generate_random_sequence(
        self, min_length: int = 3, max_length: int = 15
    ) -> List[str]:
        """Generate a random pass sequence."""
        length = random.randint(min_length, max_length)
        return random.choices(self.ALL_PASSES, k=length)

    def generate_stratified_sequence(
        self, min_length: int = 5, max_length: int = 12
    ) -> List[str]:
        """
        Generate a stratified sequence that includes passes from different categories.
        This ensures diversity in the types of optimizations applied.
        """
        sequence = []
        categories = list(self.PASSES.keys())
        random.shuffle(categories)

        length = random.randint(min_length, max_length)

        # Distribute passes across categories
        for _ in range(length):
            category = random.choice(categories)
            pass_name = random.choice(self.PASSES[category])
            sequence.append(pass_name)

        return sequence

    def generate_synergy_based_sequence(
        self, num_synergies: int = 2, extra_passes: int = 3
    ) -> List[str]:
        """
        Generate a sequence based on known synergistic pass combinations.
        Combines multiple synergy groups with additional random passes.
        """
        sequence = []

        # Select random synergy groups
        selected_synergies = random.sample(
            self.PASS_SYNERGIES, min(num_synergies, len(self.PASS_SYNERGIES))
        )

        for synergy in selected_synergies:
            sequence.extend(synergy)

        # Add some random passes
        for _ in range(extra_passes):
            sequence.append(random.choice(self.ALL_PASSES))

        # Shuffle to mix things up
        random.shuffle(sequence)

        return sequence

    def generate_o1_inspired(self) -> List[str]:
        """Generate a sequence inspired by -O1 optimization level with RISC-V awareness."""
        return [
            "mem2reg",
            "simplifycfg",
            "sroa",
            "early-cse",
            "lower-constant-intrinsics",  # RISC-V: Lower constants early
            "instcombine",
            "dce",
            "simplifycfg",
        ]

    def generate_o2_inspired(self) -> List[str]:
        """Generate a sequence inspired by -O2 optimization level with RISC-V awareness."""
        return [
            "mem2reg",
            "sroa",
            "simplifycfg",
            "early-cse",
            "lower-constant-intrinsics",  # RISC-V: Lower constants
            "inline",
            "function-attrs",
            "argpromotion",
            "gvn",
            "sccp",
            "instcombine",
            "consthoist",  # RISC-V: Hoist constants
            "simplifycfg",
            "reassociate",
            "div-rem-pairs",  # RISC-V: Optimize div/rem
            "loop-simplify",
            "licm",
            "loop-rotate",
            "loop-unroll",
            "lower-expect",  # RISC-V: Lower expect intrinsics
            "dce",
            "dse",
            "simplifycfg",
        ]

    def generate_o3_inspired(self) -> List[str]:
        """Generate a sequence inspired by -O3 optimization level with RISC-V awareness."""
        sequence = self.generate_o2_inspired()
        # Add aggressive optimizations
        sequence.extend(
            [
                "loop-vectorize",
                "slp-vectorizer",
                "loop-unroll-and-jam",
                "gvn",
                "instcombine",
                "lower-matrix-intrinsics",  # RISC-V: Lower matrix operations
                "simplifycfg",
            ]
        )
        return sequence

    def generate_mutated_sequence(
        self, base_sequence: List[str], mutation_rate: float = 0.3
    ) -> List[str]:
        """
        Generate a mutated version of an existing sequence.
        Useful for genetic algorithm-style generation.
        """
        sequence = base_sequence.copy()

        for i in range(len(sequence)):
            if random.random() < mutation_rate:
                # Mutate this position
                mutation_type = random.choice(["replace", "insert", "delete"])

                if mutation_type == "replace":
                    sequence[i] = random.choice(self.ALL_PASSES)
                elif mutation_type == "insert" and len(sequence) < 20:
                    sequence.insert(i, random.choice(self.ALL_PASSES))
                elif mutation_type == "delete" and len(sequence) > 3:
                    del sequence[i]

        return sequence

    def generate_multiple(
        self,
        count: int,
        strategy: str = "mixed",
        min_length: int = 3,
        max_length: int = 15,
    ) -> List[List[str]]:
        """
        Generate multiple pass sequences using the specified strategy.

        Args:
            count: Number of sequences to generate
            strategy: Generation strategy - "random", "stratified", "synergy", "mixed", or "all"
            min_length: Minimum sequence length
            max_length: Maximum sequence length

        Returns:
            List of pass sequences
        """
        sequences = []

        if strategy == "mixed" or strategy == "all":
            # Distribute generation across strategies
            strategies = ["random", "stratified", "synergy"]

            # Add baseline sequences
            if strategy == "all":
                sequences.append(self.generate_o1_inspired())
                sequences.append(self.generate_o2_inspired())
                sequences.append(self.generate_o3_inspired())
                count -= 3

            for i in range(count):
                chosen_strategy = strategies[i % len(strategies)]

                if chosen_strategy == "random":
                    seq = self.generate_random_sequence(min_length, max_length)
                elif chosen_strategy == "stratified":
                    seq = self.generate_stratified_sequence(min_length, max_length)
                elif chosen_strategy == "synergy":
                    seq = self.generate_synergy_based_sequence()

                sequences.append(seq)
        else:
            # Use single strategy
            for _ in range(count):
                if strategy == "random":
                    seq = self.generate_random_sequence(min_length, max_length)
                elif strategy == "stratified":
                    seq = self.generate_stratified_sequence(min_length, max_length)
                elif strategy == "synergy":
                    seq = self.generate_synergy_based_sequence()
                elif strategy == "o1":
                    seq = self.generate_o1_inspired()
                elif strategy == "o2":
                    seq = self.generate_o2_inspired()
                elif strategy == "o3":
                    seq = self.generate_o3_inspired()
                else:
                    seq = self.generate_random_sequence(min_length, max_length)

                sequences.append(seq)

        return sequences

    def deduplicate_sequences(self, sequences: List[List[str]]) -> List[List[str]]:
        """Remove duplicate sequences."""
        seen = set()
        unique = []

        for seq in sequences:
            seq_tuple = tuple(seq)
            if seq_tuple not in seen:
                seen.add(seq_tuple)
                unique.append(seq)

        return unique


def format_sequence_for_opt(sequence: List[str]) -> str:
    """Format a sequence for use with LLVM opt command."""
    return " ".join(f"-{pass_name}" for pass_name in sequence)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLVM optimization pass sequences"
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=10,
        help="Number of sequences to generate (default: 10)",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        choices=["random", "stratified", "synergy", "mixed", "all", "o1", "o2", "o3"],
        default="mixed",
        help="Generation strategy (default: mixed)",
    )
    parser.add_argument(
        "--min-length", type=int, default=3, help="Minimum sequence length (default: 3)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=15,
        help="Maximum sequence length (default: 15)",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file (JSON format). If not specified, prints to stdout.",
    )
    parser.add_argument(
        "--format",
        choices=["list", "opt", "json"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--deduplicate", action="store_true", help="Remove duplicate sequences"
    )

    args = parser.parse_args()

    generator = PassSequenceGenerator(seed=args.seed)
    sequences = generator.generate_multiple(
        count=args.count,
        strategy=args.strategy,
        min_length=args.min_length,
        max_length=args.max_length,
    )

    if args.deduplicate:
        original_count = len(sequences)
        sequences = generator.deduplicate_sequences(sequences)
        print(f"# Deduplicated: {original_count} -> {len(sequences)}")

    # Format output
    if args.format == "json":
        output = {
            "count": len(sequences),
            "strategy": args.strategy,
            "seed": args.seed,
            "sequences": [
                {
                    "id": i,
                    "passes": seq,
                    "length": len(seq),
                    "opt_format": format_sequence_for_opt(seq),
                }
                for i, seq in enumerate(sequences)
            ],
        }
        output_str = json.dumps(output, indent=2)
    elif args.format == "opt":
        output_str = "\n".join(format_sequence_for_opt(seq) for seq in sequences)
    else:  # list format
        output_str = "\n".join(str(seq) for seq in sequences)

    # Write or print
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str)
        print(f"Generated {len(sequences)} sequences to {args.output}")
    else:
        print(output_str)


if __name__ == "__main__":
    main()
