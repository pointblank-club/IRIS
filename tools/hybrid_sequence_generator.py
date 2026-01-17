#!/usr/bin/env python3
"""
Hybrid RISC-V Optimization Sequence Generator
Combines IR-level passes and machine-level flags for comprehensive optimization.
"""

import random
import json
import argparse
from typing import List, Dict, Any, Tuple

from pass_sequence_generator import PassSequenceGenerator
from machine_flags_generator_v2 import MachineFlagsGeneratorV2 as MachineFlagsGenerator


class HybridSequenceGenerator:
    """Generate hybrid optimization sequences combining IR passes and machine flags."""

    def __init__(self, seed: int = None):
        """Initialize with optional seed."""
        self.pass_generator = PassSequenceGenerator(seed=seed)
        self.machine_generator = MachineFlagsGenerator(seed=seed)
        if seed is not None:
            random.seed(seed)

    def generate_hybrid_sequence(
        self,
        ir_strategy: str = "mixed",
        machine_strategy: str = "mixed",
        min_pass_length: int = 3,
        max_pass_length: int = 15,
    ) -> Dict[str, Any]:
        """
        Generate a single hybrid optimization sequence.

        Returns:
            Dictionary with ir_passes and machine_config
        """
        # Generate IR passes
        ir_passes = self.pass_generator.generate_multiple(
            count=1,
            strategy=ir_strategy,
            min_length=min_pass_length,
            max_length=max_pass_length,
        )[0]

        # Generate machine config
        machine_result = self.machine_generator.generate_multiple(
            count=1, vary_abi=False
        )[0]
        machine_config = machine_result["config"]
        machine_abi = machine_result["abi"]

        return {
            "ir_passes": ir_passes,
            "machine_config": machine_config,
            "machine_abi": machine_abi,
            "ir_pass_count": len(ir_passes),
            "machine_flag_count": len(machine_config),
        }

    def generate_balanced_sequence(self) -> Dict[str, Any]:
        """
        Generate a balanced sequence with coordinated IR and machine opts.
        Tries to match optimization strategies between IR and machine level.
        """
        # Choose optimization focus
        focus = random.choice(["performance", "size", "balanced"])

        if focus == "performance":
            # Aggressive IR passes + performance machine flags
            ir_passes = self.pass_generator.generate_o3_inspired()
            machine_config = {
                "riscv-enable-sink-fold": True,
                "riscv-enable-dead-defs-elim": True,
                "riscv-enable-loadstore-opt": True,
                "riscv-opt-w-instrs": True,
                "enable-misched": True,
            }
        elif focus == "size":
            # Conservative IR passes + size machine flags
            ir_passes = self.pass_generator.generate_o1_inspired()
            machine_config = {
                "riscv-enable-rvc": True,
                "riscv-enable-push-pop": True,
                "riscv-enable-move-merge": True,
            }
        else:  # balanced
            ir_passes = self.pass_generator.generate_o2_inspired()
            machine_config = {
                "riscv-enable-sink-fold": True,
                "riscv-fold-mem-offset": True,
                "riscv-enable-rvc": True,
                "riscv-opt-w-instrs": True,
            }

        return {
            "ir_passes": ir_passes,
            "machine_config": machine_config,
            "machine_abi": self.machine_generator.default_abi,
            "focus": focus,
            "ir_pass_count": len(ir_passes),
            "machine_flag_count": len(machine_config),
        }

    def generate_multiple(
        self, count: int, strategy: str = "mixed", include_presets: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple hybrid sequences.

        Args:
            count: Number of sequences to generate
            strategy: Generation strategy (random, balanced, mixed, all)
            include_presets: Include O1/O2/O3 preset combinations

        Returns:
            List of hybrid sequences
        """
        sequences = []

        # Add presets if requested
        if include_presets and (strategy == "all" or strategy == "mixed"):
            for opt_level in ["O1", "O2", "O3"]:
                if opt_level == "O1":
                    ir_passes = self.pass_generator.generate_o1_inspired()
                elif opt_level == "O2":
                    ir_passes = self.pass_generator.generate_o2_inspired()
                else:
                    ir_passes = self.pass_generator.generate_o3_inspired()

                machine_config = self.machine_generator.generate_preset_config(
                    opt_level
                )

                sequences.append(
                    {
                        "ir_passes": ir_passes,
                        "machine_config": machine_config,
                        "preset": opt_level,
                        "ir_pass_count": len(ir_passes),
                        "machine_flag_count": len(machine_config),
                    }
                )

            count -= 3

        # Generate remaining sequences
        if strategy == "mixed" or strategy == "all":
            strategies = ["random", "balanced"]
            for i in range(count):
                chosen = strategies[i % len(strategies)]
                if chosen == "random":
                    sequences.append(self.generate_hybrid_sequence())
                else:
                    sequences.append(self.generate_balanced_sequence())
        elif strategy == "random":
            for _ in range(count):
                sequences.append(self.generate_hybrid_sequence())
        elif strategy == "balanced":
            for _ in range(count):
                sequences.append(self.generate_balanced_sequence())

        return sequences

    def format_for_execution(self, sequence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format hybrid sequence for execution in training pipeline.

        Returns:
            Dictionary with opt_command and llc_flags
        """
        # Format IR passes for opt
        if sequence["ir_passes"]:
            opt_passes = ",".join(sequence["ir_passes"])
            opt_command = f"-passes={opt_passes}"
        else:
            opt_command = "-passes=default<O0>"

        # Format machine config for llc
        machine_abi = sequence.get("machine_abi", self.machine_generator.default_abi)
        llc_flags = self.machine_generator.config_to_llc_flags(
            sequence["machine_config"], machine_abi
        )

        return {
            "opt_command": opt_command,
            "llc_flags": llc_flags,
            "ir_passes": sequence["ir_passes"],
            "machine_config": sequence["machine_config"],
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate hybrid RISC-V optimization sequences (IR + machine)"
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
        choices=["random", "balanced", "mixed", "all"],
        default="mixed",
        help="Generation strategy (default: mixed)",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("-o", "--output", help="Output file (JSON format)")
    parser.add_argument(
        "--no-presets",
        action="store_true",
        help="Don't include O1/O2/O3 presets (use random sequences only)",
    )
    parser.add_argument(
        "--format-execution",
        action="store_true",
        help="Format output for execution (opt/llc commands)",
    )

    args = parser.parse_args()

    generator = HybridSequenceGenerator(seed=args.seed)
    sequences = generator.generate_multiple(
        count=args.count, strategy=args.strategy, include_presets=not args.no_presets
    )

    # Format for execution if requested
    if args.format_execution:
        formatted = []
        for i, seq in enumerate(sequences):
            exec_format = generator.format_for_execution(seq)
            exec_format["id"] = i
            formatted.append(exec_format)
        sequences = formatted

    # Create output
    output = {
        "count": len(sequences),
        "strategy": args.strategy,
        "seed": args.seed,
        "sequences": sequences,
    }

    output_str = json.dumps(output, indent=2)

    # Write or print
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str)
        print(f"Generated {len(sequences)} hybrid sequences to {args.output}")
    else:
        print(output_str)


if __name__ == "__main__":
    main()
