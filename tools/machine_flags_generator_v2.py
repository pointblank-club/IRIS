#!/usr/bin/env python3
"""
Machine-Level RISC-V Optimization Flags Generator v2
Supports different ABIs for more extension variation
"""

import random
import json
import argparse
from typing import List, Dict, Set, Any, Tuple


class MachineFlagsGeneratorV2:
    """Generate RISC-V machine-level optimization flags with ABI control."""

    # RISC-V ABIs and their required extensions
    RISC_V_ABIS = {
        "ilp32": {
            "required": ["m", "a", "c"],  # No float
            "optional": ["f", "d", "v"],
            "description": "32-bit integer only",
        },
        "ilp32f": {
            "required": ["m", "a", "f", "c"],  # Single float
            "optional": ["d", "v"],
            "description": "32-bit with single-precision float",
        },
        "ilp32d": {
            "required": ["m", "a", "f", "d", "c"],  # Double float
            "optional": ["v"],
            "description": "32-bit with double-precision float",
        },
        "lp64": {
            "required": ["m", "a", "c"],  # No float
            "optional": ["f", "d", "v"],
            "description": "64-bit integer only",
        },
        "lp64f": {
            "required": ["m", "a", "f", "c"],  # Single float
            "optional": ["d", "v"],
            "description": "64-bit with single-precision float",
        },
        "lp64d": {
            "required": ["m", "a", "f", "d", "c"],  # Double float (default)
            "optional": ["v"],
            "description": "64-bit with double-precision float (default)",
        },
    }

    # Additional RISC-V extensions and optimizations
    MACHINE_FLAGS = {
        "optimization_level": {
            "O": ["0", "1", "2", "3"],
        },
        # Bit manipulation
        "bitmanip": {
            "zba": [True, False],  # Address generation
            "zbb": [True, False],  # Basic bit manipulation
            "zbc": [True, False],  # Carry-less multiplication
            "zbs": [True, False],  # Single-bit operations
        },
        # Code size
        "codesize": {
            "zcb": [True, False],  # Additional compressed
            "zcmp": [True, False],  # Push/pop (RISCVPushPopOptimization)
            "zcmt": [True, False],  # Table jump
        },
        # Performance
        "performance": {
            "lui-addi-fusion": [True, False],
            "auipc-addi-fusion": [True, False],
            "ld-add-fusion": [True, False],
            "conditional-cmv-fusion": [True, False],
            "fast-unaligned-access": [True, False],
        },
        # Memory
        "memory": {
            "relax": [True, False],
            "no-optimized-zero-stride-load": [True, False],
        },
        # Codegen
        "codegen": {
            "code-model": ["small", "medium"],
            "relocation-model": ["static", "pic"],
        },
    }

    def __init__(self, seed: int = None, target_arch: str = "riscv64"):
        """
        Initialize generator.

        Args:
            seed: Random seed
            target_arch: riscv32 or riscv64
        """
        if seed is not None:
            random.seed(seed)
        self.target_arch = target_arch

        # Default ABI for target
        if target_arch == "riscv64":
            self.default_abi = "lp64d"
        else:
            self.default_abi = "ilp32d"

    def generate_with_abi(self, abi: str = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate configuration for specific ABI.

        Args:
            abi: ABI to use (if None, uses default)

        Returns:
            Tuple of (abi_name, config_dict)
        """
        if abi is None:
            abi = self.default_abi

        abi_info = self.RISC_V_ABIS[abi]
        config = {}

        # Add required extensions for this ABI
        for ext in abi_info["required"]:
            config[ext] = True

        # Randomly enable optional extensions
        for ext in abi_info["optional"]:
            config[ext] = random.choice([True, False])

        # Add random other flags
        all_flags = {}
        for category, flags in self.MACHINE_FLAGS.items():
            all_flags.update(flags)

        num_additional = random.randint(3, 8)
        selected = random.sample(
            list(all_flags.keys()), min(num_additional, len(all_flags))
        )

        for flag in selected:
            config[flag] = random.choice(all_flags[flag])

        return abi, config

    def generate_random_config(self) -> Dict[str, Any]:
        """Generate random config with default ABI."""
        _, config = self.generate_with_abi(self.default_abi)
        return config

    def generate_with_varied_abi(self) -> Tuple[str, Dict[str, Any]]:
        """Generate config with randomly selected ABI for more variation."""
        # Filter ABIs by target architecture
        if self.target_arch == "riscv64":
            valid_abis = ["lp64", "lp64f", "lp64d"]
        else:
            valid_abis = ["ilp32", "ilp32f", "ilp32d"]

        abi = random.choice(valid_abis)
        return self.generate_with_abi(abi)

    def config_to_llc_flags(self, config: Dict[str, Any], abi: str = None) -> List[str]:
        """Convert config to llc flags."""
        flags = []
        mattr_features = []

        # Add ABI flag if specified
        if abi:
            flags.append(f"-mabi={abi}")

        for flag, value in config.items():
            if flag == "O":
                flags.append(f"-O={value}")
            elif flag in ["code-model", "relocation-model"]:
                flags.append(f"-{flag}={value}")
            else:
                if isinstance(value, bool):
                    if value:
                        mattr_features.append(f"+{flag}")
                    else:
                        mattr_features.append(f"-{flag}")
                else:
                    mattr_features.append(f"+{flag}")

        if mattr_features:
            flags.append(f"-mattr={','.join(mattr_features)}")

        return flags

    def generate_multiple(
        self, count: int, vary_abi: bool = False
    ) -> List[Dict[str, Any]]:
        """Generate multiple configurations."""
        results = []

        for _ in range(count):
            if vary_abi:
                abi, config = self.generate_with_varied_abi()
                results.append(
                    {
                        "abi": abi,
                        "config": config,
                        "description": self.RISC_V_ABIS[abi]["description"],
                    }
                )
            else:
                config = self.generate_random_config()
                results.append({"abi": self.default_abi, "config": config})

        return results


def main():
    parser = argparse.ArgumentParser(description="Generate RISC-V machine flags v2")
    parser.add_argument("-n", "--count", type=int, default=5)
    parser.add_argument(
        "--vary-abi", action="store_true", help="Vary ABI for more extension variation"
    )
    parser.add_argument("--target", choices=["riscv32", "riscv64"], default="riscv64")
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    gen = MachineFlagsGeneratorV2(seed=args.seed, target_arch=args.target)
    results = gen.generate_multiple(args.count, vary_abi=args.vary_abi)

    output = {
        "count": len(results),
        "target": args.target,
        "vary_abi": args.vary_abi,
        "configurations": [],
    }

    for i, result in enumerate(results):
        config_data = {
            "id": i,
            "abi": result["abi"],
            "config": result["config"],
        }
        if "description" in result:
            config_data["description"] = result["description"]

        config_data["llc_flags"] = gen.config_to_llc_flags(
            result["config"], result["abi"]
        )
        output["configurations"].append(config_data)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
