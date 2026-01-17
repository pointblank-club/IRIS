#!/usr/bin/env python3
"""
Hybrid Training Dataset Generator
Generates ML training data using both IR passes and machine-level flags.
"""

import os
import json
import time
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from hybrid_sequence_generator import HybridSequenceGenerator
from feature_extractor import LLVMFeatureExtractor


class HybridTrainingDataGenerator:
    """Generate training data with both IR and machine-level optimizations."""

    def __init__(
        self,
        programs_dir: str,
        output_dir: str,
        num_sequences: int = 200,
        target_arch: str = "riscv64",
        use_qemu: bool = True,
    ):
        """
        Initialize the hybrid training data generator.

        Args:
            programs_dir: Directory containing training programs (.c files)
            output_dir: Directory to store generated training data
            num_sequences: Number of hybrid sequences to generate per program
            target_arch: Target architecture (riscv64, riscv32, or native)
            use_qemu: Use QEMU emulation for cross-compiled binaries
        """
        self.programs_dir = Path(programs_dir)
        self.output_dir = Path(output_dir)
        self.num_sequences = num_sequences
        self.target_arch = target_arch
        self.use_qemu = use_qemu

        # Set target-specific flags
        if target_arch == "riscv64":
            self.target_triple = "riscv64-unknown-linux-gnu"
            self.qemu_binary = "qemu-riscv64"
            self.march = "riscv64"
        elif target_arch == "riscv32":
            self.target_triple = "riscv32-unknown-linux-gnu"
            self.qemu_binary = "qemu-riscv32"
            self.march = "riscv32"
        else:
            self.target_triple = None
            self.qemu_binary = None
            self.march = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.hybrid_generator = HybridSequenceGenerator()
        self.feature_extractor = LLVMFeatureExtractor()

    def find_programs(self) -> List[Path]:
        """Find all C programs in the programs directory."""
        return sorted(self.programs_dir.glob("*.c"))

    def compile_to_bitcode(self, c_file: Path, optimization: str = "-O0") -> Path:
        """Compile C source to LLVM bitcode."""
        bc_file = self.output_dir / f"{c_file.stem}.bc"

        clang_cmd = ["clang"]
        if self.target_triple:
            clang_cmd.extend(["--target=" + self.target_triple])
        clang_cmd.extend(
            [optimization, "-emit-llvm", "-c", str(c_file), "-o", str(bc_file)]
        )

        try:
            subprocess.run(clang_cmd, check=True, capture_output=True, timeout=30)
            return bc_file
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"Failed to compile {c_file}")

    def apply_hybrid_optimization(
        self, bc_file: Path, hybrid_sequence: Dict[str, Any]
    ) -> Tuple[Path, Path]:
        """
        Apply hybrid optimization (IR passes + machine flags).

        Args:
            bc_file: Input bitcode file
            hybrid_sequence: Dictionary with ir_passes and machine_config

        Returns:
            Tuple of (optimized_bitcode, assembly_file)
        """
        opt_bc_file = bc_file.with_suffix(".opt.bc")
        asm_file = bc_file.with_suffix(".s")

        # Step 1: Apply IR passes with opt
        ir_passes = hybrid_sequence["ir_passes"]
        if ir_passes:
            pass_arg = f"-passes={','.join(ir_passes)}"
        else:
            pass_arg = "-passes=default<O0>"

        try:
            subprocess.run(
                ["opt", pass_arg, str(bc_file), "-o", str(opt_bc_file)],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None, None

        # Step 2: Apply machine-level opts with llc
        if self.march:
            machine_config = hybrid_sequence["machine_config"]

            # Build llc command
            llc_cmd = ["llc", f"-march={self.march}"]

            # Use MachineFlags Generator to convert config to proper llc flags
            from machine_flags_generator import MachineFlagsGenerator

            flag_generator = MachineFlagsGenerator()
            llc_flags = flag_generator.config_to_llc_flags(machine_config)
            llc_cmd.extend(llc_flags)

            llc_cmd.extend([str(opt_bc_file), "-o", str(asm_file)])

            try:
                subprocess.run(llc_cmd, check=True, capture_output=True, timeout=30)
                return opt_bc_file, asm_file
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return None, None

        return opt_bc_file, None

    def compile_to_executable(self, asm_file: Path) -> Path:
        """Compile assembly to executable."""
        exe_file = asm_file.with_suffix(".exe")

        if self.target_arch in ["riscv64", "riscv32"]:
            gcc_cmd = f"{self.target_arch}-linux-gnu-gcc"
            try:
                subprocess.run(
                    [gcc_cmd, str(asm_file), "-o", str(exe_file), "-static"],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                return exe_file
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return None
        else:
            # Native: compile bitcode directly
            try:
                subprocess.run(
                    [
                        "clang",
                        str(asm_file.with_suffix(".opt.bc")),
                        "-o",
                        str(exe_file),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                return exe_file
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return None

    def measure_performance(
        self, exe_file: Path, num_runs: int = 1
    ) -> Dict[str, float]:
        """Measure execution performance."""
        times = []

        if self.use_qemu and self.qemu_binary:
            exec_cmd = [self.qemu_binary, str(exe_file)]
        else:
            exec_cmd = [str(exe_file)]

        for _ in range(num_runs):
            try:
                start = time.perf_counter()
                subprocess.run(exec_cmd, check=True, capture_output=True, timeout=10)
                end = time.perf_counter()
                times.append(end - start)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return None

        binary_size = exe_file.stat().st_size

        return {
            "execution_time": sum(times) / len(times),
            "binary_size": binary_size,
            "num_runs": num_runs,
        }

    def process_single_program(
        self, c_file: Path, num_sequences: int, strategy: str, verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process a single program with freshly generated hybrid sequences.

        Args:
            c_file: C source file
            num_sequences: Number of sequences to generate for this program
            strategy: Generation strategy
            verbose: Print progress

        Returns:
            List of training data points
        """
        program_name = c_file.stem
        data_points = []

        if verbose:
            print(f"\nProcessing {program_name}...")

        # Generate FRESH sequences for THIS program
        sequences = self.hybrid_generator.generate_multiple(
            count=num_sequences,
            strategy=strategy,
            include_presets=False,  # Never include presets (we want random only)
        )

        try:
            # Compile to unoptimized bitcode
            bc_file = self.compile_to_bitcode(c_file, optimization="-O0")

            # Extract baseline features
            baseline_features = self.feature_extractor.extract_from_file(str(bc_file))

            # Apply each hybrid sequence
            for seq_idx, sequence in enumerate(sequences):
                if verbose and seq_idx % 50 == 0:
                    print(f"  Sequence {seq_idx + 1}/{len(sequences)}...")

                try:
                    # Apply hybrid optimization
                    opt_bc_file, asm_file = self.apply_hybrid_optimization(
                        bc_file, sequence
                    )

                    if opt_bc_file is None or asm_file is None:
                        continue

                    # Compile to executable
                    exe_file = self.compile_to_executable(asm_file)

                    if exe_file is None:
                        continue

                    # Measure performance
                    metrics = self.measure_performance(exe_file)

                    if metrics is None:
                        continue

                    # Create data point
                    data_point = {
                        "program": program_name,
                        "sequence_id": seq_idx,
                        "features": baseline_features,
                        "ir_passes": sequence["ir_passes"],
                        "machine_config": sequence["machine_config"],
                        "ir_pass_count": len(sequence["ir_passes"]),
                        "machine_flag_count": len(sequence["machine_config"]),
                        "execution_time": metrics["execution_time"],
                        "binary_size": metrics["binary_size"],
                    }

                    data_points.append(data_point)

                    # Clean up
                    opt_bc_file.unlink(missing_ok=True)
                    asm_file.unlink(missing_ok=True)
                    exe_file.unlink(missing_ok=True)

                except Exception as e:
                    if verbose:
                        print(f"  Error processing sequence {seq_idx}: {e}")
                    continue

            # Clean up baseline bitcode
            bc_file.unlink(missing_ok=True)

        except Exception as e:
            if verbose:
                print(f"Error processing {program_name}: {e}")
            return []

        if verbose:
            print(f"  Generated {len(data_points)} valid data points")

        return data_points

    def generate_dataset(
        self,
        strategy: str = "mixed",
        parallel: bool = True,
        max_workers: int = 4,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Generate complete hybrid training dataset."""
        programs = self.find_programs()

        if not programs:
            raise ValueError(f"No C programs found in {self.programs_dir}")

        if verbose:
            print(f"Found {len(programs)} training programs")
            print(
                f"Will generate {self.num_sequences} FRESH hybrid sequences per program..."
            )
            print(
                f"Total sequences: {len(programs)} × {self.num_sequences} = {len(programs) * self.num_sequences}"
            )

        # Process programs
        all_data_points = []

        if parallel and len(programs) > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_single_program,
                        prog,
                        self.num_sequences,
                        strategy,
                        False,
                    ): prog
                    for prog in programs
                }

                if verbose:
                    pbar = tqdm(total=len(programs), desc="Processing programs")

                for future in as_completed(futures):
                    prog = futures[future]
                    try:
                        data_points = future.result()
                        all_data_points.extend(data_points)
                        if verbose:
                            pbar.update(1)
                            pbar.set_postfix({"points": len(all_data_points)})
                    except Exception as e:
                        if verbose:
                            print(f"\nError processing {prog.name}: {e}")

                if verbose:
                    pbar.close()
        else:
            for prog in programs:
                data_points = self.process_single_program(
                    prog, self.num_sequences, strategy, verbose
                )
                all_data_points.extend(data_points)

        # Create dataset
        dataset = {
            "metadata": {
                "num_programs": len(programs),
                "num_sequences": self.num_sequences,
                "strategy": strategy,
                "total_data_points": len(all_data_points),
                "optimization_type": "hybrid (IR + machine)",
                "programs": [p.name for p in programs],
            },
            "data": all_data_points,
        }

        if verbose:
            print(f"\n✓ Generated {len(all_data_points)} training data points")
            print(f"  Average per program: {len(all_data_points) / len(programs):.1f}")

        return dataset

    def save_dataset(
        self, dataset: Dict[str, Any], filename: str = "training_data_hybrid.json"
    ):
        """Save dataset to JSON file."""
        output_file = self.output_dir / filename

        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=2)

        print(f"✓ Saved dataset to {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate hybrid training dataset (IR + machine optimizations)"
    )
    parser.add_argument(
        "--programs-dir", required=True, help="Training programs directory"
    )
    parser.add_argument(
        "--output-dir", default="./training_data", help="Output directory"
    )
    parser.add_argument(
        "-n", "--num-sequences", type=int, default=200, help="Sequences per program"
    )
    parser.add_argument(
        "-s",
        "--strategy",
        choices=["random", "balanced", "mixed", "all"],
        default="mixed",
        help="Generation strategy",
    )
    parser.add_argument(
        "--no-presets",
        action="store_true",
        help="Disable O1/O2/O3 presets (use only random sequences)",
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel processing"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Max parallel workers"
    )
    parser.add_argument(
        "--output-file", default="training_data_hybrid.json", help="Output filename"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument(
        "--target-arch",
        choices=["riscv64", "riscv32", "native"],
        default="riscv64",
        help="Target architecture",
    )
    parser.add_argument("--no-qemu", action="store_true", help="Disable QEMU emulation")

    args = parser.parse_args()

    # Create generator
    generator = HybridTrainingDataGenerator(
        programs_dir=args.programs_dir,
        output_dir=args.output_dir,
        num_sequences=args.num_sequences,
        target_arch=args.target_arch,
        use_qemu=not args.no_qemu,
    )

    print("=" * 60)
    print("Hybrid Training Data Generator (IR + Machine Optimizations)")
    print("=" * 60)
    print(f"Target Architecture: {args.target_arch}")
    print(f"Optimization Levels: IR passes + Machine flags")
    print("=" * 60)

    # Generate dataset
    dataset = generator.generate_dataset(
        strategy=args.strategy,
        parallel=not args.no_parallel,
        max_workers=args.max_workers,
        verbose=not args.quiet,
    )

    # Save dataset
    output_file = generator.save_dataset(dataset, filename=args.output_file)

    # Print summary
    print("\n" + "=" * 60)
    print("Hybrid Dataset Generation Complete!")
    print("=" * 60)
    print(f"Programs: {dataset['metadata']['num_programs']}")
    print(f"Sequences per program: {dataset['metadata']['num_sequences']}")
    print(f"Total data points: {dataset['metadata']['total_data_points']}")
    print(f"Output: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
