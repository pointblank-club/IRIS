#!/usr/bin/env python3
"""
Training Dataset Generator
Generates ML training data by applying pass sequences to programs and measuring performance.
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

from pass_sequence_generator import PassSequenceGenerator, format_sequence_for_opt
from feature_extractor import LLVMFeatureExtractor, extract_features_from_c_source


class TrainingDataGenerator:
    """Generate training data for ML-guided compiler optimization."""

    def __init__(
        self,
        programs_dir: str,
        output_dir: str,
        num_sequences: int = 200,
        target_arch: str = "riscv64",
        use_qemu: bool = True,
    ):
        """
        Initialize the training data generator.

        Args:
            programs_dir: Directory containing training programs (.c files)
            output_dir: Directory to store generated training data
            num_sequences: Number of pass sequences to generate per program
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
        elif target_arch == "riscv32":
            self.target_triple = "riscv32-unknown-linux-gnu"
            self.qemu_binary = "qemu-riscv32"
        else:
            self.target_triple = None  # Native compilation
            self.qemu_binary = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.pass_generator = PassSequenceGenerator()
        self.feature_extractor = LLVMFeatureExtractor()

    def find_programs(self) -> List[Path]:
        """Find all C programs in the programs directory."""
        return sorted(self.programs_dir.glob("*.c"))

    def compile_to_bitcode(self, c_file: Path, optimization: str = "-O0") -> Path:
        """
        Compile C source to LLVM bitcode.

        Args:
            c_file: Path to C source file
            optimization: Optimization level (default: -O0 for unoptimized)

        Returns:
            Path to generated bitcode file
        """
        bc_file = self.output_dir / f"{c_file.stem}.bc"

        # Build clang command with target flags
        clang_cmd = ["clang"]
        if self.target_triple:
            clang_cmd.extend(["--target=" + self.target_triple])
        clang_cmd.extend(
            [optimization, "-emit-llvm", "-c", str(c_file), "-o", str(bc_file)]
        )

        try:
            subprocess.run(clang_cmd, check=True, capture_output=True, timeout=30)
            return bc_file
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to compile {c_file}: {e.stderr.decode()}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Compilation timeout for {c_file}")

    def apply_pass_sequence(self, bc_file: Path, sequence: List[str]) -> Path:
        """
        Apply a pass sequence to bitcode.

        Args:
            bc_file: Input bitcode file
            sequence: List of LLVM passes to apply

        Returns:
            Path to optimized bitcode file
        """
        opt_bc_file = bc_file.with_suffix(".opt.bc")

        # Format passes for opt command (LLVM 18+ new pass manager)
        # Join passes with commas for -passes flag
        if sequence:
            pass_arg = f"-passes={','.join(sequence)}"
        else:
            pass_arg = "-passes=default<O0>"

        try:
            subprocess.run(
                ["opt", pass_arg, str(bc_file), "-o", str(opt_bc_file)],
                check=True,
                capture_output=True,
                timeout=60,
            )
            return opt_bc_file
        except subprocess.CalledProcessError as e:
            # Some pass sequences might be invalid or fail
            # Return None to indicate failure
            return None
        except subprocess.TimeoutExpired:
            return None

    def compile_to_executable(self, bc_file: Path) -> Path:
        """
        Compile bitcode to native executable.

        Args:
            bc_file: LLVM bitcode file

        Returns:
            Path to executable
        """
        exe_file = bc_file.with_suffix(".exe")

        # For RISC-V, use 3-step process: bitcode -> assembly -> executable
        if self.target_arch in ["riscv64", "riscv32"]:
            asm_file = bc_file.with_suffix(".s")

            # Step 1: Use llc to compile bitcode to RISC-V assembly
            march = "riscv64" if self.target_arch == "riscv64" else "riscv32"
            try:
                subprocess.run(
                    [
                        "llc",
                        f"-march={march}",
                        "-mattr=+m,+a,+f,+d,+c",
                        str(bc_file),
                        "-o",
                        str(asm_file),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return None

            # Step 2: Use RISC-V GCC to link assembly to executable
            gcc_cmd = f"{self.target_arch}-linux-gnu-gcc"
            try:
                subprocess.run(
                    [gcc_cmd, str(asm_file), "-o", str(exe_file), "-static"],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                # Clean up assembly file
                asm_file.unlink(missing_ok=True)
                return exe_file
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                asm_file.unlink(missing_ok=True)
                return None
        else:
            # Native compilation: use clang directly
            try:
                subprocess.run(
                    ["clang", str(bc_file), "-o", str(exe_file)],
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
        """
        Measure execution performance of a program.

        Args:
            exe_file: Path to executable
            num_runs: Number of runs to average

        Returns:
            Dictionary with performance metrics
        """
        times = []

        # Prepare execution command (with QEMU if cross-compiling)
        if self.use_qemu and self.qemu_binary:
            exec_cmd = [self.qemu_binary, str(exe_file)]
        else:
            exec_cmd = [str(exe_file)]

        for _ in range(num_runs):
            try:
                start = time.perf_counter()
                result = subprocess.run(
                    exec_cmd, check=True, capture_output=True, timeout=10
                )
                end = time.perf_counter()
                times.append(end - start)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                # If execution fails, return None
                return None

        # Get binary size
        binary_size = exe_file.stat().st_size

        return {
            "execution_time": sum(times) / len(times),
            # 'min_time': min(times),
            # 'max_time': max(times),
            # 'std_time': self._std(times),
            "binary_size": binary_size,
            "num_runs": num_runs,
        }

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    def process_single_program(
        self, c_file: Path, sequences: List[List[str]], verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process a single program with multiple pass sequences.

        Args:
            c_file: C source file
            sequences: List of pass sequences to apply
            verbose: Print progress information

        Returns:
            List of training data points
        """
        program_name = c_file.stem
        data_points = []

        if verbose:
            print(f"\nProcessing {program_name}...")

        try:
            # Step 1: Compile to unoptimized bitcode
            bc_file = self.compile_to_bitcode(c_file, optimization="-O0")

            # Step 2: Extract baseline features
            baseline_features = self.feature_extractor.extract_from_file(str(bc_file))

            # Step 3: Apply each pass sequence
            for seq_idx, sequence in enumerate(sequences):
                if verbose and seq_idx % 50 == 0:
                    print(f"  Sequence {seq_idx + 1}/{len(sequences)}...")

                try:
                    # Apply optimization passes
                    opt_bc_file = self.apply_pass_sequence(bc_file, sequence)

                    if opt_bc_file is None:
                        continue  # Skip failed optimizations

                    # Compile to executable
                    exe_file = self.compile_to_executable(opt_bc_file)

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
                        "pass_sequence": sequence,
                        "sequence_length": len(sequence),
                        "execution_time": metrics["execution_time"],
                        "binary_size": metrics["binary_size"],
                    }

                    data_points.append(data_point)

                    # Clean up temporary files
                    opt_bc_file.unlink(missing_ok=True)
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
        """
        Generate complete training dataset.

        Args:
            strategy: Pass sequence generation strategy
            parallel: Use parallel processing
            max_workers: Maximum number of parallel workers
            verbose: Print progress information

        Returns:
            Dictionary containing all training data
        """
        # Find all programs
        programs = self.find_programs()

        if not programs:
            raise ValueError(f"No C programs found in {self.programs_dir}")

        if verbose:
            print(f"Found {len(programs)} training programs")
            print(f"Generating {self.num_sequences} sequences per program...")

        # Generate pass sequences
        sequences = self.pass_generator.generate_multiple(
            count=self.num_sequences, strategy=strategy
        )
        sequences = self.pass_generator.deduplicate_sequences(sequences)

        if verbose:
            print(f"Generated {len(sequences)} unique pass sequences")

        # Process each program
        all_data_points = []

        if parallel and len(programs) > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_single_program, prog, sequences, False
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
            # Sequential processing
            for prog in programs:
                data_points = self.process_single_program(prog, sequences, verbose)
                all_data_points.extend(data_points)

        # Create dataset
        dataset = {
            "metadata": {
                "num_programs": len(programs),
                "num_sequences": self.num_sequences,
                "strategy": strategy,
                "total_data_points": len(all_data_points),
                "programs": [p.name for p in programs],
            },
            "data": all_data_points,
        }

        if verbose:
            print(f"\n✓ Generated {len(all_data_points)} training data points")
            print(f"  Average per program: {len(all_data_points) / len(programs):.1f}")

        return dataset

    def save_dataset(
        self, dataset: Dict[str, Any], filename: str = "training_data.json"
    ):
        """Save dataset to JSON file."""
        output_file = self.output_dir / filename

        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=2)

        print(f"✓ Saved dataset to {output_file}")
        return output_file

    def get_optimization_pass_sequence(self, opt_level: str) -> List[str]:
        """
        Extract the actual LLVM pass sequence used by a given optimization level.

        Args:
            opt_level: Optimization level (-O0, -O1, -O2, -O3)

        Returns:
            List of pass names in the sequence
        """
        # Map optimization levels to their pass pipeline names
        opt_map = {
            "-O0": "default<O0>",
            "-O1": "default<O1>",
            "-O2": "default<O2>",
            "-O3": "default<O3>",
        }

        if opt_level not in opt_map:
            return []

        try:
            # Use opt with --debug-pass-manager to see the actual passes being run
            # Create a minimal test bitcode to query the pass sequence
            with tempfile.NamedTemporaryFile(
                suffix=".ll", mode="w", delete=False
            ) as tmp:
                # Minimal LLVM IR
                tmp.write(
                    """
; ModuleID = 'test'
source_filename = "test"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

define i32 @main() {
entry:
  ret i32 0
}
"""
                )
                tmp.flush()
                tmp_path = tmp.name

            # Compile to bitcode first
            bc_path = tmp_path.replace(".ll", ".bc")
            subprocess.run(
                ["llvm-as", tmp_path, "-o", bc_path],
                check=True,
                capture_output=True,
                timeout=5,
            )

            # Run opt with --debug-pass-manager to capture the passes
            result = subprocess.run(
                [
                    "opt",
                    f"-passes={opt_map[opt_level]}",
                    "--debug-pass-manager",
                    bc_path,
                    "-o",
                    "/dev/null",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            # Clean up temp files
            Path(tmp_path).unlink(missing_ok=True)
            Path(bc_path).unlink(missing_ok=True)

            # Parse the debug output to extract pass names
            if result.stderr:
                # Extract pass names from debug output
                passes = self._parse_debug_pass_output(result.stderr)
                if passes:
                    return passes

            # Fallback: return the optimization level name
            return [opt_map[opt_level]]

        except Exception as e:
            # If we can't determine the passes, return the optimization level as fallback
            return [opt_map[opt_level]]

    def _parse_debug_pass_output(self, debug_output: str) -> List[str]:
        """
        Parse LLVM debug pass manager output to extract pass names.

        Args:
            debug_output: Debug output from opt --debug-pass-manager

        Returns:
            List of pass names
        """
        import re

        passes = []

        # Look for lines like: "Running pass: PassName on ..."
        # or "Running analysis: AnalysisName on ..."
        for line in debug_output.split("\n"):
            if "Running pass:" in line:
                # Extract pass name
                match = re.search(r"Running pass: ([^\s]+)", line)
                if match:
                    pass_name = match.group(1)
                    # Filter out common wrapper passes to get the actual optimization passes
                    if pass_name not in ["VerifierPass", "PrintModulePass"]:
                        passes.append(pass_name)

        # If we found passes, return them; otherwise return empty
        return passes if passes else []

    def _parse_pass_pipeline(self, pipeline_str: str) -> List[str]:
        """
        Parse LLVM pass pipeline string into a list of pass names.

        Args:
            pipeline_str: Pass pipeline string from opt

        Returns:
            List of pass names
        """
        # Remove 'Passes:' prefix if present
        if "Passes:" in pipeline_str:
            pipeline_str = pipeline_str.split("Passes:")[-1].strip()

        # For now, return the whole pipeline as a single string
        # since LLVM's pass pipeline can be complex with nested passes
        # This preserves the full structure
        if pipeline_str:
            return [pipeline_str]

        return []

    def generate_baseline_comparisons(self, programs: List[Path]) -> Dict[str, Any]:
        """
        Generate baseline performance for standard optimization levels.

        Args:
            programs: List of program paths

        Returns:
            Dictionary with baseline results
        """
        baselines = {}

        for program in programs:
            program_baselines = {}

            for opt_level in ["-O0", "-O1", "-O2", "-O3"]:
                try:
                    # Compile with optimization level
                    bc_file = self.compile_to_bitcode(program, optimization=opt_level)
                    exe_file = self.compile_to_executable(bc_file)

                    if exe_file:
                        metrics = self.measure_performance(exe_file)
                        if metrics:
                            program_baselines[opt_level] = metrics

                        exe_file.unlink(missing_ok=True)

                    bc_file.unlink(missing_ok=True)

                except Exception as e:
                    print(
                        f"Error generating baseline for {program.name} {opt_level}: {e}"
                    )

            baselines[program.stem] = program_baselines

        return baselines

    def generate_enhanced_baselines(
        self, programs: List[Path] = None, verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Generate enhanced baseline performance with sequences and features.
        This wraps around the baseline function and adds:
        - sequence: The actual LLVM pass sequence used by the optimization level
        - features: Extracted LLVM IR features

        Args:
            programs: List of program paths (if None, finds all programs)
            verbose: Print progress information

        Returns:
            Dictionary with enhanced baseline results including sequences and features
        """
        if programs is None:
            programs = self.find_programs()

        if verbose:
            print(f"Generating enhanced baselines for {len(programs)} programs...")

        enhanced_baselines = {}

        for program in programs:
            if verbose:
                print(f"  Processing {program.stem}...")

            program_baselines = {}

            for opt_level in ["-O0", "-O1", "-O2", "-O3"]:
                try:
                    # Get the actual pass sequence for this optimization level
                    pass_sequence = self.get_optimization_pass_sequence(opt_level)

                    # Compile with optimization level
                    bc_file = self.compile_to_bitcode(program, optimization=opt_level)

                    # Extract features from the compiled bitcode
                    features = self.feature_extractor.extract_from_file(str(bc_file))

                    # Compile to executable
                    exe_file = self.compile_to_executable(bc_file)

                    if exe_file:
                        # Measure performance
                        metrics = self.measure_performance(exe_file)

                        if metrics:
                            # Create enhanced baseline entry with actual pass sequence
                            program_baselines[opt_level] = {
                                "execution_time": metrics["execution_time"],
                                "binary_size": metrics["binary_size"],
                                "sequence": pass_sequence,  # Actual LLVM pass sequence
                                "features": features,
                            }

                        exe_file.unlink(missing_ok=True)

                    bc_file.unlink(missing_ok=True)

                except Exception as e:
                    if verbose:
                        print(f"    Error with {opt_level}: {e}")
                    continue

            enhanced_baselines[program.stem] = program_baselines

            if verbose:
                print(f"    ✓ Generated {len(program_baselines)} baseline levels")

        return enhanced_baselines


def main():
    parser = argparse.ArgumentParser(
        description="Generate training dataset for ML-guided compiler optimization"
    )
    parser.add_argument(
        "--programs-dir",
        required=True,
        help="Directory containing training programs (.c files)",
    )
    parser.add_argument(
        "--output-dir",
        default="./training_data",
        help="Output directory for generated data (default: ./training_data)",
    )
    parser.add_argument(
        "-n",
        "--num-sequences",
        type=int,
        default=200,
        help="Number of pass sequences per program (default: 200)",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        choices=["random", "stratified", "synergy", "mixed", "all"],
        default="mixed",
        help="Pass sequence generation strategy (default: mixed)",
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel processing"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--output-file",
        default="training_data.json",
        help="Output filename (default: training_data.json)",
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Also generate baseline comparisons for -O0/-O1/-O2/-O3",
    )
    parser.add_argument(
        "--enhanced-baselines",
        action="store_true",
        help="Generate enhanced baselines with sequences and features for -O0/-O1/-O2/-O3",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Only generate baselines (skip training data generation)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument(
        "--target-arch",
        choices=["riscv64", "riscv32", "native"],
        default="riscv64",
        help="Target architecture (default: riscv64 for RISC-V 64-bit)",
    )
    parser.add_argument(
        "--no-qemu",
        action="store_true",
        help="Disable QEMU emulation (use for native RISC-V hardware)",
    )

    args = parser.parse_args()

    # Create generator
    generator = TrainingDataGenerator(
        programs_dir=args.programs_dir,
        output_dir=args.output_dir,
        num_sequences=args.num_sequences,
        target_arch=args.target_arch,
        use_qemu=not args.no_qemu,
    )

    print("=" * 60)
    print("ML-Guided Compiler Optimization - Training Data Generator")
    print("=" * 60)
    print(f"Target Architecture: {args.target_arch}")
    if args.target_arch in ["riscv64", "riscv32"]:
        print(f"QEMU Emulation: {'Enabled' if not args.no_qemu else 'Disabled'}")
    print("=" * 60)

    # Generate dataset (unless baselines-only mode)
    dataset = None
    output_file = None

    if not args.baselines_only:
        dataset = generator.generate_dataset(
            strategy=args.strategy,
            parallel=not args.no_parallel,
            max_workers=args.max_workers,
            verbose=not args.quiet,
        )

        # Save dataset
        output_file = generator.save_dataset(dataset, filename=args.output_file)

    # Generate baselines if requested
    if args.baselines:
        print("\nGenerating baseline comparisons...")
        programs = generator.find_programs()
        baselines = generator.generate_baseline_comparisons(programs)

        baseline_file = generator.output_dir / "baselines.json"
        with open(baseline_file, "w") as f:
            json.dump(baselines, f, indent=2)
        print(f"✓ Saved baselines to {baseline_file}")

    # Generate enhanced baselines if requested
    if args.enhanced_baselines:
        print("\nGenerating enhanced baselines with sequences and features...")
        programs = generator.find_programs()
        enhanced_baselines = generator.generate_enhanced_baselines(
            programs, verbose=not args.quiet
        )

        enhanced_baseline_file = generator.output_dir / "baselines_enhanced.json"
        with open(enhanced_baseline_file, "w") as f:
            json.dump(enhanced_baselines, f, indent=2)
        print(f"✓ Saved enhanced baselines to {enhanced_baseline_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    if dataset:
        print(f"Programs: {dataset['metadata']['num_programs']}")
        print(f"Sequences per program: {dataset['metadata']['num_sequences']}")
        print(f"Total data points: {dataset['metadata']['total_data_points']}")
        print(f"Output: {output_file}")
    if args.baselines or args.enhanced_baselines:
        programs = generator.find_programs()
        print(f"Baseline files generated for {len(programs)} programs")
    print("=" * 60)


if __name__ == "__main__":
    main()
