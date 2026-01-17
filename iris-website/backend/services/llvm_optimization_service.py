#!/usr/bin/env python3
"""
Core LLVM Optimization Service - Handles feature extraction, ML pass application, and metrics comparison
"""

import subprocess
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from torch.serialization import add_safe_globals

# Add tools directory to path for feature extraction
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "tools"))
from feature_extractor import LLVMFeatureExtractor

# Add project root for model imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from train_passformer_seqgen import (
    PassGenTransformer,
    build_allowed_token_mask,
    MAX_PASS_SEQ_LEN,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class LLVMOptimizationService:
    """Core service for LLVM optimization operations with RISC-V target."""

    def __init__(
        self,
        target_arch: str = "riscv64",
        use_qemu: bool = True,
        target_metric: str = "execution_time",
    ):
        """
        Initialize LLVM optimization service for RISC-V.

        Args:
            target_arch: Target architecture (riscv64, riscv32)
            use_qemu: Use QEMU emulation for cross-compiled binaries
            target_metric: The target metric for the ML model (execution_time or binary_size)
        """
        self.target_arch = target_arch
        self.use_qemu = use_qemu
        self.target_metric = target_metric

        # Set RISC-V specific configurations
        if target_arch == "riscv64":
            self.target_triple = "riscv64-unknown-linux-gnu"
            self.qemu_binary = "qemu-riscv64"
            self.march = "riscv64"
            self.gcc_cmd = "riscv64-linux-gnu-gcc"
            self.abi = "lp64"
        elif target_arch == "riscv32":
            self.target_triple = "riscv32-unknown-linux-gnu"
            self.qemu_binary = "qemu-riscv32"
            self.march = "riscv32"
            self.gcc_cmd = "riscv32-linux-gnu-gcc"
            self.abi = "ilp32"
        else:
            raise ValueError(f"Unsupported target architecture: {target_arch}")

        self.feature_extractor = LLVMFeatureExtractor()

        # Initialize transformer model for pass generation
        self.transformer_model = None
        self.joint_pass_vocab = None
        self.hardware_vocab = None
        self.feature_scaler = None
        self.target_metric_scaler = None
        self.feature_keys = None
        self._load_transformer_model()

        logger.info(f"LLVMOptimizationService initialized for {target_arch}")
        logger.debug(f"Target triple: {self.target_triple}")
        logger.debug(f"GCC command: {self.gcc_cmd}")
        logger.debug(f"QEMU binary: {self.qemu_binary}")

    def extract_features_from_c(
        self, c_code: str
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Extract features from C source code using feature_extractor module.

        Args:
            c_code: C source code as string

        Returns:
            Tuple of (success, features_dict, error_message)
        """
        try:
            # Create temporary C file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
                f.write(c_code)
                c_file = Path(f.name)

            # Use the extract_features_from_c_source function directly
            from feature_extractor import extract_features_from_c_source

            # Extract features with proper RISC-V target
            features = extract_features_from_c_source(
                str(c_file),
                output_bc=None,  # Will use default .bc file
                target_arch=self.target_arch,
            )

            # Cleanup
            c_file.unlink(missing_ok=True)
            bc_file = c_file.with_suffix(".bc")
            bc_file.unlink(missing_ok=True)

            logger.info(f"Successfully extracted {len(features)} features")
            return True, features, None

        except RuntimeError as e:
            # From compilation failure
            error = str(e)
            logger.error(f"Feature extraction error: {error}")
            return False, None, error
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return False, None, str(e)

    def _load_transformer_model(self):
        """Load the trained transformer model and preprocessing artifacts."""
        try:
            project_root = Path(__file__).parent.parent.parent.parent
            preprocessing_dir = project_root / "preprocessing_output"
            models_dir = project_root / "models_seqgen"

            # Construct model path based on target_metric
            model_filename = f"passgen_transformer_{self.target_metric}.pth"
            model_path = models_dir / model_filename

            if not model_path.exists():
                logger.warning(
                    f"Transformer model for {self.target_metric} not found at {model_path}. Pass prediction will not be available."
                )
                self.transformer_model = None
                return

            # Load vocabularies and scalers
            with open(preprocessing_dir / "joint_pass_vocab.json", "r") as f:
                self.joint_pass_vocab = json.load(f)
            with open(preprocessing_dir / "hardware_vocab.json", "r") as f:
                self.hardware_vocab = json.load(f)
            with open(preprocessing_dir / "feature_keys.json", "r") as f:
                self.feature_keys = json.load(f)

            self.feature_scaler = joblib.load(preprocessing_dir / "feature_scaler.pkl")

            # Load target metric scaler specific to the target_metric
            target_metric_scaler_filename = (
                f"target_metric_scaler_{self.target_metric}.pkl"
            )
            target_metric_scaler_path = (
                preprocessing_dir / target_metric_scaler_filename
            )
            if not target_metric_scaler_path.exists():
                logger.warning(
                    f"Target metric scaler for {self.target_metric} not found at {target_metric_scaler_path}. Regression metrics will be unscaled."
                )
                self.target_metric_scaler = None
            else:
                self.target_metric_scaler = joblib.load(target_metric_scaler_path)

            # Load model checkpoint
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)  # nosec B614
            model_config = checkpoint["config"]

            # Store the target metric from the loaded model's config
            self.model_target_metric = model_config.get(
                "target_metric", self.target_metric
            )

            # Initialize model
            self.transformer_model = PassGenTransformer(
                vocab_size=model_config["vocab_size"],
                num_features=model_config["num_features"],
                hardware_vocab_size=model_config["hardware_vocab_size"],
                d_model=model_config["d_model"],
                nhead=model_config["nhead"],
                num_decoder_layers=model_config["num_decoder_layers"],
                dim_feedforward=model_config["dim_feedforward"],
                feature_mlp_layers=model_config["feature_mlp_layers"],
                max_seq_len=model_config["max_seq_len"],
                dropout=model_config.get("dropout", 0.1),
                context_tokens=model_config.get("context_tokens", 7),
            )

            # Load weights (use strict=False to handle architecture mismatches)
            missing_keys, unexpected_keys = self.transformer_model.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )

            if missing_keys:
                logger.warning(f"Model loading - missing keys: {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(
                    f"Model loading - unexpected keys: {unexpected_keys[:5]}..."
                )

            self.transformer_model.to(device)
            self.transformer_model.eval()

            logger.info(
                f"Transformer model for {self.model_target_metric} loaded successfully from {model_path}"
            )

        except Exception as e:
            logger.warning(f"Failed to load transformer model: {e}")
            self.transformer_model = None

    def _get_hardware_config_string(self, opt_level: str = "O_0") -> str:
        """Convert current hardware configuration to vocab string format."""
        # Map our target arch to hardware config string
        # Based on the hardware_vocab.json, we use simplified hardware configs
        base_config = opt_level

        # Add architecture features
        if self.target_arch == "riscv64":
            # Use a common RISC-V config from vocabulary
            # Default to basic config with common extensions
            return f"{base_config}_a_c_d_f"  # Atomic, Compressed, Double, Float
        else:
            # For riscv32, similar config
            return f"{base_config}_a_c_d_f"

    def predict_passes_with_transformer(
        self,
        features: Dict[str, Any],
        opt_level: str = "O_0",
        beam_size: int = 5,
        max_length: int = 60,
    ) -> Tuple[bool, Optional[List[str]], Optional[str]]:
        """
        Predict optimization passes using the transformer model.

        Args:
            features: Extracted program features
            opt_level: Optimization level hint (O_0, O_1, O_2, O_3)
            beam_size: Beam size for beam search
            max_length: Maximum sequence length

        Returns:
            Tuple of (success, pass_list, error_message)
        """
        if self.transformer_model is None:
            return False, None, "Transformer model not loaded"

        try:
            device = next(self.transformer_model.parameters()).device

            # Get hardware configuration string
            hw_config_str = self._get_hardware_config_string(opt_level)

            # Get hardware ID from vocabulary
            hardware_id = self.hardware_vocab.get(
                hw_config_str, self.hardware_vocab.get("<unk>", 0)
            )

            # Prepare features - ensure they match training feature order
            feature_vector = []
            for key in self.feature_keys:
                alt_key = key
                if key.startswith("feature_"):
                    alt_key = key[len("feature_") :]
                value = features.get(key)
                if value is None:
                    value = features.get(alt_key)
                if value is None:
                    value = 0.0
                feature_vector.append(float(value))

            logger.debug(
                "Feature vector summary: sum=%.3f first5=%s",
                float(np.sum(feature_vector)),
                feature_vector[:5],
            )

            # Scale features
            feature_array = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
            scaled_features = self.feature_scaler.transform(feature_array)[0]

            # Convert to tensors
            program_features = torch.tensor(
                scaled_features, dtype=torch.float32, device=device
            ).unsqueeze(0)
            hardware_ids = torch.tensor([hardware_id], dtype=torch.long, device=device)

            # Generate allowed token mask for hardware-aware generation
            allowed_mask = build_allowed_token_mask(
                hardware_ids, self.joint_pass_vocab, self.hardware_vocab, device
            )

            # Generate sequence using beam search
            with torch.no_grad():
                if beam_size > 1:
                    generated_sequence = self.transformer_model.generate_sequence_beam(
                        program_features,
                        hardware_ids,
                        self.joint_pass_vocab["<sos>"],
                        self.joint_pass_vocab["<eos>"],
                        self.joint_pass_vocab["<pad>"],
                        device,
                        max_length,
                        beam_size=beam_size,
                        allowed_token_mask=allowed_mask,
                    )
                else:
                    generated_sequence = (
                        self.transformer_model.generate_sequence_greedy(
                            program_features,
                            hardware_ids,
                            self.joint_pass_vocab["<sos>"],
                            self.joint_pass_vocab["<eos>"],
                            self.joint_pass_vocab["<pad>"],
                            device,
                            max_length,
                            allowed_token_mask=allowed_mask,
                        )
                    )

            # Convert token IDs to pass names
            id_to_pass = {v: k for k, v in self.joint_pass_vocab.items()}
            special_tokens = [
                self.joint_pass_vocab["<pad>"],
                self.joint_pass_vocab["<sos>"],
                self.joint_pass_vocab["<eos>"],
            ]

            generated_ids = generated_sequence.squeeze(0).cpu().tolist()
            logger.debug("Predicted token ids: %s", generated_ids[:15])

            # Filter out special tokens and hardware-specific suffixes
            pass_list = []
            for token_id in generated_ids:
                if token_id not in special_tokens:
                    pass_name = id_to_pass.get(token_id, "<unk>")
                    if pass_name != "<unk>":
                        # Skip hardware-specific tokens (contain ::)
                        if "::" in pass_name:
                            continue
                        # Skip machine-level passes (start with 'machine')
                        if pass_name.startswith("machine"):
                            continue
                        # Skip optimization level markers
                        if pass_name.lower().startswith(("o_0", "o_1", "o_2", "o_3")):
                            continue
                        # Only include valid LLVM IR passes
                        pass_list.append(pass_name)

            # Deduplicate while preserving order (some passes may appear multiple times)
            # But we want to keep the sequence intact for LLVM

            if not pass_list:
                # Fallback to some default passes if generation failed
                pass_list = ["mem2reg", "simplifycfg", "instcombine", "reassociate"]

            logger.info(f"Generated {len(pass_list)} passes using transformer model")
            logger.debug(f"Predicted passes: {pass_list}")

            return True, pass_list, None

        except Exception as e:
            logger.error(f"Error predicting passes with transformer: {e}")
            return False, None, str(e)

    def run_ml_passes(
        self,
        c_code: str,
        ir_passes: Optional[List[str]] = None,
        machine_config: Optional[Dict] = None,
        use_transformer: bool = True,
        opt_level_hint: str = "O_0",
        beam_size: int = 5,
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Apply ML-generated optimization passes and measure metrics.

        Args:
            c_code: C source code
            ir_passes: List of LLVM IR passes to apply (if None, will predict using transformer)
            machine_config: Optional machine-level optimization config
            use_transformer: Whether to use transformer for pass prediction if ir_passes is None
            opt_level_hint: Optimization level hint for transformer (O_0, O_1, O_2, O_3)

        Returns:
            Tuple of (success, metrics_dict, error_message)
        """
        temp_files = []

        # If no passes provided and transformer is available, predict them
        if ir_passes is None and use_transformer and self.transformer_model is not None:
            # Extract features first
            success, features, error = self.extract_features_from_c(c_code)
            if not success:
                return False, None, f"Feature extraction failed: {error}"

            # Predict passes
            success, predicted_passes, error = self.predict_passes_with_transformer(
                features, opt_level_hint, beam_size=beam_size
            )
            if success:
                ir_passes = predicted_passes
                logger.info(f"Using {len(ir_passes)} transformer-predicted passes")
            else:
                logger.warning(
                    f"Pass prediction failed: {error}. Using default passes."
                )
                ir_passes = [
                    "mem2reg",
                    "simplifycfg",
                    "instcombine",
                    "reassociate",
                    "gvn",
                    "dce",
                ]
        elif ir_passes is None:
            # Fallback to default passes if no transformer and no passes provided
            ir_passes = [
                "mem2reg",
                "simplifycfg",
                "instcombine",
                "reassociate",
                "gvn",
                "dce",
            ]
            logger.info("Using default optimization passes")

        try:
            # Create temporary C file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
                f.write(c_code)
                c_file = Path(f.name)
                temp_files.append(c_file)

            # Step 1: Compile to unoptimized bitcode
            bc_file = c_file.with_suffix(".bc")
            temp_files.append(bc_file)

            clang_cmd = [
                "clang",
                f"--target={self.target_triple}",
                "-O0",
                "-emit-llvm",
                "-c",
                str(c_file),
                "-o",
                str(bc_file),
            ]

            logger.debug(f"Compiling: {' '.join(clang_cmd)}")
            result = subprocess.run(clang_cmd, capture_output=True, timeout=30)

            if result.returncode != 0:
                error = (
                    result.stderr.decode() if result.stderr else "Compilation failed"
                )
                self._cleanup_files(temp_files)
                return False, None, error

            # Step 2: Apply IR passes
            opt_bc_file = bc_file.with_suffix(".opt.bc")
            temp_files.append(opt_bc_file)

            if ir_passes and len(ir_passes) > 0:
                pass_arg = f"-passes={','.join(ir_passes)}"
            else:
                pass_arg = "-passes=default<O0>"

            opt_cmd = ["opt", pass_arg, str(bc_file), "-o", str(opt_bc_file)]

            logger.debug(f"Applying passes: {' '.join(opt_cmd)}")
            opt_start = time.perf_counter()
            result = subprocess.run(opt_cmd, capture_output=True, timeout=60)
            opt_time = time.perf_counter() - opt_start

            if result.returncode != 0:
                error = (
                    result.stderr.decode() if result.stderr else "Optimization failed"
                )
                self._cleanup_files(temp_files)
                return False, None, error

            # Step 3: Generate assembly
            asm_file = opt_bc_file.with_suffix(".s")
            temp_files.append(asm_file)

            # Use consistent floating-point ABI
            llc_cmd = ["llc", f"-march={self.march}", "-mattr=+d,+f"]

            # Apply machine config if provided
            if machine_config:
                llc_flags = self._convert_machine_config_to_flags(machine_config)
                llc_cmd.extend(llc_flags)

            llc_cmd.extend([str(opt_bc_file), "-o", str(asm_file)])

            logger.debug(f"Generating assembly: {' '.join(llc_cmd)}")
            result = subprocess.run(llc_cmd, capture_output=True, timeout=30)

            if result.returncode != 0:
                error = (
                    result.stderr.decode()
                    if result.stderr
                    else "Assembly generation failed"
                )
                self._cleanup_files(temp_files)
                return False, None, error

            # Step 4: Compile to executable
            exe_file = asm_file.with_suffix(".exe")
            temp_files.append(exe_file)

            # Use lp64d ABI for double-precision floating point support
            gcc_cmd = [
                self.gcc_cmd,
                "-mabi=lp64d" if self.target_arch == "riscv64" else "-mabi=ilp32d",
                "-march=rv64gc" if self.target_arch == "riscv64" else "-march=rv32gc",
                str(asm_file),
                "-o",
                str(exe_file),
                "-static",
                "-lm",  # Link math library for sqrt and other math functions
            ]

            logger.debug(f"Compiling executable: {' '.join(gcc_cmd)}")
            compile_start = time.perf_counter()
            result = subprocess.run(gcc_cmd, capture_output=True, timeout=30)
            compile_time = time.perf_counter() - compile_start

            if result.returncode != 0:
                error = (
                    result.stderr.decode()
                    if result.stderr
                    else "Executable compilation failed"
                )
                self._cleanup_files(temp_files)
                return False, None, error

            # Step 5: Measure performance
            exec_cmd = (
                [self.qemu_binary, str(exe_file)] if self.use_qemu else [str(exe_file)]
            )

            times = []
            num_runs = 5
            for _ in range(num_runs):
                start = time.perf_counter()
                result = subprocess.run(exec_cmd, capture_output=True, timeout=10)
                if result.returncode != 0:
                    error = "Execution failed"
                    self._cleanup_files(temp_files)
                    return False, None, error
                times.append(time.perf_counter() - start)

            binary_size = exe_file.stat().st_size

            metrics = {
                "execution_time_avg": sum(times) / len(times),
                "execution_time_min": min(times),
                "execution_time_max": max(times),
                "binary_size": binary_size,
                "optimization_time": opt_time,
                "compile_time": compile_time,
                "num_runs": num_runs,
                "ir_passes": ir_passes,
                "pass_count": len(ir_passes),
                "machine_config": machine_config or {},
            }

            # Cleanup
            self._cleanup_files(temp_files)

            logger.info(
                f"ML passes applied successfully. Avg execution: {metrics['execution_time_avg']:.6f}s"
            )
            return True, metrics, None

        except Exception as e:
            self._cleanup_files(temp_files)
            logger.error(f"Error in run_ml_passes: {e}")
            return False, None, str(e)

    def run_standard_optimizations(
        self, c_code: str, opt_levels: List[str] = ["-O0", "-O1", "-O2", "-O3"]
    ) -> Dict[str, Dict]:
        """
        Run standard optimization levels for comparison.

        Args:
            c_code: C source code
            opt_levels: List of optimization levels to test

        Returns:
            Dictionary mapping opt_level to metrics
        """
        results = {}

        for opt_level in opt_levels:
            logger.info(f"Running standard optimization {opt_level}")

            temp_files = []

            try:
                # Create temporary C file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".c", delete=False
                ) as f:
                    f.write(c_code)
                    c_file = Path(f.name)
                    temp_files.append(c_file)

                # Compile with optimization level using RISC-V GCC
                exe_file = c_file.with_suffix(".exe")
                temp_files.append(exe_file)

                # Use RISC-V GCC directly for better compatibility
                gcc_cmd = [
                    self.gcc_cmd,
                    "-mabi=lp64d" if self.target_arch == "riscv64" else "-mabi=ilp32d",
                    (
                        "-march=rv64gc"
                        if self.target_arch == "riscv64"
                        else "-march=rv32gc"
                    ),
                    opt_level,
                    str(c_file),
                    "-o",
                    str(exe_file),
                    "-static",
                    "-lm",
                ]

                logger.debug(f"Compiling with {opt_level}: {' '.join(gcc_cmd)}")
                compile_start = time.perf_counter()
                result = subprocess.run(gcc_cmd, capture_output=True, timeout=30)
                compile_time = time.perf_counter() - compile_start

                if result.returncode != 0:
                    error = (
                        result.stderr.decode()
                        if result.stderr
                        else "Compilation failed"
                    )
                    results[opt_level] = {"success": False, "error": error}
                    self._cleanup_files(temp_files)
                    continue

                # Measure performance
                exec_cmd = (
                    [self.qemu_binary, str(exe_file)]
                    if self.use_qemu
                    else [str(exe_file)]
                )

                times = []
                num_runs = 5
                for _ in range(num_runs):
                    start = time.perf_counter()
                    result = subprocess.run(exec_cmd, capture_output=True, timeout=10)
                    if result.returncode != 0:
                        break
                    times.append(time.perf_counter() - start)

                if times:
                    binary_size = exe_file.stat().st_size

                    results[opt_level] = {
                        "success": True,
                        "execution_time_avg": sum(times) / len(times),
                        "execution_time_min": min(times),
                        "execution_time_max": max(times),
                        "binary_size": binary_size,
                        "compile_time": compile_time,
                        "num_runs": len(times),
                    }
                else:
                    results[opt_level] = {"success": False, "error": "Execution failed"}

                # Cleanup
                self._cleanup_files(temp_files)

            except Exception as e:
                self._cleanup_files(temp_files)
                results[opt_level] = {"success": False, "error": str(e)}

        logger.info(f"Standard optimizations complete. Tested {len(results)} levels")
        return results

    def compare_with_standard(
        self,
        c_code: str,
        ir_passes: Optional[List[str]] = None,
        machine_config: Optional[Dict] = None,
        use_transformer: bool = True,
        opt_level_hint: str = "O_0",
        beam_size: int = 5,
    ) -> Dict[str, Any]:
        """
        Run ML passes and compare with standard optimizations.

        Args:
            c_code: C source code
            ir_passes: ML-generated IR passes (if None, will use transformer)
            machine_config: Optional machine-level config
            use_transformer: Whether to use transformer for pass prediction
            opt_level_hint: Optimization level hint for transformer

        Returns:
            Comparison results dictionary
        """
        results = {
            "ml_optimization": None,
            "standard_optimizations": {},
            "comparison": {},
            "features": None,
        }

        # Extract features
        success, features, error = self.extract_features_from_c(c_code)
        if success:
            results["features"] = features

        # Run ML optimization
        success, ml_metrics, error = self.run_ml_passes(
            c_code,
            ir_passes,
            machine_config,
            use_transformer=use_transformer,
            opt_level_hint=opt_level_hint,
            beam_size=beam_size,
        )
        if success:
            results["ml_optimization"] = ml_metrics
        else:
            results["ml_optimization"] = {"success": False, "error": error}

        # Run standard optimizations
        results["standard_optimizations"] = self.run_standard_optimizations(c_code)

        # Compute comparison if ML optimization succeeded
        if ml_metrics:
            ml_exec_time = ml_metrics["execution_time_avg"]
            ml_binary_size = ml_metrics["binary_size"]

            comparisons = {}
            for opt_level, std_metrics in results["standard_optimizations"].items():
                if std_metrics.get("success"):
                    std_exec_time = std_metrics["execution_time_avg"]
                    std_binary_size = std_metrics["binary_size"]

                    comparisons[opt_level] = {
                        "speedup": (
                            std_exec_time / ml_exec_time if ml_exec_time > 0 else 0
                        ),
                        "size_reduction": (
                            1 - (ml_binary_size / std_binary_size)
                            if std_binary_size > 0
                            else 0
                        ),
                        "ml_faster": ml_exec_time < std_exec_time,
                        "ml_smaller": ml_binary_size < std_binary_size,
                    }

            results["comparison"] = comparisons

            # Find best standard optimization for performance
            best_std = None
            best_time = float("inf")
            for opt_level, metrics in results["standard_optimizations"].items():
                if metrics.get("success") and metrics["execution_time_avg"] < best_time:
                    best_time = metrics["execution_time_avg"]
                    best_std = opt_level

            # Find best standard optimization for size
            best_size_std = None
            best_size = float("inf")
            for opt_level, metrics in results["standard_optimizations"].items():
                if metrics.get("success") and metrics["binary_size"] < best_size:
                    best_size = metrics["binary_size"]
                    best_size_std = opt_level

            if best_std:
                results["comparison"]["vs_best"] = {
                    "best_standard": best_std,
                    "ml_beats_best": ml_exec_time < best_time,
                    "speedup_vs_best": (
                        best_time / ml_exec_time if ml_exec_time > 0 else 0
                    ),
                }

            # Add size comparison with best size optimization
            if best_size_std:
                results["comparison"]["vs_best_size"] = {
                    "best_size_standard": best_size_std,
                    "best_size_bytes": best_size,
                    "ml_size_bytes": ml_binary_size,
                    "ml_beats_best_size": ml_binary_size < best_size,
                    "size_reduction_vs_best": (
                        1 - (ml_binary_size / best_size) if best_size > 0 else 0
                    ),
                }

        return results

    def _convert_machine_config_to_flags(self, machine_config: Dict) -> List[str]:
        """
        Convert machine configuration to LLC flags.

        Args:
            machine_config: Dictionary with machine-level settings

        Returns:
            List of LLC command-line flags
        """
        flags = []

        # Map common machine config options to LLC flags
        if machine_config.get("fast_isel"):
            flags.append("-fast-isel")

        if machine_config.get("enable_machine_outliner"):
            flags.append("-enable-machine-outliner")

        if "mcpu" in machine_config:
            flags.append(f"-mcpu={machine_config['mcpu']}")

        if "mattr" in machine_config:
            attrs = machine_config["mattr"]
            if isinstance(attrs, list):
                attrs = ",".join(attrs)
            flags.append(f"-mattr={attrs}")

        return flags

    def _cleanup_files(self, files: List[Path]):
        """Clean up temporary files."""
        for file in files:
            if file and file.exists():
                file.unlink(missing_ok=True)
