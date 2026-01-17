#!/usr/bin/env python3
"""
Simplified LLVM API Routes - Core functionality for ML optimization
"""

from flask import Blueprint, request, jsonify
from pathlib import Path
import tempfile
from typing import Dict, List, Any

from services.llvm_optimization_service import LLVMOptimizationService
from utils.logger import get_logger

logger = get_logger(__name__)

llvm_api = Blueprint("llvm_api", __name__, url_prefix="/api/llvm")

# Initialize service (singleton pattern)
_service_runtime = None
_service_binary_size = None


def get_service(
    target_arch: str = "riscv64", target_metric: str = "execution_time"
) -> LLVMOptimizationService:
    """Get or create the LLVM optimization service based on target metric and architecture."""
    global _service_runtime, _service_binary_size

    if target_metric == "execution_time":
        if _service_runtime is None or _service_runtime.target_arch != target_arch:
            _service_runtime = LLVMOptimizationService(
                target_arch=target_arch, target_metric="execution_time"
            )
        return _service_runtime
    elif target_metric == "binary_size":
        if (
            _service_binary_size is None
            or _service_binary_size.target_arch != target_arch
        ):
            _service_binary_size = LLVMOptimizationService(
                target_arch=target_arch, target_metric="binary_size"
            )
        return _service_binary_size
    else:
        raise ValueError(f"Unsupported target metric: {target_metric}")


@llvm_api.route("/features", methods=["POST"])
def extract_features():
    """
    Extract features from C source code.

    Request JSON:
    {
        "code": "C source code string",
        "target_arch": "riscv64" (optional)
    }

    Response JSON:
    {
        "success": true/false,
        "features": {...},
        "error": "error message if failed"
    }
    """
    try:
        data = request.get_json()

        if not data or "code" not in data:
            return jsonify({"success": False, "error": "No code provided"}), 400

        c_code = data["code"]
        target_arch = data.get("target_arch", "riscv64")

        service = get_service(target_arch)
        success, features, error = service.extract_features_from_c(c_code)

        if success:
            return jsonify(
                {
                    "success": True,
                    "features": features,
                    "feature_count": len(features) if features else 0,
                }
            )
        else:
            return jsonify({"success": False, "error": error}), 500

    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@llvm_api.route("/optimize", methods=["POST"])
def run_optimization():
    """
    Run ML-generated optimization passes on C code.

    Request JSON:
    {
        "code": "C source code string",
        "ir_passes": ["pass1", "pass2", ...] (optional, will use transformer if not provided),
        "machine_config": {...} (optional),
        "target_arch": "riscv64" (optional),
        "use_transformer": true/false (optional, default true),
        "opt_level_hint": "O_0"/"O_1"/"O_2"/"O_3" (optional, default "O_0")
    }

    Response JSON:
    {
        "success": true/false,
        "metrics": {
            "execution_time_avg": float,
            "binary_size": int,
            ...
        },
        "passes_used": ["pass1", "pass2", ...],
        "passes_source": "transformer"/"manual",
        "error": "error message if failed"
    }
    """
    try:
        data = request.get_json()

        if not data or "code" not in data:
            return jsonify({"success": False, "error": "No code provided"}), 400

        c_code = data["code"]
        ir_passes = data.get("ir_passes", None)  # Now optional
        machine_config = data.get("machine_config", None)
        target_arch = data.get("target_arch", "riscv64")
        use_transformer = data.get("use_transformer", True)
        opt_level_hint = data.get("opt_level_hint", "O_0")
        beam_size = data.get("beam_size", 5)  # Retrieve beam_size
        target_metric = data.get(
            "target_metric", "execution_time"
        )  # Retrieve target_metric

        # Validate passes if provided
        if ir_passes is not None and not isinstance(ir_passes, list):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "IR passes must be a list when provided",
                    }
                ),
                400,
            )

        service = get_service(
            target_arch, target_metric
        )  # Pass target_metric to get_service

        # Determine passes source for response
        passes_source = "manual" if ir_passes is not None else "transformer"

        success, metrics, error = service.run_ml_passes(
            c_code,
            ir_passes,
            machine_config,
            use_transformer=use_transformer,
            opt_level_hint=opt_level_hint,
            beam_size=beam_size,  # Pass beam_size to run_ml_passes
        )

        if success:
            response = {
                "success": True,
                "metrics": metrics,
                "passes_source": passes_source,
            }
            if metrics and "ir_passes" in metrics:
                response["passes_used"] = metrics["ir_passes"]
            return jsonify(response)
        else:
            return jsonify({"success": False, "error": error}), 500

    except Exception as e:
        logger.error(f"Optimization error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@llvm_api.route("/standard", methods=["POST"])
def run_standard_optimizations():
    """
    Run standard optimization levels on C code.

    Request JSON:
    {
        "code": "C source code string",
        "opt_levels": ["-O0", "-O1", "-O2", "-O3"] (optional),
        "target_arch": "riscv64" (optional)
    }

    Response JSON:
    {
        "success": true,
        "results": {
            "-O0": {...},
            "-O1": {...},
            ...
        }
    }
    """
    try:
        data = request.get_json()

        if not data or "code" not in data:
            return jsonify({"success": False, "error": "No code provided"}), 400

        c_code = data["code"]
        opt_levels = data.get("opt_levels", ["-O0", "-O1", "-O2", "-O3"])
        target_arch = data.get("target_arch", "riscv64")

        service = get_service(target_arch)
        results = service.run_standard_optimizations(c_code, opt_levels)

        return jsonify({"success": True, "results": results})

    except Exception as e:
        logger.error(f"Standard optimization error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@llvm_api.route("/compare", methods=["POST"])
def compare_optimizations():
    """
    Compare ML optimization with standard optimizations.

    Request JSON:
    {
        "code": "C source code string",
        "ir_passes": ["pass1", "pass2", ...] (optional, will use transformer if not provided),
        "machine_config": {...} (optional),
        "target_arch": "riscv64" (optional),
        "use_transformer": true/false (optional, default true),
        "opt_level_hint": "O_0"/"O_1"/"O_2"/"O_3" (optional, default "O_0")
    }

    Response JSON:
    {
        "success": true,
        "features": {...},
        "ml_optimization": {...},
        "standard_optimizations": {
            "-O0": {...},
            "-O1": {...},
            ...
        },
        "comparison": {
            "-O0": {
                "speedup": float,
                "size_reduction": float,
                "ml_faster": bool,
                "ml_smaller": bool
            },
            ...
        }
    }
    """
    try:
        data = request.get_json()

        if not data or "code" not in data:
            return jsonify({"success": False, "error": "No code provided"}), 400

        c_code = data["code"]
        ir_passes = data.get("ir_passes", None)  # Now optional
        machine_config = data.get("machine_config", None)
        target_arch = data.get("target_arch", "riscv64")
        use_transformer = data.get("use_transformer", True)
        opt_level_hint = data.get("opt_level_hint", "O_0")
        beam_size = data.get("beam_size", 5)  # Retrieve beam_size
        target_metric = data.get(
            "target_metric", "execution_time"
        )  # Retrieve target_metric

        service = get_service(
            target_arch, target_metric
        )  # Pass target_metric to get_service

        # Pass transformer parameters to compare_with_standard
        results = service.compare_with_standard(
            c_code,
            ir_passes,
            machine_config,
            use_transformer=use_transformer,
            opt_level_hint=opt_level_hint,
            beam_size=beam_size,  # Pass beam_size to compare_with_standard
        )

        return jsonify({"success": True, **results})

    except Exception as e:
        logger.error(f"Comparison error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@llvm_api.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.

    Response JSON:
    {
        "status": "healthy",
        "service": "llvm_optimization",
        "target_arch": "riscv64"
    }
    """
    try:
        service = get_service()
        return jsonify(
            {
                "status": "healthy",
                "service": "llvm_optimization",
                "target_arch": service.target_arch,
            }
        )
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
