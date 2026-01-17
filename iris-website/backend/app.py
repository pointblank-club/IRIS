#!/usr/bin/env python3
"""
Simplified IRIS Backend - Core LLVM Optimization API
"""

from flask import Flask, jsonify
from flask_cors import CORS
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Import routes
from routes.llvm_api import llvm_api

# Register blueprints
app.register_blueprint(llvm_api)


# Root endpoint
@app.route("/")
def index():
    """Root endpoint with API information."""
    return jsonify(
        {
            "service": "IRIS Backend - LLVM Optimization Service",
            "version": "2.0.0",
            "target": "RISC-V",
            "endpoints": {
                "features": "/api/llvm/features",
                "optimize": "/api/llvm/optimize",
                "standard": "/api/llvm/standard",
                "compare": "/api/llvm/compare",
                "health": "/api/llvm/health",
            },
            "description": "Core API for RISC-V LLVM optimization with ML-generated passes",
        }
    )


# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({"success": False, "error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting IRIS Backend Server (Simplified)")
    logger.info("Target Architecture: RISC-V")
    logger.info("API Documentation: http://localhost:5001/")

    app.run(host="0.0.0.0", port=5001, debug=True)
