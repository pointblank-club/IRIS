# Contributing to IRis

Thank you for your interest in contributing to IRis! This document provides guidelines and instructions for contributing to this ML-powered compiler optimization project for RISC-V.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please be considerate in your interactions with other contributors.

## Getting Started

Before contributing, please:

1. Read the [README.md](README.md) to understand the project's purpose and architecture
2. Familiarize yourself with the ML pipeline: feature extraction, pass sequence generation, model training, and inference
3. Ensure you have the required dependencies installed

## Development Setup

### Prerequisites

**System Requirements:**
- LLVM/Clang 18+ with RISC-V support
- RISC-V cross-compiler (`riscv64-linux-gnu-gcc`)
- QEMU RISC-V emulator
- Python 3.8+
- Node.js 18+ (for frontend)
- Git LFS (for large model files)

### Linux Setup

```bash
# Clone the repository
git clone https://github.com/your-username/IRis.git
cd IRis

# Install System Dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y clang llvm qemu-user-static gcc-riscv64-linux-gnu

# Install Python dependencies
pip install -r tools/requirements.txt
pip install black bandit flake8 pytest

# Verify toolchain installation
cd tools && chmod +x test_tools.sh && ./test_tools.sh
```

## CI/CD & Local Testing

We use a comprehensive CI pipeline to ensure code quality. Please run these checks locally before submitting a PR.

### 1. Code Formatting (Black)
We use `black` to ensure consistent code style.
```bash
# Check for formatting issues
black --check .

# Automatically format code
black .
```

### 2. Security Checks (Bandit)
We use `bandit` to scan for common security vulnerabilities.
```bash
# Run security scan
bandit -r . -ll
```

### 3. Linting (Flake8)
We use `flake8` to catch syntax errors and undefined names.
```bash
# Run linting
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

### 4. Integration Tests
Run the core tools to ensure everything is working:

```bash
# Test Feature Extractor
python tools/test_feature_extractor.py

# Test Pass Sequence Generator
python tools/pass_sequence_generator.py -n 5 -s mixed

# Smoke Test Training Data Generation (Native)
python tools/generate_training_data_hybrid.py \
    --programs-dir training_programs \
    --output-dir test_output \
    --num-sequences 2 \
    --strategy random \
    --no-qemu \
    --target-arch native
```

### Backend Setup (Flask API)

```bash
cd iris-website/backend
pip install -r requirements.txt
python app.py  # Starts on http://localhost:5001
```

### Frontend Setup (Next.js)

```bash
cd iris-website
npm install
npm run dev  # Starts on http://localhost:3000
```

## Project Structure

```
IRis/
├── iris.py                    # Core Transformer model & dataset
├── train_iris_ranker.py       # Ranking model training
├── train_passformer_hybrid.py # Hybrid optimization training
├── inference.py               # Model inference
├── tools/                     # ML pipeline utilities
│   ├── feature_extractor.py   # LLVM IR feature extraction
│   ├── pass_sequence_generator.py
│   └── generate_training_data.py
├── training_programs/         # 176+ C programs for training
├── models/                    # Trained model checkpoints
└── iris-website/              # Full-stack web application
    ├── src/                   # Next.js frontend
    └── backend/               # Flask REST API
```

## How to Contribute

### Types of Contributions

We welcome contributions in the following areas:

1. **ML Pipeline Improvements**
   - New feature extraction methods
   - Model architecture enhancements
   - Training data generation improvements

2. **Training Programs**
   - Additional C programs for training diversity
   - Programs covering edge cases or specific optimization patterns

3. **Frontend/Backend Development**
   - UI/UX improvements
   - API endpoint enhancements
   - Performance optimizations

4. **Documentation**
   - Code documentation
   - Tutorial improvements
   - Usage examples

5. **Bug Fixes**
   - Compilation issues
   - Model inference bugs
   - API errors

### Contribution Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Test your changes thoroughly
5. Commit with clear, descriptive messages
6. Push to your fork
7. Open a Pull Request

## Coding Standards

### Python

- Follow PEP 8 style guidelines
- Use type hints where applicable
- Include docstrings for functions and classes
- Keep functions focused and modular

```python
def extract_features(ir_file: str, verbose: bool = False) -> dict:
    """
    Extract LLVM IR features from the given file.

    Args:
        ir_file: Path to the LLVM IR file
        verbose: Enable detailed logging

    Returns:
        Dictionary containing extracted features
    """
    # Implementation
```

### TypeScript/React

- Use TypeScript for type safety
- Follow React best practices and hooks patterns
- Use Tailwind CSS for styling
- Keep components small and reusable

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add new feature extraction method for loop analysis
fix: Resolve compilation error with nested functions
docs: Update API documentation for optimize endpoint
refactor: Simplify pass sequence generation logic
```

## Testing Guidelines

### Running Tests

**Feature Extraction:**
```bash
python tools/feature_extractor.py path/to/program.c --verbose
```

**Backend API:**
```bash
cd iris-website/backend
python test_api.py
python test_transformer_integration.py
```

**Toolchain Verification:**
```bash
cd tools
./test_tools.sh  # Linux
python test_tools.py  # Windows
```

### Writing Tests

- Test new features thoroughly before submitting
- Include both positive and negative test cases
- Verify compilation success with RISC-V toolchain
- Test model predictions against baseline optimizations

### Performance Validation

When modifying the ML pipeline:
- Compare results against standard optimization levels (-O0, -O1, -O2, -O3)
- Document performance improvements or regressions
- Test on diverse program types

## Pull Request Process

1. **Before Submitting:**
   - Ensure all tests pass
   - Update documentation if needed
   - Verify no regressions in existing functionality

2. **PR Description:**
   - Clearly describe the changes
   - Reference any related issues
   - Include test results if applicable

3. **Review Process:**
   - Address reviewer feedback promptly
   - Keep discussions constructive
   - Be open to suggestions

4. **Merging:**
   - PRs require review approval before merging
   - Squash commits if requested
   - Ensure CI passes (if configured)

## Reporting Bugs

When reporting bugs, please include:

1. **Environment Information:**
   - Operating system and version
   - Python version
   - LLVM/Clang version
   - QEMU version

2. **Steps to Reproduce:**
   - Minimal code example if applicable
   - Command sequence that triggers the bug

3. **Expected vs Actual Behavior:**
   - What you expected to happen
   - What actually happened

4. **Error Messages:**
   - Full error output/stack traces
   - Relevant log files

## Feature Requests

When proposing new features:

1. **Describe the Problem:**
   - What limitation are you facing?
   - How does this affect your workflow?

2. **Propose a Solution:**
   - How should the feature work?
   - Are there alternative approaches?

3. **Consider Impact:**
   - How does this fit with existing architecture?
   - What are potential side effects?

## Questions?

If you have questions about contributing:
- Open an issue with the `question` label
- Review existing documentation
- Check closed issues for similar questions

---

Thank you for contributing to IRis! Your efforts help advance ML-powered compiler optimization for the RISC-V ecosystem.
