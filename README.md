# IRis — ML-Guided RISC-V Compiler Optimization

An ML-powered compiler optimization system that uses Transformers to predict optimal LLVM pass sequences for C programs, specifically targeting **RISC-V hardware**. Beat standard optimization levels (-O0/-O1/-O2/-O3) with intelligent, program-specific optimizations!

---

## Quick Start

### Linux Setup

#### Prerequisites
```bash
# 1. Install LLVM/Clang with RISC-V support (18+)
sudo apt install clang llvm llvm-tools

# Verify RISC-V support
llc --version | grep riscv

# 2. Install QEMU for RISC-V emulation
sudo apt install qemu-user qemu-user-static

# 3. Install RISC-V toolchain (recommended)
sudo apt install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu

# 4. Install Python dependencies
pip install xgboost scikit-learn pandas numpy tqdm
# Or use venv:
python3 -m venv venv
source venv/bin/activate
pip install xgboost scikit-learn pandas numpy tqdm
```

#### Run Tests
```bash
# Verify setup
cd tools
chmod +x test_tools.sh run_full_generation.sh
./test_tools.sh

# Test feature extraction
python3 feature_extractor.py ../training_programs/01_insertion_sort.c

# Test pass sequence generation
python3 pass_sequence_generator.py -n 5 -s mixed
```

#### Generate Training Data
```bash
# Quick test (10 sequences per program, ~10 min)
./run_full_generation.sh --test

# Full dataset (200 sequences per program, 4-10 hours)
./run_full_generation.sh

# Custom configuration
python3 generate_training_data.py \
    --programs-dir ../training_programs \
    --output-dir ./training_data \
    --num-sequences 200 \
    --strategy mixed \
    --max-workers 4 \
    --baselines
```

---

### Windows Setup

#### Prerequisites
1. **Install LLVM/Clang**
   - Download from https://releases.llvm.org/
   - Get version 18+ with RISC-V support
   - Add to PATH: `C:\Program Files\LLVM\bin`

2. **Install QEMU**
   - Download from https://www.qemu.org/download/#windows
   - Install to `C:\Program Files\qemu`
   - Add to PATH

3. **Install Python 3.8+**
   - Download from https://www.python.org/downloads/
   - Check "Add Python to PATH" during installation

4. **Install Python dependencies**
   ```cmd
   pip install xgboost scikit-learn pandas numpy tqdm
   ```

#### Run Tests
```cmd
cd tools

REM Verify setup
python test_tools.py

REM Test feature extraction
python feature_extractor.py ..\training_programs\01_insertion_sort.c

REM Test pass sequence generation
python pass_sequence_generator.py -n 5 -s mixed
```

#### Generate Training Data
```cmd
REM Quick test
python generate_training_data.py --programs-dir ..\training_programs --output-dir .\training_data -n 10 --strategy mixed

REM Full dataset
python generate_training_data.py --programs-dir ..\training_programs --output-dir .\training_data -n 200 --strategy mixed --max-workers 4 --baselines
```

---

## Project Overview

### What Does This Do?

This project uses **Machine Learning (XGBoost)** to learn which LLVM compiler optimization passes work best for different types of programs on **RISC-V architecture**. Instead of using one-size-fits-all optimization levels like `-O2` or `-O3`, it predicts custom pass sequences tailored to each program's characteristics.

### Workflow

1. **Training Data Generation**
   - Compile 30+ training programs to RISC-V
   - Try 200+ different pass sequences per program
   - Measure execution time and binary size
   - Extract ~50 features from LLVM IR

2. **Model Training**
   - Train XGBoost on program features → performance
   - Learn which passes work best for which program types
   - Model: `program_features → best_pass_sequence`

3. **Evaluation**
   - Test on 20 unseen programs
   - Compare ML predictions vs `-O0`/`-O1`/`-O2`/`-O3`
   - **Goal:** Beat `-O3` on >50% of test programs

---

## Project Structure

```
hackman/
├── README.md                    # This file
├── training_programs/           # 30+ programs for training (~176 programs)
│   ├── 01_insertion_sort.c
│   ├── 02_selection_sort.c
│   └── ...
├── test_programs/               # 20 programs for evaluation
│   ├── 01_quicksort.c
│   ├── 02_mergesort.c
│   └── ...
├── tools/                       # ML pipeline tools
│   ├── feature_extractor.py           # Extract IR features
│   ├── pass_sequence_generator.py     # Generate pass sequences
│   ├── hybrid_sequence_generator.py   # Hybrid pass + machine optimization
│   ├── machine_flags_generator_v2.py  # RISC-V machine-level flags (ABI support)
│   ├── generate_training_data.py      # Main data generation script
│   ├── train_passformer.py            # Train ML model
│   ├── combined_model.py              # Combined pass + machine optimization model
│   ├── test_tools.sh                  # Verify setup (Linux)
│   ├── run_full_generation.sh         # Convenience script (Linux)
│   └── training_data/                 # Generated datasets
│       ├── training_data_hybrid.json  # Hybrid pass + machine data
│       └── baselines.json             # -O0/-O1/-O2/-O3 results
├── combined_model.py            # Model training script
└── train_passformer.py          # Transformer-based model training
```

---

## 🔧 Tool Usage

### Feature Extractor
```bash
# Extract features from C program
python3 feature_extractor.py program.c -o features.json

# Specify RISC-V target
python3 feature_extractor.py program.c --target-arch riscv64

# Show all features
python3 feature_extractor.py program.c --verbose
```

### Pass Sequence Generator
```bash
# Generate random sequences
python3 pass_sequence_generator.py -n 10 -s random

# Mixed strategy (random + synergy-based)
python3 pass_sequence_generator.py -n 20 -s mixed

# Genetic algorithm
python3 pass_sequence_generator.py -n 50 -s genetic

# Custom length range
python3 pass_sequence_generator.py -n 10 --min-length 5 --max-length 15
```

### Hybrid Sequence Generator (Pass + Machine Optimization)
```bash
# Generate hybrid sequences with machine-level flags
python3 hybrid_sequence_generator.py -n 10 --strategy mixed

# Include machine flags
python3 hybrid_sequence_generator.py -n 20 --include-machine-flags
```

### Machine Flags Generator V2 (ABI Support)
```bash
# Generate machine-level configs with default ABI
python3 machine_flags_generator_v2.py -n 5

# Vary ABI for more diversity (lp64/lp64f/lp64d)
python3 machine_flags_generator_v2.py -n 10 --vary-abi

# For 32-bit RISC-V
python3 machine_flags_generator_v2.py -n 5 --target riscv32 --vary-abi
```

### Training Data Generation
```bash
# Full pipeline with all options
python3 generate_training_data.py \
    --programs-dir ../training_programs \
    --output-dir ./training_data \
    --num-sequences 200 \
    --strategy mixed \
    --target-arch riscv64 \
    --max-workers 4 \
    --baselines \
    --runs 3

# Quick test run
python3 generate_training_data.py \
    --programs-dir ../training_programs \
    --output-dir ./training_data \
    -n 10 \
    --no-parallel
```

---

## Output Data Format

### Training Data (`training_data_hybrid.json`)
```json
{
  "metadata": {
    "num_programs": 30,
    "num_sequences": 200,
    "strategy": "mixed",
    "total_data_points": 5123
  },
  "data": [
    {
      "program": "insertion_sort",
      "sequence_id": 0,
      "features": {
        "total_instructions": 87,
        "num_load": 23,
        "num_store": 15,
        "cyclomatic_complexity": 5,
        "memory_intensity": 0.437
      },
      "pass_sequence": ["mem2reg", "simplifycfg", "gvn"],
      "machine_config": {
        "abi": "lp64d",
        "config": {"m": true, "a": true, "f": true, "d": true, "c": true}
      },
      "execution_time": 0.0234,
      "binary_size": 8192
    }
  ]
}
```

### Baselines (`baselines.json`)
```json
{
  "insertion_sort": {
    "O0": {"time": 0.145, "size": 12288},
    "O1": {"time": 0.089, "size": 9216},
    "O2": {"time": 0.067, "size": 8704},
    "O3": {"time": 0.054, "size": 8192}
  }
}
```

---

## Training the Model

### Using Combined Model (Recommended)
```bash
# Train on hybrid data (pass sequences + machine flags)
python3 combined_model.py \
    --data tools/training_data/training_data_hybrid.json \
    --baselines tools/training_data/baselines.json \
    --output models/combined_model.pkl

# Evaluate model
python3 combined_model.py \
    --data tools/training_data/training_data_hybrid.json \
    --baselines tools/training_data/baselines.json \
    --evaluate
```

### Using Transformer Model
```bash
# Train PassFormer (Transformer-based sequence model)
python3 train_passformer.py \
    --data tools/training_data/training_data_hybrid.json \
    --epochs 50 \
    --batch-size 32 \
    --output models/passformer.pth
```

---

## Troubleshooting

### Linux

**Issue: "clang: unknown target triple 'riscv64'"**
```bash
# Verify RISC-V support
llc --version | grep riscv
# If missing, reinstall LLVM with RISC-V backend
```

**Issue: "qemu-riscv64: not found"**
```bash
sudo apt install qemu-user-static
which qemu-riscv64  # Should show /usr/bin/qemu-riscv64
```

**Issue: "error while loading shared libraries"**
```bash
# Install RISC-V sysroot
sudo apt install gcc-riscv64-linux-gnu
# Or run with explicit library path
qemu-riscv64 -L /usr/riscv64-linux-gnu ./program
```

### Windows

**Issue: "clang not recognized"**
- Add LLVM to PATH: `C:\Program Files\LLVM\bin`
- Restart terminal after adding to PATH

**Issue: "qemu-riscv64.exe not found"**
- Install QEMU for Windows
- Add to PATH: `C:\Program Files\qemu`

**Issue: Python package installation fails**
```cmd
REM Use --user flag
pip install --user xgboost scikit-learn pandas numpy tqdm

REM Or create virtual environment
python -m venv venv
venv\Scripts\activate
pip install xgboost scikit-learn pandas numpy tqdm
```

---

## 📈 Performance Expectations

### Training Data Generation
- **Small test** (10 sequences × 30 programs): ~10 minutes
- **Medium run** (50 sequences × 30 programs): ~1 hour
- **Full dataset** (200 sequences × 30 programs): 4-10 hours
- **Success rate**: ~85% (some sequences fail to compile)

### Expected Results
- **Data points**: 5,000-6,000 valid samples
- **File size**: 10-50 MB JSON
- **Model training**: 5-30 minutes
- **Evaluation**: 10-60 minutes on 20 test programs

---

## Success Metrics

| Metric | Target |
|--------|--------|
| **Beat -O3** | >50% of test programs |
| **Average speedup** | 5-10% faster than -O3 |
| **Generalization** | Works on unseen programs |

---

## Getting Help

1. **Check tools work:** `./test_tools.sh` (Linux) or `python test_tools.py` (Windows)
2. **Verify RISC-V support:** `llc --version | grep riscv`
3. **Test simple compilation:** 
   ```bash
   echo 'int main() { return 0; }' > test.c
   clang --target=riscv64-unknown-linux-gnu test.c -o test
   qemu-riscv64 test
   ```
4. **Use `--help` flag:** All tools support `--help` for detailed options

---

## Key Files

- **training_programs/**: Programs used to train the ML model
- **test_programs/**: Programs used to evaluate the model (unseen during training)
- **tools/training_data/**: Generated training datasets
- **combined_model.py**: Main model training script
- **train_passformer.py**: Transformer-based model training

---

## Resources

- LLVM Pass Documentation: https://llvm.org/docs/Passes.html
- RISC-V ISA: https://riscv.org/technical/specifications/
- QEMU User Mode: https://www.qemu.org/docs/master/user/main.html
- XGBoost: https://xgboost.readthedocs.io/

---

## Ready to Run!

**Linux:**
```bash
cd tools
./run_full_generation.sh
```

**Windows:**
```cmd
cd tools
python generate_training_data.py --programs-dir ..\training_programs --output-dir .\training_data -n 200 --strategy mixed
```

Good luck beating -O3! 🚀
