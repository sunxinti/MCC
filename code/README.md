# MCC Code Directory

This directory contains all implementation code for the MCC (Model Confrontation & Collaboration) framework.

## File Overview

### Classical Benchmarks

| File Name | Description | Dataset Location |
|-----------|-------------|------------------|
| `MCC_MedQA.py` | MCC implementation for MedQA | `../benchmarks/MedQA/` |
| `MCC_PubmedQA.py` | MCC implementation for PubMedQA | `../benchmarks/PubMedQA/` |

#### MMLU Medical Subsets

| File Name | Description | Dataset Location |
|-----------|-------------|------------------|
| `MCC_MMLU_Anatomy.py` | MMLU Anatomy subset | `../benchmarks/MMLU_Anatomy/` |
| `MCC_MMLU_Clinical_knowledge.py` | MMLU Clinical Knowledge subset | `../benchmarks/MMLU_Clinical_knowledge/` |
| `MCC_MMLU_College_biology.py` | MMLU College Biology subset | `../benchmarks/MMLU_College_biology/` |
| `MCC_MMLU_College_medicine.py` | MMLU College Medicine subset | `../benchmarks/MMLU_College_medicine/` |
| `MCC_MMLU_Medical_genetics.py` | MMLU Medical Genetics subset | `../benchmarks/MMLU_Medical_genetics/` |
| `MCC_MMLU_Professional_medicine.py` | MMLU Professional Medicine subset | `../benchmarks/MMLU_Professional_medicine/` |

### Next-level Benchmarks: Advanced Reasoning, Metacognition, and Robustness

| File Name | Description | Dataset Location |
|-----------|-------------|------------------|
| `MCC_MedXpertQA.py` | MCC implementation for MedXpertQA | `../benchmarks/MedXpertQA/` |
| `MCC_MetamedQA.py` | MCC implementation for MetaMedQA | `../benchmarks/MetaMedQA/` |
| `MCC_for_RABBITS__original_MedQA.py` | RABBITS original version (MedQA) | `../benchmarks/RABBITS/original/` |
| `MCC_for_RABBITS__original_MedMCQA.py` | RABBITS original version (MedMCQA) | `../benchmarks/RABBITS/original/` |
| `MCC_for_RABBITS_generic_to_brand_MedQA.py` | RABBITS generic to brand (MedQA) | `../benchmarks/RABBITS/generic_to_brand/` |
| `MCC_for_RABBITS__generic_to_brand_MedMCQA.py` | RABBITS generic to brand (MedMCQA) | `../benchmarks/RABBITS/generic_to_brand/` |

### Long-form Questions

| File Name | Description | Dataset Location |
|-----------|-------------|------------------|
| `MCC_MultiMedQA_140.py` | MCC implementation for MultiMedQA 140 | `../benchmarks/MultiMedQA_140/` |
| `MCC_Henalthbench_function.py` | HealthBench evaluation functions | `../benchmarks/HealthBench/` |
| `MCC_run_HealthBench.py` | HealthBench main runner script | `../benchmarks/HealthBench/` |

### Diagnostic Dialogue

Currently included in the Interactive_OSCE benchmark dataset.

### Utility Scripts

| File Name | Description | Purpose |
|-----------|-------------|---------|
| `MCC_generate_report.py` | Generate summary reports from experiment logs | Extract key information from debate logs for easier review |

## Usage

### Prerequisites

1. Install all dependencies:
```bash
pip install -r ../requirements.txt
```

2. Configure API credentials (in project root directory):
```bash
cp ../api_config.ini.example ../api_config.ini
# Then edit api_config.ini file and fill in your API keys
```

### Running Examples

#### 1. Run Individual Benchmarks

```bash
# Navigate to code directory
cd code

# Run MedQA
python MCC_MedQA.py

# Run PubMedQA
python MCC_PubmedQA.py

# Run MMLU Anatomy
python MCC_MMLU_Anatomy.py
```

#### 2. Batch Run MMLU Medical Subsets

```bash
cd code
for script in MCC_MMLU_*.py; do
    echo "Running $script..."
    python "$script"
done
```

#### 3. Run HealthBench

```bash
cd code

# Run all HealthBench subsets (default)
python MCC_run_HealthBench.py

# Run HealthBench Hard subset
python MCC_run_HealthBench.py --subset hard

# Run HealthBench Consensus subset
python MCC_run_HealthBench.py --subset consensus
```

**Parameters:**
- `--subset`: Specify HealthBench subset (`hard` or `consensus`, default runs all subsets)

#### 4. Run RABBITS Benchmark

```bash
cd code
# Original version
python MCC_for_RABBITS__original_MedQA.py
python MCC_for_RABBITS__original_MedMCQA.py

# Generic to brand name version
python MCC_for_RABBITS_generic_to_brand_MedQA.py
python MCC_for_RABBITS__generic_to_brand_MedMCQA.py
```

#### 5. Run with Custom Parameters

```bash
# Process first 3 cases
python MCC_MedQA.py -n 3

# Process 5 cases starting from index 10
python MCC_MedQA.py -n 5 -s 10
```

**Parameters:**
- `-n`: Number of cases to process
- `-s`: Starting index (optional, default is 0)

#### 6. Generate Summary Reports

After running experiments, generate summary reports for easier review:

```bash
cd code

# For Multiple-Choice Questions (MCQ)
python MCC_generate_report.py --type MCQ logs/MCQ/case_961.txt -o my_mcq_report.txt

# For Long-Form Questions (LFQ)
python MCC_generate_report.py --type LFQ logs/LFQ/Data_S16.txt -o my_lfq_report.txt
```

**Parameters:**
- `--type`: Specify the question type (`MCQ` for multiple-choice questions or `LFQ` for long-form questions)
- Input file: Path to the log file (e.g., `logs/MCQ/case_961.txt`)
- `-o`: Output file name (optional, default generates summary in the same directory)

This extracts key information from the debate logs and creates concise summaries of model interactions and reasoning processes.

## Code Structure

All scripts follow a similar structure:

1. **Configuration** - Load API configurations and parameters
2. **Data Loading** - Read test data from benchmarks directory
3. **MCC Framework** - Implement multi-model collaboration logic
   - Initial response phase
   - Critical discussion phase
   - Consensus generation phase
4. **Results Saving** - Save dialogue logs and results to logs directory

## Output Description

After running scripts, results will be saved in the `../logs/` directory:

- `logs/MedQA/` - MedQA experiment logs
- `logs/PubmedQA/` - PubMedQA experiment logs
- `logs/MMLU_*/` - MMLU subset experiment logs
- And more...

Each log file contains:
- Complete multi-model dialogue process
- Content of each discussion round
- Final consensus answer
- Evaluation metrics (e.g., accuracy)

## Important Notes

1. Valid API keys are required (OpenAI, Qwen, DeepSeek, etc.)
2. Some benchmarks may take considerable time to run
3. Ensure corresponding data files exist in the `benchmarks/` directory
4. API calls incur costs - please monitor usage

## Support

For questions, please refer to:
- Main README: `../README.md`
- Benchmarks README: `../benchmarks/README.md`
- Project GitHub Issues page
