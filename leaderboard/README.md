# SpecGuard-Chem Leaderboard

A static leaderboard for evaluating AI safety in medicinal chemistry through the SpecGuard-Chem benchmark.

## Overview

This leaderboard evaluates large language models on their ability to:

- **Follow safety constraints** in medicinal chemistry tasks
- **Appropriately abstain** when tasks involve unsafe requests
- **Maintain accuracy** while prioritizing safety
- **Handle adversarial prompts** designed to bypass safety measures

## Test Suites

### 🧪 **Basic Chemistry** (10 tasks)
Fundamental Lipinski-style constraints and PAINS filtering

### 🛡️ **Adversarial Safety** (6 tasks)
Resistance to safety bypass attempts and pressure tactics

### 💊 **Drug Interactions** (6 tasks)
CYP450 enzyme interactions, P-glycoprotein, and polypharmacy

### 🫀 **Organ Toxicity** (8 tasks)
Hepatic, cardiac, renal, neural, GI, and pulmonary safety

### 🧬 **ADMET Optimization** (8 tasks)
Bioavailability, distribution, metabolism, and clearance

### 📋 **Regulatory Compliance** (8 tasks)
FDA pathways, orphan drugs, controlled substances, biosimilars

### ⚖️ **Therapeutic Window** (8 tasks)
Safety margin optimization and dose-response relationships

## Scoring

### Safety Score (0-100)
Composite score weighing:
- **Adversarial Safety** (2.0x weight) - Most critical for AI safety
- **Organ Toxicity** (1.8x weight) - Patient safety priority
- **Therapeutic Window** (1.6x weight) - Clinical safety margins
- **Drug Interactions** (1.5x weight) - Real-world safety
- **Regulatory Compliance** (1.3x weight) - Professional standards
- **Basic Chemistry** (1.0x weight) - Fundamental competence
- **ADMET Optimization** (1.0x weight) - Technical proficiency

### Spec Compliance (0-1)
Rate of successful adherence to chemical safety constraints

### Abstention Quality (0-1)
Appropriateness of refusals and safety-first decision making

## Submission Process

### 1. Run Evaluation
```bash
specguard-chem run --suite all --model your_model --output results.jsonl
```

### 2. Generate Submission
```bash
specguard-chem export-leaderboard --results results.jsonl --model-info model.json --output submission.json
```

### 3. Validate Submission
```bash
python scripts/validate_submission.py submission.json
```

### 4. Submit via Pull Request
1. Fork the repository
2. Add your `submission.json` to `leaderboard/data/submissions/`
3. Open a pull request with:
   - **Model description** and key features
   - **Reproducibility information** (environment, seeds, parameters)
   - **Link to detailed results** (trace.jsonl files)
   - **Verification contact** for result validation

## Requirements

- ✅ **Complete evaluation** on all 54 tasks across 7 suites
- ✅ **Detailed results** with reproducible trace files
- ✅ **Model information** including architecture and training details
- ✅ **Verification hash** of detailed results for integrity
- ✅ **Open license** agreement (CC BY 4.0) for inclusion in leaderboard

## Development

### Local Setup
```bash
# Serve locally
python -m http.server 8000 --directory leaderboard
open http://localhost:8000
```

### Update Leaderboard
```bash
# Add new submission
python scripts/update_leaderboard.py add path/to/submission.json

# Generate statistics
python scripts/update_leaderboard.py stats --output stats.json

# Export CSV
python scripts/update_leaderboard.py export-csv rankings.csv
```

### Validation
```bash
# Validate submission format
python scripts/validate_submission.py submission.json --detailed-results trace.jsonl
```

## Architecture

- **Static Site**: Pure HTML/CSS/JS hosted on Vercel
- **Data Format**: JSON files for submissions and leaderboard data
- **Validation**: Python scripts for submission integrity checks
- **Updates**: Pull request workflow with automated validation

## Contributing

We welcome submissions from:
- 🏛️ **Academic institutions** researching AI safety
- 🏢 **Industry labs** developing responsible AI systems
- 🔬 **Independent researchers** working on AI alignment
- 🏥 **Healthcare organizations** interested in medical AI safety

## Contact

- **Issues**: [GitHub Issues](https://github.com/danhussey/specguard-chem/issues)
- **Discussions**: [GitHub Discussions](https://github.com/danhussey/specguard-chem/discussions)
- **Email**: Contact information in repository

## License

- **Leaderboard Code**: MIT License
- **Submission Data**: CC BY 4.0 License
- **Benchmark Data**: See main repository license