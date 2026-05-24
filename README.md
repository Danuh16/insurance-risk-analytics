# Insurance Risk Analytics & Predictive Modeling
**AlphaCare Insurance Solutions (ACIS) — South Africa Auto-Insurance**

## Overview
End-to-end analytics project covering 18 months of historical insurance claim data (Feb 2014 – Aug 2015). The goal is to identify low-risk customer segments and build a dynamic, risk-based pricing model.

## Project Structure
```
insurance-risk-analytics/
├── .github/workflows/ci.yml      # GitHub Actions CI (lint + test on every push)
├── .dvc/                          # DVC internals (committed to Git)
│   ├── config                     # Remote storage configuration
│   └── .gitignore
├── data/                          # Tracked by DVC — NOT committed to Git
│   ├── insurance_data.txt.dvc     # Version 1: raw data pointer
│   └── insurance_data_cleaned.txt.dvc  # Version 2: cleaned data pointer
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis (Task 1)
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Load & type-cast raw data
│   ├── data_cleaner.py            # DVC pipeline: raw → cleaned
│   ├── eda_utils.py               # Reusable plotting & analysis functions
│   ├── hypothesis_tests.py        # (Task 3)
│   └── modeling.py                # (Task 4)
├── reports/
│   ├── cleaning_metrics.json      # DVC-tracked pipeline metrics
│   └── figures/                   # EDA visualizations
├── tests/
│   └── test_data_loader.py        # Unit tests
├── dvc.yaml                       # DVC pipeline definition
├── requirements.txt
└── README.md
```

## Key Metrics
- **Loss Ratio** = TotalClaims / TotalPremium
- **Margin** = TotalPremium − TotalClaims

## Setup & Installation
```bash
git clone <repo-url>
cd insurance-risk-analytics
pip install -r requirements.txt
```

## Data Pipeline (DVC)

### First-time setup
```bash
# Install DVC
pip install dvc

# Initialize DVC (already done — .dvc/ folder is committed)
dvc init

# Add local remote storage (storage lives outside the project)
dvc remote add -d localstorage C:/Users/<your-username>/acis-dvc-storage
```

### Restore data
```bash
# Pull all tracked data versions from remote
dvc pull

# Downloads:
#   data/insurance_data.txt          (raw, 1,000,098 rows)
#   data/insurance_data_cleaned.txt  (cleaned, 618,176 rows)
```

### Run the full pipeline
```bash
# Reproduce all stages (clean → EDA) in correct order
dvc repro

# Check what changed between runs
dvc diff

# View pipeline metrics
dvc metrics show
```

### Push new data versions
```bash
# After modifying data:
dvc add data/insurance_data.txt
git add data/insurance_data.txt.dvc
git commit -m "data: update raw dataset"
dvc push
```

## Data Versions
| Version | File | Rows | MD5 | Description |
|---------|------|------|-----|-------------|
| v1 (raw) | `insurance_data.txt` | 1,000,098 | `f6b7009b` | Original ACIS dataset |
| v2 (clean) | `insurance_data_cleaned.txt` | 618,176 | `edfd0bcd` | After removing zero-premium rows, capping outliers, dropping empty cols |

## EDA Key Findings (Task 1)
| Metric | Value |
|--------|-------|
| Overall Loss Ratio | **1.0477** (portfolio is unprofitable) |
| Highest-risk province | Gauteng (LR = 1.222) |
| Lowest-risk province | Northern Cape (LR = 0.283) |
| Riskiest vehicle type | Heavy Commercial (LR = 1.628) |
| Female vs Male LR | 0.82 vs 0.88 (females lower risk) |
| Empty columns dropped | 8 (AlarmImmobiliser, TrackingDevice, etc.) |

## Tasks
| Task | Branch   | Status |
|------|----------|--------|
| 1 – Git setup & EDA       | task-1 | ✅ Merged |
| 2 – DVC data versioning   | task-2 | ✅ Merged |
| 3 – Hypothesis Testing    | task-3 | ⏳ Next |
| 4 – Predictive Modeling   | task-4 | ⏳ |
