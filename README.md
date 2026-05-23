# Insurance Risk Analytics & Predictive Modeling
**AlphaCare Insurance Solutions (ACIS) — South Africa Auto-Insurance**

## Overview
End-to-end analytics project covering 18 months of historical insurance claim data (Feb 2014 – Aug 2015). The goal is to identify low-risk customer segments and build a dynamic, risk-based pricing model.

## Project Structure
```
insurance-risk-analytics/
├── .github/workflows/ci.yml   # GitHub Actions CI pipeline
├── data/                       # Tracked by DVC, not Git
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_loader.py          # Data ingestion utilities
│   ├── eda_utils.py            # EDA helper functions
│   ├── hypothesis_tests.py     # Statistical tests
│   └── modeling.py             # ML pipeline
├── reports/
│   └── final_report.md
├── tests/                      # Unit tests
├── requirements.txt
└── README.md
```

## Key Metrics
- **Loss Ratio** = TotalClaims / TotalPremium
- **Margin** = TotalPremium − TotalClaims

## Setup
```bash
git clone <repo-url>
cd insurance-risk-analytics
pip install -r requirements.txt
# Pull data via DVC (see Task 2)
dvc pull
jupyter notebook notebooks/01_eda.ipynb
```

## Data Pipeline (DVC)
```bash
dvc init
dvc remote add -d localstorage /path/to/local/storage
dvc pull          # Restore tracked data
dvc push          # Push new versions
```

## Tasks
| Task | Branch   | Status |
|------|----------|--------|
| 1 – EDA              | task-1 | ✅ Done |
| 2 – DVC              | task-2 | 🔄 Next |
| 3 – Hypothesis Tests | task-3 | ⏳ |
| 4 – Modeling         | task-4 | ⏳ |
