# Credit Risk Analysis and Modeling

This project performs statistical analysis and builds baseline credit risk models using the Kaggle **Default of Credit Card Clients** dataset.

## Dataset

The analysis uses the Kaggle dataset **Default of Credit Card Clients** (UCI Credit Card): https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

## Setup

1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
2. Unzip it and place the CSV in `data/raw/` (e.g., `data/raw/UCI_Credit_Card.csv`).
3. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Statistical Analysis + Modeling

```bash
python src/credit_risk_analysis.py \
  --data-path data/raw/UCI_Credit_Card.csv \
  --output-dir outputs
```

## Outputs

The script writes:
- Summary statistics, missing-value report, and target distribution.
- Feature correlation heatmap.
- Model metrics (AUC, accuracy, precision, recall, F1).

All outputs are stored in `outputs/`.
