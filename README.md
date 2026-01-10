# Loan Default Prediction (Credit Risk) - Starter Project

This project trains baseline ML models to predict loan default on a tabular dataset.
It includes:
- synthetic data generator (so it runs without private bank data)
- preprocessing + train/valid split
- baseline models (Logistic Regression, RandomForest)
- evaluation (AUC, PR-AUC, confusion matrix)
- model export

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
python predict.py --amount 50000 --income 65000 --age 32 --term_months 24 --interest 0.16
```

## Where to plug real data
Replace `data/synthetic_credit.csv` with your real (anonymized) dataset and adjust `features.py`.
