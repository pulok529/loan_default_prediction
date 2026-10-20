"""Feature definitions, derivations, and preprocessing pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL: list[str] = [
    "home_ownership",
    "purpose",
]

NUMERIC: list[str] = [
    "loan_amount",
    "annual_income",
    "age",
    "employment_years",
    "term_months",
    "interest_rate",
    "debt_to_income",
    "credit_score",
    "num_open_accounts",
    "num_delinquencies_2y",
    "credit_utilization",
    "loan_to_income",
    "monthly_payment",
    "payment_to_income",
]

ALL_FEATURES: list[str] = NUMERIC + CATEGORICAL


def calculate_amortisation_payment(
    principal: np.ndarray,
    annual_rate: np.ndarray,
    term_months: np.ndarray,
) -> np.ndarray:
    """Calculate fixed monthly amortisation payment using standard loan formula.

    Formula:
        M = P * [r(1 + r)^n] / [(1 + r)^n - 1]
        where P is principal, r is monthly interest rate, and n is term in months.
        If r == 0, M = P / n.
    """
    monthly_rate = annual_rate / 12.0
    factor = np.power(1.0 + monthly_rate, term_months)
    denominator = factor - 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        payment = np.where(
            monthly_rate == 0.0,
            principal / term_months,
            principal * (monthly_rate * factor) / denominator,
        )
    return payment


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived credit risk features to a DataFrame.

    Derived features:
      - loan_to_income: loan_amount / annual_income
      - monthly_payment: amortisation formula payment
      - payment_to_income: (monthly_payment * 12) / annual_income
    """
    out = df.copy()

    loan_amount = out["loan_amount"].to_numpy(dtype=float)
    annual_income = out["annual_income"].to_numpy(dtype=float)
    term_months = out["term_months"].to_numpy(dtype=float)
    interest_rate = out["interest_rate"].to_numpy(dtype=float)

    valid_income = np.where(annual_income > 0, annual_income, np.nan)
    out["loan_to_income"] = loan_amount / valid_income

    monthly_payment = calculate_amortisation_payment(loan_amount, interest_rate, term_months)
    out["monthly_payment"] = np.round(monthly_payment, 2)
    out["payment_to_income"] = (out["monthly_payment"] * 12.0) / valid_income

    return out


def build_preprocessor() -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for loan applicant features.

    Numeric features: SimpleImputer(strategy='median') + StandardScaler()
    Categorical features: SimpleImputer(strategy='most_frequent') + OneHotEncoder(ignore)
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC),
            ("categorical", categorical_pipeline, CATEGORICAL),
        ]
    )
