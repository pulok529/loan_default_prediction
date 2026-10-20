"""Unit tests for feature engineering, amortisation formulas, and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from loan_default.data import generate
from loan_default.features import (
    ALL_FEATURES,
    add_derived,
    build_preprocessor,
    calculate_amortisation_payment,
)


def test_amortisation_hand_computed_value() -> None:
    """Test monthly amortisation calculation against a hand-computed financial textbook benchmark.

    Benchmark case:
      Principal = $10,000
      Annual interest rate = 12% (monthly rate = 1%)
      Term = 12 months
      Hand calculation:
        (1 + 0.01)^12 = 1.12682503
        Monthly payment = 10000 * [0.01 * 1.12682503] / [1.12682503 - 1] = $888.49
    """
    principal = np.array([10000.0])
    rate = np.array([0.12])
    term = np.array([12.0])

    payment = calculate_amortisation_payment(principal, rate, term)
    assert np.isclose(payment[0], 888.49, atol=0.01), f"Expected 888.49, got {payment[0]:.4f}"


def test_add_derived_columns() -> None:
    """Test that add_derived creates loan_to_income, monthly_payment, and payment_to_income."""
    raw = pd.DataFrame({
        "loan_amount": [12000.0, 30000.0],
        "annual_income": [60000.0, 100000.0],
        "term_months": [24, 36],
        "interest_rate": [0.10, 0.15],
    })

    derived = add_derived(raw)
    for col in ["loan_to_income", "monthly_payment", "payment_to_income"]:
        assert col in derived.columns

    # Check loan_to_income
    assert np.isclose(derived["loan_to_income"].iloc[0], 12000.0 / 60000.0)
    assert np.isclose(derived["loan_to_income"].iloc[1], 30000.0 / 100000.0)

    # Check payment_to_income = (monthly_payment * 12) / annual_income
    p1 = derived["monthly_payment"].iloc[0]
    expected_pti1 = (p1 * 12.0) / 60000.0
    assert np.isclose(derived["payment_to_income"].iloc[0], expected_pti1)


def test_build_preprocessor_no_nan() -> None:
    """Test that build_preprocessor properly imputes missing values and yields zero NaNs."""
    raw_df = generate(n=500, seed=99)
    df = add_derived(raw_df)

    # Confirm missing values exist in the raw features
    assert df["employment_years"].isna().sum() > 0 or df["annual_income"].isna().sum() > 0

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df[ALL_FEATURES])

    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(transformed).any(), "Preprocessor output contains NaNs"
