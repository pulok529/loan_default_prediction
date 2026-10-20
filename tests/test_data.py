"""Unit tests for synthetic borrower data generation and saving."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from loan_default.data import generate, save_csv


def test_generate_default_rate_and_shape() -> None:
    """Test that generated dataset matches requested count and default rate is calibrated."""
    n = 10000
    df = generate(n=n, seed=11)

    assert len(df) == n
    default_rate = df["default"].mean()
    # Default rate should fall within calibrated range [0.06, 0.16] (target is ~0.10)
    assert 0.06 <= default_rate <= 0.16, f"Default rate {default_rate:.3f} outside bounds"


def test_generate_columns() -> None:
    """Test that all required schema columns are present in generated dataset."""
    df = generate(n=1000, seed=42)
    expected_columns = {
        "loan_id",
        "application_date",
        "loan_amount",
        "annual_income",
        "age",
        "employment_years",
        "home_ownership",
        "purpose",
        "term_months",
        "interest_rate",
        "debt_to_income",
        "credit_score",
        "num_open_accounts",
        "num_delinquencies_2y",
        "credit_utilization",
        "default",
    }
    assert expected_columns.issubset(set(df.columns))


def test_missing_values_present() -> None:
    """Test that employment_years and annual_income contain realistic missing values (~3%)."""
    n = 5000
    df = generate(n=n, seed=123)

    emp_missing = df["employment_years"].isna().sum() / n
    inc_missing = df["annual_income"].isna().sum() / n

    assert 0.01 <= emp_missing <= 0.06, f"Employment missing rate: {emp_missing:.3f}"
    assert 0.01 <= inc_missing <= 0.06, f"Income missing rate: {inc_missing:.3f}"


def test_save_csv(tmp_path: Path) -> None:
    """Test saving generated dataset to CSV."""
    df = generate(n=500, seed=7)
    file_path = tmp_path / "sub" / "test_data.csv"
    save_csv(df, file_path)

    assert file_path.exists()
    loaded = pd.read_csv(file_path)
    assert len(loaded) == 500
    assert "loan_id" in loaded.columns
