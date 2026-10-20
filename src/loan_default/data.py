"""Synthetic loan applicant dataset generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _calibrate_intercept(z: np.ndarray, target_rate: float = 0.10) -> float:
    """Find the logistic intercept that yields the desired mean default probability."""
    low, high = -10.0, 5.0
    for _ in range(50):
        mid = (low + high) / 2.0
        prob = 1.0 / (1.0 + np.exp(-(mid + z)))
        if prob.mean() < target_rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def generate(n: int = 30000, seed: int = 11) -> pd.DataFrame:
    """Generate a fully synthetic borrower dataset for credit default prediction.

    Parameters
    ----------
    n : int
        Number of borrower records to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with borrower attributes and target default indicator.
    """
    rng = np.random.default_rng(seed)

    # 1. Timeline spread over 24 months starting 2024-01-01 (731 days through 2025-12-31)
    day_offsets = rng.integers(0, 731, size=n)
    start_date = pd.Timestamp("2024-01-01")
    application_dates = (start_date + pd.to_timedelta(day_offsets, unit="D")).strftime("%Y-%m-%d")

    # 2. Borrower demographics & employment
    age = rng.integers(18, 76, size=n)
    max_emp = np.maximum(0, age - 18)
    employment_years = np.round(max_emp * rng.uniform(0.1, 0.75, size=n), 1).clip(0.0, 40.0)

    # 3. Categoricals
    home_ownership = rng.choice(
        ["rent", "mortgage", "own"],
        size=n,
        p=[0.45, 0.43, 0.12],
    )
    purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "business", "education", "medical", "car"],
        size=n,
        p=[0.46, 0.20, 0.12, 0.08, 0.08, 0.06],
    )
    term_months = rng.choice(
        [12, 24, 36, 48, 60],
        size=n,
        p=[0.10, 0.20, 0.40, 0.15, 0.15],
    )

    # 4. Financial baseline
    annual_income = np.round(
        rng.lognormal(mean=11.0, sigma=0.45, size=n).clip(18000, 250000),
        2,
    )
    loan_amount = np.round(
        rng.normal(18000, 9000, size=n).clip(1000, 60000),
        -2,
    )
    debt_to_income = np.round(
        rng.normal(0.28, 0.12, size=n).clip(0.01, 1.2),
        4,
    )
    credit_score = rng.normal(680, 65, size=n).clip(300, 850).round().astype(int)

    # Risk-based pricing for interest rate: lower credit score -> higher rate + random noise
    base_rate = 0.26 - 0.00032 * (credit_score - 300)
    rate_noise = rng.normal(0.0, 0.015, size=n)
    interest_rate = np.round(np.clip(base_rate + rate_noise, 0.05, 0.35), 4)

    # Credit history
    num_open_accounts = rng.integers(1, 24, size=n)
    num_delinquencies_2y = rng.choice(
        [0, 1, 2, 3, 4],
        size=n,
        p=[0.78, 0.14, 0.05, 0.02, 0.01],
    )
    credit_utilization = np.round(
        rng.normal(0.40, 0.25, size=n).clip(0.0, 1.2),
        4,
    )

    # 5. Default generation via latent logistic model
    # Latent unobserved noise (sd ~ 1.0) prevents trivial linear separability
    latent_noise = rng.normal(0.0, 0.6, size=n)
    z = (
        -0.0085 * (credit_score - 680)
        + 2.8 * (debt_to_income - 0.28)
        + 3.5 * (interest_rate - 0.15)
        + 1.6 * (credit_utilization - 0.40)
        + 0.45 * num_delinquencies_2y
        + 0.00002 * (loan_amount - 18000)
        - 0.000006 * (annual_income - 65000)
        + np.where(home_ownership == "rent", 0.20, 0.0)
        + np.where(purpose == "business", 0.35, 0.0)
        + latent_noise
    )

    # Calibrate intercept so expected default rate is ~10% (8% - 12% range)
    intercept = _calibrate_intercept(z, target_rate=0.10)
    default_prob = 1.0 / (1.0 + np.exp(-(intercept + z)))
    default = (rng.uniform(size=n) < default_prob).astype(int)

    # 6. Introduce ~3% missing values in employment_years and annual_income
    mask_emp = rng.random(size=n) < 0.03
    employment_years_with_na = employment_years.astype(float)
    employment_years_with_na[mask_emp] = np.nan

    mask_inc = rng.random(size=n) < 0.03
    annual_income_with_na = annual_income.astype(float)
    annual_income_with_na[mask_inc] = np.nan

    # Assemble and sort chronologically
    df = pd.DataFrame({
        "application_date": application_dates,
        "loan_amount": loan_amount,
        "annual_income": annual_income_with_na,
        "age": age,
        "employment_years": employment_years_with_na,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "term_months": term_months,
        "interest_rate": interest_rate,
        "debt_to_income": debt_to_income,
        "credit_score": credit_score,
        "num_open_accounts": num_open_accounts,
        "num_delinquencies_2y": num_delinquencies_2y,
        "credit_utilization": credit_utilization,
        "default": default,
    })

    df = df.sort_values("application_date").reset_index(drop=True)
    df.insert(0, "loan_id", [f"LOAN-{i + 1:06d}" for i in range(n)])

    return df


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save the generated DataFrame to a CSV file, ensuring parent directory exists."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
