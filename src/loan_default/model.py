"""Model loading, inference, and explainable decisioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from loan_default.features import add_derived


def load_bundle(path: str | Path = "artifacts/model.joblib") -> dict[str, Any]:
    """Load the model bundle from disk."""
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found at {bundle_path.resolve()}")
    bundle: dict[str, Any] = joblib.load(bundle_path)
    return bundle


def extract_top_factors(applicant: dict[str, Any], payment_to_income: float | None) -> list[str]:
    """Extract up to 3 human-readable adverse factors based on risk rules."""
    factors: list[str] = []

    dti = applicant.get("debt_to_income")
    if dti is not None and dti > 0.45:
        factors.append("High debt-to-income ratio (> 0.45)")

    credit_score = applicant.get("credit_score")
    if credit_score is not None and credit_score < 600:
        factors.append("Low credit score (< 600)")

    utilization = applicant.get("credit_utilization")
    if utilization is not None and utilization > 0.8:
        factors.append("High credit utilization (> 0.80)")

    delinquencies = applicant.get("num_delinquencies_2y")
    if delinquencies is not None and delinquencies > 0:
        factors.append("Recent delinquencies (> 0 in last 2 years)")

    pti_known = payment_to_income is not None and not np.isnan(payment_to_income)
    if pti_known and payment_to_income > 0.2:
        factors.append("High payment-to-income ratio (> 0.20)")

    return factors[:3]


def assign_risk_band(probability_of_default: float) -> str:
    """Map probability of default to risk band A-E.

    Bands:
      A: < 5%
      B: 5% - 10%
      C: 10% - 20%
      D: 20% - 35%
      E: > 35%
    """
    if probability_of_default < 0.05:
        return "A"
    if probability_of_default < 0.10:
        return "B"
    if probability_of_default < 0.20:
        return "C"
    if probability_of_default < 0.35:
        return "D"
    return "E"


def predict(bundle: dict[str, Any], applicant: dict[str, Any]) -> dict[str, Any]:
    """Score a single loan applicant dictionary and return decision details.

    Parameters
    ----------
    bundle : dict
        Model bundle loaded via load_bundle.
    applicant : dict
        Raw applicant features.

    Returns
    -------
    dict
        Prediction results including probability_of_default, risk_band, decision,
        and top_factors.
    """
    df = pd.DataFrame([applicant])
    df = add_derived(df)

    chosen_model_name = bundle.get("chosen_model_name", "hgb_calibrated")
    model = bundle["models"][chosen_model_name]
    feature_cols = bundle["feature_lists"]["all"]

    proba = float(model.predict_proba(df[feature_cols])[:, 1][0])
    risk_band = assign_risk_band(proba)

    threshold = float(bundle.get("threshold", 0.10))
    decision = "approve" if proba < threshold else "review"

    pti_val = None
    if "payment_to_income" in df.columns:
        raw_pti = df["payment_to_income"].iloc[0]
        pti_val = float(raw_pti) if pd.notna(raw_pti) else None

    top_factors = extract_top_factors(applicant, pti_val)

    return {
        "probability_of_default": round(proba, 4),
        "risk_band": risk_band,
        "decision": decision,
        "top_factors": top_factors,
    }


def predict_batch(
    bundle: dict[str, Any],
    applicants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score a batch of loan applicants efficiently."""
    if not applicants:
        return []

    df = pd.DataFrame(applicants)
    df = add_derived(df)

    chosen_model_name = bundle.get("chosen_model_name", "hgb_calibrated")
    model = bundle["models"][chosen_model_name]
    feature_cols = bundle["feature_lists"]["all"]

    probabilities = model.predict_proba(df[feature_cols])[:, 1]
    threshold = float(bundle.get("threshold", 0.10))

    results: list[dict[str, Any]] = []
    for i, applicant in enumerate(applicants):
        proba = float(probabilities[i])
        risk_band = assign_risk_band(proba)
        decision = "approve" if proba < threshold else "review"

        raw_pti = df["payment_to_income"].iloc[i]
        pti_val = float(raw_pti) if pd.notna(raw_pti) else None
        factors = extract_top_factors(applicant, pti_val)

        results.append({
            "probability_of_default": round(proba, 4),
            "risk_band": risk_band,
            "decision": decision,
            "top_factors": factors,
        })

    return results
