"""FastAPI application for serving real-time loan default predictions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from loan_default import __version__
from loan_default.model import load_bundle, predict, predict_batch
from loan_default.train import train_and_evaluate


class Applicant(BaseModel):
    """Borrower application schema with strict domain validation rules."""

    loan_amount: float = Field(..., gt=0, description="Requested loan amount in USD")
    annual_income: float | None = Field(
        default=None,
        gt=0,
        description="Self-reported annual gross income in USD",
    )
    age: int = Field(..., ge=18, le=100, description="Applicant age in years (18-100)")
    employment_years: float | None = Field(
        default=None,
        ge=0,
        description="Continuous years in current employment",
    )
    home_ownership: Literal["rent", "mortgage", "own"] = Field(
        ...,
        description="Residential status",
    )
    purpose: Literal[
        "debt_consolidation",
        "home_improvement",
        "business",
        "education",
        "medical",
        "car",
    ] = Field(..., description="Stated purpose of the loan")
    term_months: Literal[12, 24, 36, 48, 60] = Field(
        ...,
        description="Repayment period in months",
    )
    interest_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Annual interest rate expressed as decimal (e.g. 0.12)",
    )
    debt_to_income: float = Field(
        ...,
        ge=0.0,
        le=3.0,
        description="Monthly debt obligations divided by monthly income",
    )
    credit_score: float = Field(
        ...,
        ge=300.0,
        le=850.0,
        description="FICO / Vantage credit score between 300 and 850",
    )
    num_open_accounts: int = Field(
        ...,
        ge=0,
        description="Total active credit lines and accounts",
    )
    num_delinquencies_2y: int = Field(
        ...,
        ge=0,
        description="Number of 30+ days past-due delinquencies in past 2 years",
    )
    credit_utilization: float = Field(
        ...,
        ge=0.0,
        le=2.0,
        description="Revolving credit balance divided by total credit limit",
    )


class PredictionResponse(BaseModel):
    """Credit scoring prediction response."""

    probability_of_default: float = Field(
        ...,
        description="Calibrated probability of loan default (0.0 to 1.0)",
    )
    risk_band: str = Field(
        ...,
        description="Risk tier: A (<5%), B (5-10%), C (10-20%), D (20-35%), E (>35%)",
    )
    decision: str = Field(..., description="Automated policy recommendation: 'approve' or 'review'")
    top_factors: list[str] = Field(
        ...,
        description="Adverse risk drivers explaining non-prime risk scores",
    )


class HealthResponse(BaseModel):
    """System health check payload."""

    status: str
    model_version: str


def create_app(model_path: str | None = None) -> FastAPI:
    """Factory function to build and configure the FastAPI credit risk service."""
    app = FastAPI(
        title="Credit Risk Default Prediction API",
        description="Real-time default probability scoring and policy-driven loan underwriting.",
        version=__version__,
    )

    resolved_path = Path(model_path or os.environ.get("MODEL_PATH", "artifacts/model.joblib"))

    # If model bundle does not exist on disk, train a small initial model (n=8000)
    if not resolved_path.exists():
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        train_and_evaluate(n=8000, out_dir=resolved_path.parent)

    bundle: dict[str, Any] = load_bundle(resolved_path)
    app.state.bundle = bundle

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Health check endpoint confirming service readiness and active model version."""
        version = str(app.state.bundle.get("version", __version__))
        return HealthResponse(status="ok", model_version=version)

    @app.post("/predict", response_model=PredictionResponse)
    def predict_endpoint(applicant: Applicant, request: Request) -> PredictionResponse:
        """Score an individual loan applicant and return calibrated default probability."""
        result = predict(request.app.state.bundle, applicant.model_dump())
        return PredictionResponse(**result)

    @app.post("/predict/batch", response_model=list[PredictionResponse])
    def predict_batch_endpoint(
        applicants: list[Applicant],
        request: Request,
    ) -> list[PredictionResponse]:
        """Score a batch of loan applicants (maximum 1000 applicants per request)."""
        if len(applicants) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Batch size exceeds maximum limit of 1000 applicants.",
            )
        applicant_dicts = [a.model_dump() for a in applicants]
        results = predict_batch(request.app.state.bundle, applicant_dicts)
        return [PredictionResponse(**r) for r in results]

    return app


app: FastAPI = create_app()
