"""Integration tests for the FastAPI credit risk scoring service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from loan_default.train import train_and_evaluate


@pytest.fixture(scope="session")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    """Train a test model bundle into a temporary folder and instantiate the TestClient."""
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    train_and_evaluate(n=4000, out_dir=artifact_dir, seed=42)
    model_path = artifact_dir / "model.joblib"
    app = create_app(str(model_path))
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    """Test that /health returns status ok and current model version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_strong_vs_weak_applicant(client: TestClient) -> None:
    """A prime applicant gets a lower default probability than a subprime one."""
    strong_applicant = {
        "loan_amount": 10000.0,
        "annual_income": 125000.0,
        "age": 45,
        "employment_years": 12.0,
        "home_ownership": "mortgage",
        "purpose": "debt_consolidation",
        "term_months": 36,
        "interest_rate": 0.075,
        "debt_to_income": 0.12,
        "credit_score": 800.0,
        "num_open_accounts": 9,
        "num_delinquencies_2y": 0,
        "credit_utilization": 0.10,
    }

    weak_applicant = {
        "loan_amount": 40000.0,
        "annual_income": 28000.0,
        "age": 22,
        "employment_years": 1.0,
        "home_ownership": "rent",
        "purpose": "business",
        "term_months": 60,
        "interest_rate": 0.28,
        "debt_to_income": 0.55,
        "credit_score": 520.0,
        "num_open_accounts": 3,
        "num_delinquencies_2y": 3,
        "credit_utilization": 0.95,
    }

    res_strong = client.post("/predict", json=strong_applicant)
    res_weak = client.post("/predict", json=weak_applicant)

    assert res_strong.status_code == 200
    assert res_weak.status_code == 200

    strong_data = res_strong.json()
    weak_data = res_weak.json()

    assert strong_data["probability_of_default"] < weak_data["probability_of_default"]
    assert strong_data["decision"] == "approve"
    assert len(weak_data["top_factors"]) > 0


def test_validation_error_credit_score_900(client: TestClient) -> None:
    """Test that a credit score of 900 triggers a 422 Unprocessable Entity error (max 850)."""
    invalid_applicant = {
        "loan_amount": 15000.0,
        "annual_income": 70000.0,
        "age": 34,
        "employment_years": 6.0,
        "home_ownership": "own",
        "purpose": "home_improvement",
        "term_months": 36,
        "interest_rate": 0.11,
        "debt_to_income": 0.25,
        "credit_score": 900.0,  # Exceeds max 850
        "num_open_accounts": 6,
        "num_delinquencies_2y": 0,
        "credit_utilization": 0.35,
    }

    response = client.post("/predict", json=invalid_applicant)
    assert response.status_code == 422


def test_batch_prediction_length(client: TestClient) -> None:
    """Test that batch prediction endpoint returns an array matching input batch size."""
    applicant = {
        "loan_amount": 18000.0,
        "annual_income": 65000.0,
        "age": 38,
        "employment_years": 7.0,
        "home_ownership": "mortgage",
        "purpose": "debt_consolidation",
        "term_months": 36,
        "interest_rate": 0.12,
        "debt_to_income": 0.30,
        "credit_score": 690.0,
        "num_open_accounts": 7,
        "num_delinquencies_2y": 0,
        "credit_utilization": 0.40,
    }

    batch_payload = [applicant, applicant, applicant, applicant]
    response = client.post("/predict/batch", json=batch_payload)

    assert response.status_code == 200
    results = response.json()
    assert len(results) == len(batch_payload)
    for item in results:
        assert "probability_of_default" in item
        assert "risk_band" in item
        assert "decision" in item
        assert "top_factors" in item
