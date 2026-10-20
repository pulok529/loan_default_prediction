"""Model training, calibration, evaluation, and artifact serialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from loan_default import __version__  # noqa: E402
from loan_default.data import generate  # noqa: E402
from loan_default.features import (  # noqa: E402
    ALL_FEATURES,
    CATEGORICAL,
    NUMERIC,
    add_derived,
    build_preprocessor,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Calculate key credit risk evaluation metrics."""
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks_stat = float(np.max(tpr - fpr))
    gini = float(2.0 * roc_auc - 1.0)

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "ks_stat": round(ks_stat, 4),
        "gini": round(gini, 4),
    }


def plot_roc_curves(
    y_test: np.ndarray,
    probs_dict: dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """Generate and save ROC curve comparison plot."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_calibration_curves(
    y_test: np.ndarray,
    probs_dict: dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """Generate and save calibration (reliability) curves."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, probs in probs_dict.items():
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test,
            probs,
            n_bins=10,
            strategy="uniform",
        )
        ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            label=name,
            lw=2,
            markersize=5,
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectly Calibrated")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Mean Predicted Probability of Default")
    ax.set_ylabel("Empirical Default Rate")
    ax.set_title("Probability Calibration (Reliability Diagram)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(
    feature_names: list[str],
    importances_mean: np.ndarray,
    importances_std: np.ndarray,
    save_path: Path,
) -> None:
    """Generate horizontal bar chart of top 10 permutation importances."""
    fig, ax = plt.subplots(figsize=(8, 5))

    indices = np.argsort(importances_mean)[-10:]
    y_pos = np.arange(len(indices))

    ax.barh(
        y_pos,
        importances_mean[indices],
        xerr=importances_std[indices],
        align="center",
        color="#2b5c8f",
        edgecolor="#1a3b5c",
        alpha=0.85,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Permutation Feature Importance (Validation ROC-AUC Drop)")
    ax.set_title("Top 10 Feature Importances (HistGradientBoosting)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def train_and_evaluate(
    n: int = 30000,
    out_dir: str | Path = "artifacts",
    seed: int = 11,
) -> dict[str, Any]:
    """Execute end-to-end dataset generation, training, calibration, and evaluation."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic dataset and add derived features
    raw_df = generate(n=n, seed=seed)
    df = add_derived(raw_df)

    # 2. Time-based split: first 18 months train, months 19-21 val, months 22-24 test
    train_df = df[df["application_date"] < "2025-07-01"].copy()
    val_df = df[
        (df["application_date"] >= "2025-07-01") & (df["application_date"] < "2025-10-01")
    ].copy()
    test_df = df[df["application_date"] >= "2025-10-01"].copy()

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["default"].to_numpy(dtype=int)

    X_val = val_df[ALL_FEATURES]
    y_val = val_df["default"].to_numpy(dtype=int)

    X_test = test_df[ALL_FEATURES]
    y_test = test_df["default"].to_numpy(dtype=int)

    # 3. Model A: Logistic Regression Baseline ("Scorecard")
    scorecard_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
            ),
        ]
    )
    scorecard_pipeline.fit(X_train, y_train)

    # 4. Model B: HistGradientBoosting pipeline and 3-fold isotonic calibration on train
    hgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", HistGradientBoostingClassifier(random_state=42)),
        ]
    )
    hgb_pipeline.fit(X_train, y_train)

    hgb_calibrated = CalibratedClassifierCV(
        estimator=hgb_pipeline,
        method="isotonic",
        cv=3,
    )
    hgb_calibrated.fit(X_train, y_train)

    # 5. Validation Permutation Importance for HGB pipeline (top 10)
    perm_result = permutation_importance(
        hgb_pipeline,
        X_val,
        y_val,
        n_repeats=5,
        random_state=42,
        scoring="roc_auc",
    )
    top_indices = np.argsort(perm_result.importances_mean)[::-1][:10]
    top_features = [
        {
            "feature": ALL_FEATURES[i],
            "importance_mean": round(float(perm_result.importances_mean[i]), 5),
            "importance_std": round(float(perm_result.importances_std[i]), 5),
        }
        for i in top_indices
    ]

    # 6. Predictions on test set
    scorecard_test_probs = scorecard_pipeline.predict_proba(X_test)[:, 1]
    hgb_test_probs = hgb_calibrated.predict_proba(X_test)[:, 1]

    metrics_scorecard = compute_metrics(y_test, scorecard_test_probs)
    metrics_hgb = compute_metrics(y_test, hgb_test_probs)

    # 7. Approval Policy Cut-off chosen on validation to keep approval rate at 80%
    val_probs = hgb_calibrated.predict_proba(X_val)[:, 1]
    threshold_80 = float(np.percentile(val_probs, 80))

    approved_mask_test = hgb_test_probs <= threshold_80
    test_approval_rate = float(np.mean(approved_mask_test))
    if np.sum(approved_mask_test) > 0:
        test_default_rate_approved = float(np.mean(y_test[approved_mask_test]))
    else:
        test_default_rate_approved = 0.0
    test_default_rate_overall = float(np.mean(y_test))

    policy_results = {
        "threshold": round(threshold_80, 4),
        "target_approval_rate": 0.80,
        "test_approval_rate": round(test_approval_rate, 4),
        "test_default_rate_approved": round(test_default_rate_approved, 4),
        "test_default_rate_overall": round(test_default_rate_overall, 4),
    }

    # 8. Plot artifacts
    plot_roc_curves(
        y_test,
        {
            "Logistic Regression (Scorecard)": scorecard_test_probs,
            "HistGradientBoosting (Calibrated)": hgb_test_probs,
        },
        out_path / "roc_curve.png",
    )

    plot_calibration_curves(
        y_test,
        {
            "Logistic Regression": scorecard_test_probs,
            "HistGradientBoosting (Calibrated)": hgb_test_probs,
        },
        out_path / "calibration.png",
    )

    plot_feature_importance(
        ALL_FEATURES,
        perm_result.importances_mean,
        perm_result.importances_std,
        out_path / "feature_importance.png",
    )

    # 9. Save model bundle
    bundle = {
        "models": {
            "scorecard": scorecard_pipeline,
            "hgb_uncalibrated": hgb_pipeline,
            "hgb_calibrated": hgb_calibrated,
        },
        "chosen_model_name": "hgb_calibrated",
        "threshold": round(threshold_80, 4),
        "feature_lists": {
            "numeric": NUMERIC,
            "categorical": CATEGORICAL,
            "all": ALL_FEATURES,
        },
        "version": __version__,
    }
    joblib.dump(bundle, out_path / "model.joblib")

    # 10. Save metrics.json
    all_metrics = {
        "version": __version__,
        "dataset": {
            "n_total": len(df),
            "n_train": len(train_df),
            "n_validation": len(val_df),
            "n_test": len(test_df),
        },
        "models": {
            "logistic_regression": metrics_scorecard,
            "hist_gradient_boosting_calibrated": metrics_hgb,
        },
        "approval_policy": policy_results,
        "top_features": top_features,
    }
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # 11. Print markdown table (no line over 100 chars)
    table_lines: list[str] = [
        "| Model | ROC-AUC | PR-AUC | Brier | KS Stat | Gini |",
        "|:---|---:|---:|---:|---:|---:|",
        (
            f"| Logistic Regression (Scorecard) | {metrics_scorecard['roc_auc']:.4f} | "
            f"{metrics_scorecard['pr_auc']:.4f} | {metrics_scorecard['brier_score']:.4f} | "
            f"{metrics_scorecard['ks_stat']:.4f} | {metrics_scorecard['gini']:.4f} |"
        ),
        (
            f"| HistGradientBoosting (Calibrated) | {metrics_hgb['roc_auc']:.4f} | "
            f"{metrics_hgb['pr_auc']:.4f} | {metrics_hgb['brier_score']:.4f} | "
            f"{metrics_hgb['ks_stat']:.4f} | {metrics_hgb['gini']:.4f} |"
        ),
        "",
        "| Policy Metric | Value |",
        "|:---|---:|",
        f"| Validation Threshold (80% Approval) | {policy_results['threshold']:.4f} |",
        f"| Test Approval Rate | {policy_results['test_approval_rate'] * 100:.2f}% |",
        (
            "| Test Default Rate (Approved) | "
            f"{policy_results['test_default_rate_approved'] * 100:.2f}% |"
        ),
        (
            "| Test Default Rate (Overall) | "
            f"{policy_results['test_default_rate_overall'] * 100:.2f}% |"
        ),
    ]
    markdown_output = "\n".join(table_lines)
    print(markdown_output)

    return all_metrics


def main() -> None:
    """CLI entry point for loan_default.train."""
    parser = argparse.ArgumentParser(description="Train credit risk loan default prediction models")
    parser.add_argument("--n", type=int, default=30000, help="Number of synthetic samples")
    parser.add_argument("--out", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    args = parser.parse_args()

    train_and_evaluate(n=args.n, out_dir=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
