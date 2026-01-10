import argparse
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from features import load_dataset, split_xy

def train_models(X_train, y_train):
    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000))
        ]),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    }
    for m in models.values():
        m.fit(X_train, y_train)
    return models

def eval_model(model, X_valid, y_valid):
    proba = model.predict_proba(X_valid)[:,1]
    auc = roc_auc_score(y_valid, proba)
    ap = average_precision_score(y_valid, proba)
    pred = (proba >= 0.5).astype(int)
    return auc, ap, classification_report(y_valid, pred, digits=3)

def main(data_path: str):
    df = load_dataset(data_path)
    X, y = split_xy(df)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = train_models(X_train, y_train)
    best_name, best_auc = None, -1
    for name, model in models.items():
        auc, ap, rep = eval_model(model, X_valid, y_valid)
        print(f"== {name} ==")
        print(f"AUC={auc:.3f}  PR-AUC={ap:.3f}")
        print(rep)
        if auc > best_auc:
            best_auc = auc
            best_name = name
    best = models[best_name]
    joblib.dump(best, "model.joblib")
    print(f"Saved best model: {best_name} -> model.joblib")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_credit.csv")
    args = ap.parse_args()
    import os
    if not os.path.exists(args.data):
        from make_synthetic import main as gen
        gen(out_path=args.data)
    main(args.data)
