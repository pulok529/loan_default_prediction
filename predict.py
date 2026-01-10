import argparse
import joblib
import pandas as pd
from features import FEATURES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amount", type=float, required=True)
    ap.add_argument("--income", type=float, required=True)
    ap.add_argument("--age", type=int, required=True)
    ap.add_argument("--term_months", type=int, required=True)
    ap.add_argument("--interest", type=float, required=True)
    ap.add_argument("--dti", type=float, default=0.35)
    ap.add_argument("--credit_score", type=float, default=680)
    ap.add_argument("--num_open_accounts", type=int, default=4)
    args = ap.parse_args()

    model = joblib.load("model.joblib")
    row = {k: getattr(args, k) for k in FEATURES}
    X = pd.DataFrame([row])
    p = model.predict_proba(X)[0,1]
    print(f"Predicted default probability: {p:.3f}")

if __name__ == "__main__":
    main()
