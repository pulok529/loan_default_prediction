import numpy as np
import pandas as pd

def main(out_path: str = "data/synthetic_credit.csv", n: int = 20000, seed: int = 7):
    rng = np.random.default_rng(seed)
    amount = rng.normal(40000, 20000, n).clip(1000, 200000)
    income = rng.normal(60000, 25000, n).clip(8000, 250000)
    age = rng.integers(18, 70, n)
    term_months = rng.choice([6,12,18,24,36,48,60], size=n, p=[0.1,0.18,0.14,0.2,0.18,0.12,0.08])
    interest = rng.normal(0.16, 0.05, n).clip(0.02, 0.45)
    dti = rng.normal(0.35, 0.15, n).clip(0.01, 1.2)
    credit_score = rng.normal(680, 70, n).clip(300, 850)
    num_open_accounts = rng.integers(0, 18, n)

    # default probability (toy but plausible): higher DTI, higher interest, lower credit score, higher amount/income
    logit = (
        2.2*dti
        + 3.0*(interest-0.12)
        + 0.00001*(amount-30000)
        - 0.006*(credit_score-650)
        + 0.002*(term_months-24)
        - 0.00001*(income-60000)
    )
    p = 1/(1+np.exp(-logit))
    y = rng.binomial(1, p.clip(0.01, 0.85))

    df = pd.DataFrame({
        "amount": amount,
        "income": income,
        "age": age,
        "term_months": term_months,
        "interest": interest,
        "dti": dti,
        "credit_score": credit_score,
        "num_open_accounts": num_open_accounts,
        "default": y
    })
    import os
    os.makedirs("data", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(df)} rows. Default rate={df['default'].mean():.3f}")

if __name__ == "__main__":
    main()
