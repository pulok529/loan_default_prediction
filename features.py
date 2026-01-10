import pandas as pd

FEATURES = ["amount","income","age","term_months","interest","dti","credit_score","num_open_accounts"]
TARGET = "default"

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def split_xy(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return X, y
