"""
Data loading and preprocessing for the German Credit Dataset.

What is the German Credit Dataset?
  - 1,000 loan applicants from a German bank
  - 20 features: age, job, savings, loan amount, duration, etc.
  - Target: 1 = default/bad credit risk, 0 = good credit risk
  - Publicly available via UCI / scikit-learn's OpenML integration

Why this dataset for Model Risk?
  - Used in academic literature and industry benchmarks for credit scoring
  - Mix of categorical + numeric features (realistic, messy, like real bank data)
  - Class imbalance (700 good : 300 bad) — a real challenge in credit risk
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, RAW_DATA_PATH, PROCESSED_PATH,
    TRAIN_PATH, TEST_PATH, OOT_PATH,
    TARGET_COLUMN, TRAIN_RATIO, TEST_RATIO, OOT_RATIO, RANDOM_STATE
)


def download_and_save_raw():
    """
    Pull the German Credit dataset from OpenML and save as CSV.
    OpenML is a public repository of ML datasets — sklearn can fetch from it directly.
    """
    print("[1/4] Downloading German Credit dataset from OpenML...")
    dataset = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")

    df = dataset.frame.copy()

    # The original target is 'good'/'bad' as strings — convert to 0/1
    # Convention in credit risk: 1 = BAD (default), 0 = GOOD
    # This way, our model predicts "probability of default"
    df[TARGET_COLUMN] = (df["class"] == "bad").astype(int)
    df.drop(columns=["class"], inplace=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"    Saved raw data → {RAW_DATA_PATH}  ({len(df)} rows, {df.shape[1]} columns)")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and encode the dataset.

    Why encoding?
    ML models are just math — they can't process the string "checking_status=no_checking".
    We use Label Encoding to convert each category to an integer.
    (For a production model, we'd use One-Hot Encoding or Target Encoding,
    but for this validation project Label Encoding is sufficient and keeps things readable.)
    """
    print("[2/4] Preprocessing — encoding categorical features...")

    df = df.copy()

    categorical_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    # Remove target if it accidentally ended up in here
    categorical_cols = [c for c in categorical_cols if c != TARGET_COLUMN]

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Ensure all remaining columns are numeric
    for col in df.columns:
        if col != TARGET_COLUMN:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with any NaN (there are very few in this dataset)
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    if before != after:
        print(f"    Dropped {before - after} rows with missing values")

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"    Saved processed data → {PROCESSED_PATH}")
    return df


def split_data(df: pd.DataFrame):
    """
    Split into Train / Test / OOT (Out-of-Time).

    Why 3 splits instead of the usual 2?

    In real banks, a model is trained on historical data (e.g., 2018–2020)
    and then validated on more recent data (e.g., 2021) that it never saw.
    This "OOT" set tests whether the model degrades over time as customer
    behavior and economic conditions change.

    We simulate this by holding out the last 20% of data as OOT.
    Train=60%, Test=20% (for hyperparameter tuning), OOT=20% (for final validation).
    """
    print("[3/4] Splitting data into Train / Test / OOT...")

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    # First: carve out the OOT set (last 20% — simulates temporal holdout)
    oot_size = int(len(df) * OOT_RATIO)
    X_dev, X_oot = X.iloc[:-oot_size], X.iloc[-oot_size:]
    y_dev, y_oot = y.iloc[:-oot_size], y.iloc[-oot_size:]

    # Second: split development data into Train and Test
    test_fraction_of_dev = TEST_RATIO / (TRAIN_RATIO + TEST_RATIO)
    X_train, X_test, y_train, y_test = train_test_split(
        X_dev, y_dev,
        test_size=test_fraction_of_dev,
        random_state=RANDOM_STATE,
        stratify=y_dev   # keep same class balance in each split
    )

    # Save each split with the target column included
    train_df = X_train.copy(); train_df[TARGET_COLUMN] = y_train.values
    test_df  = X_test.copy();  test_df[TARGET_COLUMN]  = y_test.values
    oot_df   = X_oot.copy();   oot_df[TARGET_COLUMN]   = y_oot.values

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH,   index=False)
    oot_df.to_csv(OOT_PATH,     index=False)

    print(f"    Train  : {len(train_df)} rows  | Default rate: {y_train.mean():.1%}")
    print(f"    Test   : {len(test_df)} rows  | Default rate: {y_test.mean():.1%}")
    print(f"    OOT    : {len(oot_df)} rows  | Default rate: {y_oot.mean():.1%}")

    return X_train, X_test, X_oot, y_train, y_test, y_oot


def print_data_summary(df: pd.DataFrame):
    """Print a quick summary of the dataset — useful for the SR 11-7 data section."""
    print("\n[4/4] Dataset Summary:")
    print(f"    Shape        : {df.shape}")
    print(f"    Target dist  : {df[TARGET_COLUMN].value_counts().to_dict()}")
    print(f"    Class balance: {df[TARGET_COLUMN].mean():.1%} default rate")
    print(f"    Features     : {[c for c in df.columns if c != TARGET_COLUMN]}")


if __name__ == "__main__":
    df_raw  = download_and_save_raw()
    df_proc = preprocess(df_raw)
    split_data(df_proc)
    print_data_summary(df_proc)
    print("\nData preparation complete.")
