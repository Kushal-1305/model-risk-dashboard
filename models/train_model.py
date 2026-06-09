"""
Model training script — trains an XGBoost classifier on the German Credit dataset.

Why XGBoost for credit scoring?
  - It's the industry standard for tabular data in banking/fintech
  - Handles missing values natively
  - Produces calibrated probability scores (needed for risk scoring)
  - Interpretable via SHAP values (required for SR 11-7 model explainability)
  - Outperforms logistic regression on non-linear relationships in credit data

What we save:
  - xgb_credit_model.pkl    → the trained model (to be loaded by dashboard/validation)
  - feature_names.json      → ordered list of feature names (critical for SHAP)
  - baseline_metrics.json   → AUC, KS, Gini on train/test/OOT (our benchmark)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    classification_report, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_PATH, TEST_PATH, OOT_PATH,
    MODEL_PATH, FEATURE_NAMES_PATH, BASELINE_METRICS,
    TARGET_COLUMN, XGBOOST_PARAMS, MODELS_DIR
)


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    KS (Kolmogorov-Smirnov) Statistic.

    What it measures: how well the model separates "good" customers from "bad" ones.
    Think of it as: if you rank all customers by their predicted risk score,
    how far apart are the cumulative distributions of good vs bad customers?

    Interpretation:
      KS > 0.40 → Excellent separation
      KS 0.30–0.40 → Good
      KS 0.20–0.30 → Acceptable (minimum threshold in most banks)
      KS < 0.20 → Poor — model needs revalidation
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = float(np.max(tpr - fpr))
    return ks


def compute_gini(auc: float) -> float:
    """
    Gini Coefficient = 2 * AUC - 1.

    Gini normalizes AUC to a [-1, 1] scale.
    A random model has Gini=0, a perfect model has Gini=1.
    Most banks require Gini > 0.30 for production credit models.
    """
    return 2 * auc - 1


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    PSI (Population Stability Index) — measures how much the score distribution
    has shifted between two populations (e.g., train vs OOT).

    PSI < 0.10  → Stable (no action needed)
    PSI 0.10–0.20 → Moderate shift (monitor closely)
    PSI > 0.20  → Major shift (investigate, possible retraining)
    """
    # Create score buckets based on training distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0]

    # Add small epsilon to avoid log(0) or division by zero
    eps = 1e-6
    expected_pct = (expected_counts + eps) / (len(expected) + eps * bins)
    actual_pct   = (actual_counts   + eps) / (len(actual)   + eps * bins)

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return psi


def evaluate(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all standard credit model metrics for a given split."""
    auc  = float(roc_auc_score(y_true, y_prob))
    ks   = compute_ks_statistic(y_true, y_prob)
    gini = compute_gini(auc)

    print(f"\n  [{name}]")
    print(f"    AUC  : {auc:.4f}")
    print(f"    KS   : {ks:.4f}")
    print(f"    Gini : {gini:.4f}")

    return {"split": name, "auc": auc, "ks": ks, "gini": gini}


# ── Main training pipeline ────────────────────────────────────────────────────

def load_splits():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    oot   = pd.read_csv(OOT_PATH)

    feature_cols = [c for c in train.columns if c != TARGET_COLUMN]

    X_train, y_train = train[feature_cols], train[TARGET_COLUMN]
    X_test,  y_test  = test[feature_cols],  test[TARGET_COLUMN]
    X_oot,   y_oot   = oot[feature_cols],   oot[TARGET_COLUMN]

    return X_train, y_train, X_test, y_test, X_oot, y_oot, feature_cols


def train():
    print("=" * 60)
    print("  MODEL TRAINING — XGBoost Credit Scoring Model")
    print("=" * 60)

    X_train, y_train, X_test, y_test, X_oot, y_oot, feature_cols = load_splits()

    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)} | OOT size: {len(X_oot)}")

    # ── Train the model ────────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        verbose=False
    )
    print("Training complete.")

    # ── Compute probabilities ──────────────────────────────────────────────────
    # predict_proba returns [P(good), P(bad)] — we want P(bad) = column index 1
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs  = model.predict_proba(X_test)[:, 1]
    oot_probs   = model.predict_proba(X_oot)[:, 1]

    # ── Evaluate on all three splits ───────────────────────────────────────────
    print("\nModel Performance:")
    metrics = {
        "train": evaluate("TRAIN", y_train.values, train_probs),
        "test":  evaluate("TEST",  y_test.values,  test_probs),
        "oot":   evaluate("OOT",   y_oot.values,   oot_probs),
    }

    # ── PSI: score stability between train and test/OOT ───────────────────────
    psi_test = compute_psi(train_probs, test_probs)
    psi_oot  = compute_psi(train_probs, oot_probs)
    metrics["psi_train_vs_test"] = round(psi_test, 4)
    metrics["psi_train_vs_oot"]  = round(psi_oot, 4)
    print(f"\n  PSI (Train vs Test) : {psi_test:.4f}  {'GREEN' if psi_test < 0.10 else 'YELLOW' if psi_test < 0.20 else 'RED'}")
    print(f"  PSI (Train vs OOT)  : {psi_oot:.4f}   {'GREEN' if psi_oot < 0.10 else 'YELLOW' if psi_oot < 0.20 else 'RED'}")

    # ── Overfit check ──────────────────────────────────────────────────────────
    auc_gap = metrics["train"]["auc"] - metrics["test"]["auc"]
    metrics["overfit_gap_auc"] = round(auc_gap, 4)
    print(f"\n  Overfit Gap (Train AUC - Test AUC): {auc_gap:.4f}  {'OK' if auc_gap < 0.05 else 'WARNING: Possible overfit'}")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Feature names saved → {FEATURE_NAMES_PATH}")

    with open(BASELINE_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Baseline metrics saved → {BASELINE_METRICS}")

    print("\n" + "=" * 60)
    print("  Phase 1 Complete. Model is ready for validation.")
    print("=" * 60)

    return model, metrics


if __name__ == "__main__":
    train()
