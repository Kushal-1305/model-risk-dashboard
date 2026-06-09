"""
Data Drift Detection — Validation Module 2

SR 11-7 Requirement: "The bank should monitor whether the characteristics of
the data used to build the model continue to be representative of current
conditions. Significant changes in the input data distribution should trigger
model review."

What is data drift?
  The real world changes after a model is deployed. New customer segments emerge,
  economic conditions shift, lending policies change. If the model keeps running
  on data that looks completely different from what it was trained on, its
  predictions become unreliable — even if we never changed the model.

Tests in this module:
  1. Feature-level drift (KS test + PSI per feature)  — which inputs have shifted?
  2. Chi-square test for categorical features          — have category frequencies changed?
  3. Target drift                                      — has the default rate itself changed?
  4. Covariate shift detection                         — can a classifier tell train from OOT?
  5. Drift summary heatmap data                        — structured for dashboard visualisation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_PATH, TEST_PATH, OOT_PATH,
    FEATURE_NAMES_PATH, TARGET_COLUMN,
    PSI_GREEN, PSI_YELLOW, DRIFT_REPORT_PATH
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def psi_feature(train_vals, compare_vals, bins=10):
    """
    Compute PSI for a single numeric feature.
    Buckets are defined by the training data percentiles.
    """
    bp = np.percentile(train_vals, np.linspace(0, 100, bins + 1))
    bp = np.unique(bp)                 # remove duplicate breakpoints
    if len(bp) < 3:
        return None                    # can't compute PSI with too few unique values

    bp[0] = -np.inf; bp[-1] = np.inf
    eps = 1e-6
    e_pct = (np.histogram(train_vals,   bp)[0] + eps) / (len(train_vals)   + eps * bins)
    a_pct = (np.histogram(compare_vals, bp)[0] + eps) / (len(compare_vals) + eps * bins)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_flag(value):
    if value is None:
        return "N/A"
    return "GREEN" if value < PSI_GREEN else ("YELLOW" if value < PSI_YELLOW else "RED")


def ks_test(train_vals, compare_vals):
    """
    KS (Kolmogorov-Smirnov) test for numeric features.
    Returns: statistic (0 to 1) and p-value.
    p-value < 0.05 means the distributions are significantly different.
    """
    stat, p_val = stats.ks_2samp(train_vals, compare_vals)
    return round(float(stat), 4), round(float(p_val), 4)


def chi_square_test(train_vals, compare_vals):
    """
    Chi-square test for categorical features (encoded as integers here).
    Compares observed category frequencies in train vs comparison set.
    p-value < 0.05 = significantly different distribution.
    """
    # Get all unique categories present in either split
    all_cats = sorted(set(train_vals) | set(compare_vals))

    train_counts   = np.array([np.sum(train_vals   == c) for c in all_cats]) + 1   # +1 smoothing
    compare_counts = np.array([np.sum(compare_vals == c) for c in all_cats]) + 1

    # Normalise to same total to make counts comparable
    compare_counts = compare_counts * (train_counts.sum() / compare_counts.sum())

    stat, p_val = stats.chisquare(compare_counts, f_exp=train_counts)
    return round(float(stat), 4), round(float(p_val), 4)


# ── Test 1: Feature-Level Drift (KS + PSI) ───────────────────────────────────

def feature_drift_test(train_df, compare_df, features, label="OOT"):
    """
    For every input feature, compute:
    - KS statistic + p-value (is the distribution different?)
    - PSI (how much has it shifted?)
    - Flag: GREEN / YELLOW / RED

    This tells us exactly WHICH features are drifting — a critical finding
    that the model developer must address.
    """
    print(f"\n  [Test 1] Feature-Level Drift (Train vs {label})")

    results = []
    red_features    = []
    yellow_features = []

    for feat in features:
        train_vals   = train_df[feat].values.astype(float)
        compare_vals = compare_df[feat].values.astype(float)

        n_unique = len(np.unique(train_vals))

        if n_unique <= 10:
            # Treat as categorical: use chi-square
            stat, p_val  = chi_square_test(train_vals.astype(int), compare_vals.astype(int))
            test_used    = "chi_square"
        else:
            # Treat as continuous: use KS
            stat, p_val  = ks_test(train_vals, compare_vals)
            test_used    = "ks"

        psi_val  = psi_feature(train_vals, compare_vals)
        flag     = psi_flag(psi_val)

        result = {
            "feature":    feat,
            "test":       test_used,
            "statistic":  stat,
            "p_value":    p_val,
            "psi":        round(psi_val, 4) if psi_val is not None else None,
            "drift_flag": flag,
            "significant": p_val < 0.05,
        }
        results.append(result)

        if flag == "RED":
            red_features.append(feat)
        elif flag == "YELLOW":
            yellow_features.append(feat)

    # Print summary
    print(f"    RED features    ({len(red_features)}): {red_features}")
    print(f"    YELLOW features ({len(yellow_features)}): {yellow_features}")
    print(f"    GREEN features  ({len(features) - len(red_features) - len(yellow_features)}): stable")

    return {
        "label":         label,
        "features":      results,
        "n_red":         len(red_features),
        "n_yellow":      len(yellow_features),
        "n_green":       len(features) - len(red_features) - len(yellow_features),
        "red_features":  red_features,
        "yellow_features": yellow_features,
    }


# ── Test 2: Target Drift ──────────────────────────────────────────────────────

def target_drift_test(train_df, test_df, oot_df):
    """
    Has the actual default rate changed across splits?

    This is the most direct form of drift — if the base rate changes,
    every probability the model produces is wrong in a systematic way.
    A model trained on 30% default rate and deployed on a 50% default rate
    population will dramatically under-predict risk.
    """
    print("\n  [Test 2] Target Drift (Default Rate)")

    train_rate = float(train_df[TARGET_COLUMN].mean())
    test_rate  = float(test_df[TARGET_COLUMN].mean())
    oot_rate   = float(oot_df[TARGET_COLUMN].mean())

    drift_test_vs_train = abs(test_rate - train_rate)
    drift_oot_vs_train  = abs(oot_rate  - train_rate)

    result = {
        "train_default_rate": round(train_rate, 4),
        "test_default_rate":  round(test_rate, 4),
        "oot_default_rate":   round(oot_rate, 4),
        "abs_drift_test": round(drift_test_vs_train, 4),
        "abs_drift_oot":  round(drift_oot_vs_train, 4),
        "flag_test": "GREEN" if drift_test_vs_train < 0.03 else ("YELLOW" if drift_test_vs_train < 0.07 else "RED"),
        "flag_oot":  "GREEN" if drift_oot_vs_train  < 0.03 else ("YELLOW" if drift_oot_vs_train  < 0.07 else "RED"),
    }

    print(f"    Train default rate : {train_rate:.1%}")
    print(f"    Test  default rate : {test_rate:.1%}  (drift: {drift_test_vs_train:.1%})  [{result['flag_test']}]")
    print(f"    OOT   default rate : {oot_rate:.1%}  (drift: {drift_oot_vs_train:.1%})   [{result['flag_oot']}]")

    return result


# ── Test 3: Covariate Shift Detection (Discriminative Classifier) ─────────────

def covariate_shift_test(train_df, oot_df, features):
    """
    This is a clever technique: train a SEPARATE classifier whose job is
    to predict 'is this row from the training set or the OOT set?'

    If this classifier achieves high AUC (say 0.75+), it means the two
    datasets are easily distinguishable — strong covariate shift.
    If AUC ~ 0.5, the datasets look identical — no significant shift.

    Intuition: if you can't tell train from OOT apart, there's no drift.
    """
    print("\n  [Test 3] Covariate Shift Detection")

    train_X = train_df[features].copy()
    oot_X   = oot_df[features].copy()

    train_X["_split"] = 0    # label: 0 = training data
    oot_X["_split"]   = 1    # label: 1 = OOT data

    combined = pd.concat([train_X, oot_X], ignore_index=True)
    X = combined[features]
    y = combined["_split"]

    clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    auc_scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    mean_auc   = float(np.mean(auc_scores))

    flag = "GREEN" if mean_auc < 0.60 else ("YELLOW" if mean_auc < 0.70 else "RED")
    print(f"    Discriminative AUC (train vs OOT): {mean_auc:.4f}  [{flag}]")
    print(f"    Interpretation: {'Datasets are indistinguishable — no significant covariate shift' if mean_auc < 0.60 else 'Datasets are distinguishable — covariate shift detected'}")

    return {
        "discriminative_auc": round(mean_auc, 4),
        "cv_scores": [round(s, 4) for s in auc_scores.tolist()],
        "flag": flag,
        "interpretation": (
            "No significant covariate shift" if mean_auc < 0.60
            else "Moderate covariate shift detected" if mean_auc < 0.70
            else "Significant covariate shift — model inputs have changed substantially"
        ),
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run_drift_validation():
    print("=" * 60)
    print("  VALIDATION MODULE 2: Data Drift Detection")
    print("=" * 60)

    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    oot   = pd.read_csv(OOT_PATH)

    with open(FEATURE_NAMES_PATH) as f:
        features = json.load(f)

    report = {
        "feature_drift_oot":    feature_drift_test(train, oot,  features, label="OOT"),
        "feature_drift_test":   feature_drift_test(train, test, features, label="Test"),
        "target_drift":         target_drift_test(train, test, oot),
        "covariate_shift":      covariate_shift_test(train, oot, features),
    }

    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved → {DRIFT_REPORT_PATH}")

    print("\n  FINDINGS SUMMARY:")
    print(f"    RED features (OOT)   : {report['feature_drift_oot']['n_red']}")
    print(f"    Target drift (OOT)   : {report['target_drift']['flag_oot']}")
    print(f"    Covariate shift      : {report['covariate_shift']['flag']}")

    return report


if __name__ == "__main__":
    run_drift_validation()
