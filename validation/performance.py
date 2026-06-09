"""
Performance Degradation Tests — Validation Module 1

SR 11-7 Requirement: "The model should be tested across different time periods,
subpopulations, and stress scenarios to assess performance stability."

Tests in this module:
  1. Rolling window performance  — AUC/KS across data slices (simulates time)
  2. Score distribution shift    — histogram comparison train vs test vs OOT
  3. Threshold sensitivity       — precision/recall behaviour at different cutoffs
  4. Calibration check           — are predicted probabilities realistic?
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_PATH, TEST_PATH, OOT_PATH,
    MODEL_PATH, FEATURE_NAMES_PATH,
    TARGET_COLUMN, MIN_AUC_THRESHOLD, MIN_KS_THRESHOLD,
    PERF_REPORT_PATH
)


# ── Helpers (reused from train_model.py — kept here for module independence) ──

def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def gini(auc):
    return round(2 * auc - 1, 4)


def psi(expected, actual, bins=10):
    bp = np.percentile(expected, np.linspace(0, 100, bins + 1))
    bp[0] = -np.inf; bp[-1] = np.inf
    eps = 1e-6
    e_pct = (np.histogram(expected, bp)[0] + eps) / (len(expected) + eps * bins)
    a_pct = (np.histogram(actual,   bp)[0] + eps) / (len(actual)   + eps * bins)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def flag(value, green, red, higher_is_better=True):
    """Return RED / YELLOW / GREEN based on thresholds."""
    if higher_is_better:
        return "GREEN" if value >= green else ("YELLOW" if value >= red else "RED")
    else:
        return "GREEN" if value <= green else ("YELLOW" if value <= red else "RED")


# ── Test 1: Rolling Window Performance ───────────────────────────────────────

def rolling_window_test(X_all, y_all, model, n_windows=5):
    """
    Split all available data into N equal windows and evaluate the model
    on each one. This simulates: 'if we deployed the model and checked its
    performance every quarter, would it stay stable or drift?'

    A well-behaved model should show AUC variance < 0.05 across windows.
    """
    print("\n  [Test 1] Rolling Window Performance")

    window_size = len(X_all) // n_windows
    results = []

    for i in range(n_windows):
        start = i * window_size
        end   = start + window_size if i < n_windows - 1 else len(X_all)

        X_w = X_all.iloc[start:end]
        y_w = y_all.iloc[start:end]

        if y_w.nunique() < 2:          # skip windows with only one class
            continue

        probs = model.predict_proba(X_w)[:, 1]
        auc   = float(roc_auc_score(y_w, probs))
        ks    = ks_stat(y_w.values, probs)

        results.append({
            "window":     i + 1,
            "start_idx":  start,
            "end_idx":    end,
            "n_samples":  len(X_w),
            "default_rate": float(y_w.mean()),
            "auc":  round(auc, 4),
            "ks":   round(ks, 4),
            "auc_flag": flag(auc,  green=0.72, red=MIN_AUC_THRESHOLD),
            "ks_flag":  flag(ks,   green=0.35, red=MIN_KS_THRESHOLD),
        })
        print(f"    Window {i+1}: AUC={auc:.4f}  KS={ks:.4f}  N={len(X_w)}")

    auc_values = [r["auc"] for r in results]
    auc_variance = round(float(np.var(auc_values)), 6)
    print(f"    AUC variance across windows: {auc_variance:.6f}  "
          f"{'STABLE' if auc_variance < 0.003 else 'UNSTABLE — investigate'}")

    return {"windows": results, "auc_variance": auc_variance}


# ── Test 2: Score Distribution Shift ─────────────────────────────────────────

def score_distribution_test(train_probs, test_probs, oot_probs):
    """
    Compare the histogram of predicted scores across splits.

    Why it matters: if the score distribution looks completely different
    on OOT vs Train, then the model is seeing a different kind of population —
    a red flag for data drift or model instability.

    We compute:
    - Mean and std of scores per split
    - PSI (how far distributions have shifted)
    - Decile breakdown: what % of customers fall in each score band?
    """
    print("\n  [Test 2] Score Distribution Shift")

    def decile_breakdown(probs, label):
        breakpoints = np.percentile(probs, np.arange(0, 101, 10))
        counts = np.histogram(probs, bins=breakpoints)[0]
        return {f"decile_{i+1}": int(counts[i]) for i in range(len(counts))}

    psi_test = round(psi(train_probs, test_probs), 4)
    psi_oot  = round(psi(train_probs, oot_probs),  4)

    result = {
        "train": {
            "mean":  round(float(np.mean(train_probs)), 4),
            "std":   round(float(np.std(train_probs)),  4),
            "min":   round(float(np.min(train_probs)),  4),
            "max":   round(float(np.max(train_probs)),  4),
            "deciles": decile_breakdown(train_probs, "train"),
        },
        "test": {
            "mean":  round(float(np.mean(test_probs)), 4),
            "std":   round(float(np.std(test_probs)),  4),
            "min":   round(float(np.min(test_probs)),  4),
            "max":   round(float(np.max(test_probs)),  4),
            "deciles": decile_breakdown(test_probs, "test"),
        },
        "oot": {
            "mean":  round(float(np.mean(oot_probs)), 4),
            "std":   round(float(np.std(oot_probs)),  4),
            "min":   round(float(np.min(oot_probs)),  4),
            "max":   round(float(np.max(oot_probs)),  4),
            "deciles": decile_breakdown(oot_probs, "oot"),
        },
        "psi_train_vs_test": psi_test,
        "psi_train_vs_oot":  psi_oot,
        "psi_test_flag": "GREEN" if psi_test < 0.10 else ("YELLOW" if psi_test < 0.20 else "RED"),
        "psi_oot_flag":  "GREEN" if psi_oot  < 0.10 else ("YELLOW" if psi_oot  < 0.20 else "RED"),
    }

    print(f"    Train  → mean score: {result['train']['mean']:.4f}  std: {result['train']['std']:.4f}")
    print(f"    Test   → mean score: {result['test']['mean']:.4f}   std: {result['test']['std']:.4f}")
    print(f"    OOT    → mean score: {result['oot']['mean']:.4f}    std: {result['oot']['std']:.4f}")
    print(f"    PSI (Train vs Test): {psi_test:.4f}  [{result['psi_test_flag']}]")
    print(f"    PSI (Train vs OOT) : {psi_oot:.4f}   [{result['psi_oot_flag']}]")

    return result


# ── Test 3: Threshold Sensitivity ────────────────────────────────────────────

def threshold_sensitivity_test(y_true, y_prob, split_name="OOT"):
    """
    Banks don't just use the raw probability — they set a cutoff threshold.
    e.g., 'Reject anyone with predicted default probability > 0.40'

    This test sweeps thresholds from 0.1 to 0.9 and records precision, recall, F1.

    What we're looking for:
    - A GOOD model: metrics degrade smoothly as threshold changes
    - A BAD model: metrics collapse suddenly at certain thresholds (fragile)

    The 'optimal threshold' is the one maximising F1 — this is what the bank
    should actually use in deployment, not the default 0.5.
    """
    print(f"\n  [Test 3] Threshold Sensitivity ({split_name})")

    thresholds = np.arange(0.1, 0.91, 0.05)
    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue
        p  = float(precision_score(y_true, y_pred, zero_division=0))
        r  = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        results.append({
            "threshold": round(float(t), 2),
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
            "approvals": int((y_pred == 0).sum()),
            "rejections": int((y_pred == 1).sum()),
        })

    # Find optimal threshold (max F1)
    best = max(results, key=lambda x: x["f1"])
    print(f"    Optimal threshold (max F1): {best['threshold']}  →  "
          f"Precision={best['precision']:.4f}  Recall={best['recall']:.4f}  F1={best['f1']:.4f}")
    print(f"    At threshold 0.50         :  ", end="")
    t50 = next((r for r in results if r["threshold"] == 0.50), results[len(results)//2])
    print(f"Precision={t50['precision']:.4f}  Recall={t50['recall']:.4f}  F1={t50['f1']:.4f}")

    return {"split": split_name, "thresholds": results, "optimal_threshold": best}


# ── Test 4: Calibration Check ─────────────────────────────────────────────────

def calibration_test(y_true, y_prob, split_name="OOT"):
    """
    Calibration = 'Are the model's probabilities trustworthy?'

    If the model says 'this person has a 30% chance of defaulting',
    then among all people it gave ~30% scores, roughly 30% should actually default.

    We use a 'reliability diagram' — split scores into 10 buckets and
    compare predicted probability vs actual default rate in each bucket.

    Brier Score = mean squared error between predicted probs and actual outcomes.
    Lower is better. A perfect model = 0, random model ~ 0.25 for 50/50 data.
    """
    print(f"\n  [Test 4] Calibration Check ({split_name})")

    fraction_pos, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10)

    calibration_data = [
        {
            "predicted_prob": round(float(mp), 4),
            "actual_rate":    round(float(fp), 4),
            "gap":            round(float(abs(fp - mp)), 4),
        }
        for mp, fp in zip(mean_predicted, fraction_pos)
    ]

    brier = float(brier_score_loss(y_true, y_prob))
    max_gap = max(d["gap"] for d in calibration_data)

    print(f"    Brier Score : {brier:.4f}  (lower=better; random~0.25)")
    print(f"    Max cal gap : {max_gap:.4f}  {'WELL CALIBRATED' if max_gap < 0.05 else 'MISCALIBRATED — probabilities are misleading'}")

    return {
        "split": split_name,
        "brier_score": round(brier, 4),
        "max_calibration_gap": round(max_gap, 4),
        "calibration_flag": "GREEN" if max_gap < 0.05 else ("YELLOW" if max_gap < 0.10 else "RED"),
        "calibration_curve": calibration_data,
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run_performance_validation():
    print("=" * 60)
    print("  VALIDATION MODULE 1: Performance Degradation Tests")
    print("=" * 60)

    # Load data and model
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    oot   = pd.read_csv(OOT_PATH)
    model = joblib.load(MODEL_PATH)

    with open(FEATURE_NAMES_PATH) as f:
        features = json.load(f)

    X_train, y_train = train[features], train[TARGET_COLUMN]
    X_test,  y_test  = test[features],  test[TARGET_COLUMN]
    X_oot,   y_oot   = oot[features],   oot[TARGET_COLUMN]

    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs  = model.predict_proba(X_test)[:, 1]
    oot_probs   = model.predict_proba(X_oot)[:, 1]

    # Combine all data for rolling window test
    X_all = pd.concat([X_train, X_test, X_oot], ignore_index=True)
    y_all = pd.concat([y_train, y_test, y_oot], ignore_index=True)

    # Run all 4 tests
    report = {
        "rolling_window":        rolling_window_test(X_all, y_all, model),
        "score_distribution":    score_distribution_test(train_probs, test_probs, oot_probs),
        "threshold_sensitivity": threshold_sensitivity_test(y_oot.values, oot_probs, "OOT"),
        "calibration":           calibration_test(y_oot.values, oot_probs, "OOT"),
    }

    # Save report
    os.makedirs(os.path.dirname(PERF_REPORT_PATH), exist_ok=True)
    with open(PERF_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved → {PERF_REPORT_PATH}")

    # Summary findings
    print("\n  FINDINGS SUMMARY:")
    psi_flag = report["score_distribution"]["psi_oot_flag"]
    cal_flag = report["calibration"]["calibration_flag"]
    print(f"    Score distribution (PSI OOT) : {psi_flag}")
    print(f"    Calibration (max gap)        : {cal_flag}")
    print(f"    Optimal decision threshold   : {report['threshold_sensitivity']['optimal_threshold']['threshold']}")

    return report


if __name__ == "__main__":
    run_performance_validation()
