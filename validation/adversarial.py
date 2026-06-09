"""
Adversarial Input Testing — Validation Module 3

SR 11-7 Requirement: "Sensitivity analysis should include testing under stressed
or adverse conditions to identify potential model weaknesses and failure modes."

What is adversarial testing?
  Normal testing checks: 'does the model work on typical data?'
  Adversarial testing checks: 'does the model behave sensibly on edge cases,
  extreme inputs, and scenarios it was never designed for?'

  In banking, this is critical because:
  - Fraudsters deliberately craft inputs to game credit models
  - Edge cases can cause unexpectedly large or small predictions
  - Directional violations (e.g. more savings → higher risk) are a red flag
    that the model has learned spurious correlations

Tests in this module:
  1. Boundary probing          — stability near the decision threshold
  2. Extreme value injection   — out-of-range / impossible inputs
  3. Null / missing data       — how model handles missing inputs
  4. Recession stress scenario — systematic feature worsening
  5. Directional sanity checks — monotonicity of key features
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    OOT_PATH, MODEL_PATH, FEATURE_NAMES_PATH,
    TARGET_COLUMN, ADV_REPORT_PATH
)


# ── Test 1: Boundary Probing ──────────────────────────────────────────────────

def boundary_probe_test(X, model, threshold=0.50, perturbation_pct=0.05):
    """
    Find samples whose predicted score is close to the decision threshold (±5%).
    Then perturb each feature by a small amount and check if the decision flips.

    Why this matters: if tiny input changes cause the decision to flip
    (approve ↔ reject), the model is unstable and unfair — two almost
    identical applicants get opposite decisions.

    Instability rate > 20% is a significant finding.
    """
    print("\n  [Test 1] Boundary Probing")

    probs = model.predict_proba(X)[:, 1]

    # Find borderline samples (within ±5% of the threshold)
    margin = 0.05
    borderline_idx = np.where(
        (probs >= threshold - margin) & (probs <= threshold + margin)
    )[0]

    print(f"    Borderline samples (within ±{margin} of threshold {threshold}): {len(borderline_idx)}")

    if len(borderline_idx) == 0:
        return {"borderline_count": 0, "instability_rate": 0.0, "details": []}

    flip_count = 0
    details = []

    for idx in borderline_idx[:50]:   # test first 50 borderline samples
        original_prob  = float(probs[idx])
        original_label = int(original_prob >= threshold)

        flip_results = []
        for feat in X.columns:
            X_perturbed = X.astype(float).copy()
            feat_val    = float(X_perturbed.iloc[idx][feat])

            # Perturb up and down by perturbation_pct
            X_perturbed.iloc[idx, X_perturbed.columns.get_loc(feat)] = feat_val * (1 + perturbation_pct)
            prob_up    = float(model.predict_proba(X_perturbed)[:, 1][idx])
            label_up   = int(prob_up >= threshold)

            X_perturbed.iloc[idx, X_perturbed.columns.get_loc(feat)] = feat_val * (1 - perturbation_pct)
            prob_down  = float(model.predict_proba(X_perturbed)[:, 1][idx])
            label_down = int(prob_down >= threshold)

            # Reset
            X_perturbed.iloc[idx, X_perturbed.columns.get_loc(feat)] = feat_val

            if label_up != original_label or label_down != original_label:
                flip_results.append(feat)

        if flip_results:
            flip_count += 1
            details.append({
                "sample_idx":    int(idx),
                "original_prob": round(original_prob, 4),
                "flipping_features": flip_results[:5],   # cap at 5
            })

    instability_rate = round(flip_count / len(borderline_idx), 4)
    flag = "GREEN" if instability_rate < 0.20 else ("YELLOW" if instability_rate < 0.40 else "RED")

    print(f"    Decision flips: {flip_count}/{len(borderline_idx)} = {instability_rate:.1%}  [{flag}]")

    return {
        "borderline_count":  len(borderline_idx),
        "flips":             flip_count,
        "instability_rate":  instability_rate,
        "flag":              flag,
        "details":           details[:10],
    }


# ── Test 2: Extreme Value Injection ──────────────────────────────────────────

def extreme_value_test(X, model, feature_names):
    """
    Feed the model inputs that are far outside the training distribution.
    E.g. age = 200, credit_amount = 1,000,000, duration = 0.

    A robust model should:
    - Not crash or produce NaN
    - Produce probabilities that are still in [0, 1]
    - Produce scores that make intuitive sense (very high loan → higher risk)

    We test 3 scenarios:
    - All features at max observed value × 10
    - All features at 0
    - Each feature at max individually while others stay at median
    """
    print("\n  [Test 2] Extreme Value Injection")

    median_row = X.median()
    max_row    = X.max()
    results    = []

    # Scenario A: All features set to 0
    row_zeros = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
    prob_zero = float(model.predict_proba(row_zeros)[:, 1][0])
    valid_zero = not (np.isnan(prob_zero) or np.isinf(prob_zero))
    results.append({
        "scenario": "all_zeros",
        "predicted_prob": round(prob_zero, 4) if valid_zero else None,
        "valid_output":   valid_zero,
    })

    # Scenario B: All features at 10× their max
    row_extreme = pd.DataFrame([(max_row * 10).values], columns=feature_names)
    prob_extreme = float(model.predict_proba(row_extreme)[:, 1][0])
    valid_extreme = not (np.isnan(prob_extreme) or np.isinf(prob_extreme))
    results.append({
        "scenario": "all_10x_max",
        "predicted_prob": round(prob_extreme, 4) if valid_extreme else None,
        "valid_output":   valid_extreme,
    })

    # Scenario C: Each feature individually at 10× max, others at median
    feature_sensitivity = []
    for feat in feature_names:
        row = median_row.copy()
        row[feat] = max_row[feat] * 10
        row_df = pd.DataFrame([row.values], columns=feature_names)
        prob   = float(model.predict_proba(row_df)[:, 1][0])
        valid  = not (np.isnan(prob) or np.isinf(prob))
        base_prob = float(model.predict_proba(pd.DataFrame([median_row.values], columns=feature_names))[:, 1][0])
        feature_sensitivity.append({
            "feature":    feat,
            "base_prob":  round(base_prob, 4),
            "extreme_prob": round(prob, 4) if valid else None,
            "delta":      round(prob - base_prob, 4) if valid else None,
            "valid":      valid,
        })

    all_valid = all(r["valid_output"] for r in results)
    print(f"    All-zeros prob    : {prob_zero:.4f}   {'OK' if valid_zero else 'ERROR: NaN/Inf'}")
    print(f"    All-10x-max prob  : {prob_extreme:.4f}  {'OK' if valid_extreme else 'ERROR: NaN/Inf'}")
    print(f"    Model stability   : {'All outputs valid' if all_valid else 'INVALID outputs detected'}")

    # Top features causing largest swing under extremes
    top_sensitive = sorted(
        [f for f in feature_sensitivity if f["delta"] is not None],
        key=lambda x: abs(x["delta"]), reverse=True
    )[:5]
    print(f"    Top sensitive features: {[f['feature'] for f in top_sensitive]}")

    return {
        "scenarios":            results,
        "all_outputs_valid":    all_valid,
        "feature_sensitivity":  feature_sensitivity,
        "top_5_sensitive":      top_sensitive,
    }


# ── Test 3: Missing Data Robustness ──────────────────────────────────────────

def missing_data_test(X, model, feature_names):
    """
    Replace each feature with its median (simulating missing data being imputed),
    then with a zero. Check how much the predictions shift.

    Features whose absence causes large prediction changes are 'fragile' —
    the model heavily depends on them, which is a risk if data pipelines fail.
    """
    print("\n  [Test 3] Missing Data Robustness")

    baseline_probs = model.predict_proba(X)[:, 1]
    results = []

    for feat in feature_names:
        X_missing = X.copy()
        X_missing[feat] = 0    # simulate missing → imputed as 0

        missing_probs = model.predict_proba(X_missing)[:, 1]
        mean_delta    = float(np.mean(np.abs(missing_probs - baseline_probs)))
        max_delta     = float(np.max(np.abs(missing_probs - baseline_probs)))

        results.append({
            "feature":   feat,
            "mean_abs_delta": round(mean_delta, 4),
            "max_abs_delta":  round(max_delta, 4),
            "fragile": mean_delta > 0.10,
        })

    fragile = [r for r in results if r["fragile"]]
    print(f"    Fragile features (mean delta > 0.10 when missing): {[f['feature'] for f in fragile]}")

    top_impact = sorted(results, key=lambda x: x["mean_abs_delta"], reverse=True)[:5]
    return {
        "feature_impact":  results,
        "fragile_features": [f["feature"] for f in fragile],
        "top_5_impact":    top_impact,
    }


# ── Test 4: Recession Stress Scenario ────────────────────────────────────────

def recession_stress_test(X, model, feature_names):
    """
    Simulate a recession by systematically worsening financial features:
    - Reduce savings (savings_status → worst bucket)
    - Increase loan amounts by 50%
    - Worsen employment stability (employment → lowest bucket)

    Then ask: by how much does the average predicted default risk increase?

    This is a common scenario banks include in model validation reports —
    'how would our portfolio perform in a 2008-style downturn?'
    """
    print("\n  [Test 4] Recession Stress Scenario")

    baseline_probs = model.predict_proba(X)[:, 1]
    baseline_mean  = float(np.mean(baseline_probs))

    X_stressed = X.copy()

    # Worsen features if they exist in our dataset
    # (using min/max values to simulate worst-case)
    stress_applied = []

    if "credit_amount" in feature_names:
        X_stressed["credit_amount"] = X_stressed["credit_amount"] * 1.5
        stress_applied.append("credit_amount × 1.5")

    if "savings_status" in feature_names:
        X_stressed["savings_status"] = X_stressed["savings_status"].min()   # worst savings
        stress_applied.append("savings_status → worst bucket")

    if "employment" in feature_names:
        X_stressed["employment"] = X_stressed["employment"].min()           # worst employment
        stress_applied.append("employment → worst bucket")

    if "duration" in feature_names:
        X_stressed["duration"] = X_stressed["duration"] * 1.25             # longer loan term
        stress_applied.append("duration × 1.25")

    stressed_probs = model.predict_proba(X_stressed)[:, 1]
    stressed_mean  = float(np.mean(stressed_probs))

    risk_increase  = stressed_mean - baseline_mean
    pct_increase   = risk_increase / baseline_mean if baseline_mean > 0 else 0

    print(f"    Stress factors applied    : {stress_applied}")
    print(f"    Baseline avg risk score   : {baseline_mean:.4f}")
    print(f"    Stressed avg risk score   : {stressed_mean:.4f}")
    print(f"    Absolute risk increase    : +{risk_increase:.4f}")
    print(f"    Relative risk increase    : +{pct_increase:.1%}")

    flag = "GREEN" if pct_increase < 0.20 else ("YELLOW" if pct_increase < 0.50 else "RED")

    return {
        "stress_factors":       stress_applied,
        "baseline_mean_risk":   round(baseline_mean, 4),
        "stressed_mean_risk":   round(stressed_mean, 4),
        "absolute_increase":    round(risk_increase, 4),
        "relative_increase_pct": round(pct_increase, 4),
        "flag": flag,
    }


# ── Test 5: Directional Sanity Checks ────────────────────────────────────────

def directional_sanity_test(X, model, feature_names):
    """
    For key features, we expect a clear direction:
    - Higher credit_amount → HIGHER risk (borrowing more = more risky)
    - Longer duration → HIGHER risk (longer loan = more time to default)
    - Better savings_status → LOWER risk

    We test this by:
    1. Sorting samples by a feature
    2. Checking if model scores increase/decrease in the expected direction

    Correlation between feature rank and model score rank tells us:
    - Positive correlation where we expect positive: PASS
    - Negative correlation where we expect negative: PASS
    - Wrong direction: FAIL (model has learned something backwards)
    """
    print("\n  [Test 5] Directional Sanity Checks")

    # Expected direction: +1 = higher feature value should increase risk
    #                     -1 = higher feature value should decrease risk
    expected_directions = {
        "credit_amount":         +1,   # more debt → more risk
        "duration":              +1,   # longer loan → more risk
        "age":                   -1,   # older → more experience → less risk
        "savings_status":        -1,   # more savings → less risk
        "installment_commitment": +1,  # higher installment % → more burden
    }

    results = []
    baseline_probs = model.predict_proba(X)[:, 1]

    for feat, expected_dir in expected_directions.items():
        if feat not in feature_names:
            continue

        feat_ranks  = X[feat].rank()
        score_ranks = pd.Series(baseline_probs).rank()

        # Spearman rank correlation between feature and model score
        corr = float(feat_ranks.corr(score_ranks, method="spearman"))
        actual_dir = +1 if corr > 0 else -1

        passed = actual_dir == expected_dir
        results.append({
            "feature":       feat,
            "expected_dir":  "increases_risk" if expected_dir == +1 else "decreases_risk",
            "actual_corr":   round(corr, 4),
            "direction_match": passed,
            "flag":          "GREEN" if passed else "RED",
        })

        direction_str = "↑ risk" if expected_dir == +1 else "↓ risk"
        status_str    = "PASS" if passed else "FAIL (direction reversed!)"
        print(f"    {feat:30s} expected {direction_str}  corr={corr:+.4f}  {status_str}")

    n_pass = sum(1 for r in results if r["direction_match"])
    n_fail = len(results) - n_pass

    return {
        "checks":   results,
        "n_pass":   n_pass,
        "n_fail":   n_fail,
        "overall_flag": "GREEN" if n_fail == 0 else ("YELLOW" if n_fail <= 1 else "RED"),
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run_adversarial_validation():
    print("=" * 60)
    print("  VALIDATION MODULE 3: Adversarial Input Testing")
    print("=" * 60)

    oot   = pd.read_csv(OOT_PATH)
    model = joblib.load(MODEL_PATH)

    with open(FEATURE_NAMES_PATH) as f:
        features = json.load(f)

    X_oot = oot[features]

    report = {
        "boundary_probing":  boundary_probe_test(X_oot, model),
        "extreme_values":    extreme_value_test(X_oot, model, features),
        "missing_data":      missing_data_test(X_oot, model, features),
        "recession_stress":  recession_stress_test(X_oot, model, features),
        "directional_sanity": directional_sanity_test(X_oot, model, features),
    }

    os.makedirs(os.path.dirname(ADV_REPORT_PATH), exist_ok=True)
    with open(ADV_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved → {ADV_REPORT_PATH}")

    print("\n  FINDINGS SUMMARY:")
    print(f"    Boundary instability   : {report['boundary_probing']['flag']}")
    print(f"    Directional violations : {report['directional_sanity']['n_fail']} features")
    print(f"    Recession risk increase: {report['recession_stress']['relative_increase_pct']:.1%}  [{report['recession_stress']['flag']}]")

    return report


if __name__ == "__main__":
    run_adversarial_validation()
