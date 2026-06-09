"""
SHAP Feature Importance Stability — Validation Module 4

SR 11-7 Requirement: "Model developers should provide evidence that the model's
key drivers are economically sensible, stable across time periods, and consistent
with the model's intended purpose."

What is SHAP?
  SHAP (SHapley Additive exPlanations) is a method from game theory that
  assigns each feature a 'contribution score' for every single prediction.

  Example: For a specific loan applicant, SHAP might say:
    - credit_amount pushed the risk score UP by +0.12 (large loan = risky)
    - savings_status pushed the risk score DOWN by -0.08 (has savings = safer)
    - age pushed the risk score DOWN by -0.04 (older = more reliable)
    - Net result: baseline risk 0.30 + 0.12 - 0.08 - 0.04 = risk score 0.30

  Global SHAP importance = average |SHAP value| across all customers.
  This tells us: "on average, which feature affects predictions the most?"

Tests in this module:
  1. Global feature importance   — overall ranking on test and OOT
  2. Bootstrap stability         — do rankings stay consistent across 50 subsamples?
  3. Top-feature flip test       — does flipping the #1 feature move predictions?
  4. Local explanation sample    — SHAP breakdown for 5 individual predictions
  5. Train vs OOT importance     — have key drivers shifted between periods?
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_PATH, TEST_PATH, OOT_PATH,
    MODEL_PATH, FEATURE_NAMES_PATH, TARGET_COLUMN,
    MAX_SHAP_RANK_VARIANCE, SHAP_REPORT_PATH, RANDOM_STATE
)


# ── Test 1: Global Feature Importance ────────────────────────────────────────

def global_importance_test(X, model, features, label="Test"):
    """
    Compute mean absolute SHAP values across all samples.
    This gives a ranked list: 'feature X is the most important driver overall'.

    In SR 11-7 terms: the top features must make business sense.
    If 'own_telephone' is #1 and 'credit_amount' is #10, that's a red flag —
    it suggests the model learned spurious correlations.
    """
    print(f"\n  [Test 1] Global Feature Importance ({label})")

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)

    # For binary classification, shap_values returns values for class 1 (default)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    importance_df = pd.DataFrame({
        "feature":    features,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance_df["rank"] = importance_df.index + 1

    print(f"    Top 10 features by SHAP importance ({label}):")
    for _, row in importance_df.head(10).iterrows():
        bar = "█" * int(row["mean_abs_shap"] * 200)
        print(f"      {int(row['rank']):2d}. {row['feature']:30s}  {row['mean_abs_shap']:.4f}  {bar}")

    return {
        "label":      label,
        "importance": importance_df.to_dict(orient="records"),
        "shap_values": shap_vals.tolist(),
    }


# ── Test 2: Bootstrap Stability ──────────────────────────────────────────────

def bootstrap_stability_test(X, model, features, n_bootstrap=30):
    """
    Run SHAP importance on 30 random subsamples (bootstrap).
    For each subsample, record the rank of every feature.
    Then compute the variance of each feature's rank across all 30 runs.

    High rank variance = the feature's importance is unstable.
    If 'checking_status' is ranked #1 in some runs and #8 in others,
    we can't trust that it's genuinely important.

    Threshold: rank variance > MAX_SHAP_RANK_VARIANCE is a concern.
    """
    print(f"\n  [Test 2] Bootstrap Stability ({n_bootstrap} subsamples)")

    np.random.seed(RANDOM_STATE)
    sample_size = min(200, len(X))

    all_ranks = {feat: [] for feat in features}
    explainer = shap.TreeExplainer(model)

    for i in range(n_bootstrap):
        idx     = np.random.choice(len(X), size=sample_size, replace=True)
        X_boot  = X.iloc[idx]

        shap_vals = explainer.shap_values(X_boot)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        ranks    = pd.Series(mean_abs, index=features).rank(ascending=False)

        for feat in features:
            all_ranks[feat].append(float(ranks[feat]))

    stability_results = []
    for feat in features:
        ranks_arr  = np.array(all_ranks[feat])
        mean_rank  = float(np.mean(ranks_arr))
        rank_var   = float(np.var(ranks_arr))
        rank_std   = float(np.std(ranks_arr))

        stability_results.append({
            "feature":   feat,
            "mean_rank": round(mean_rank, 2),
            "rank_std":  round(rank_std, 2),
            "rank_var":  round(rank_var, 2),
            "stable":    rank_std <= MAX_SHAP_RANK_VARIANCE,
        })

    stability_results.sort(key=lambda x: x["mean_rank"])

    unstable = [r for r in stability_results if not r["stable"]]
    print(f"    Unstable features (rank std > {MAX_SHAP_RANK_VARIANCE}): {[r['feature'] for r in unstable]}")
    print(f"    Stable features: {len(stability_results) - len(unstable)}/{len(stability_results)}")

    top5 = stability_results[:5]
    print(f"    Top 5 feature stability:")
    for r in top5:
        status = "STABLE" if r["stable"] else "UNSTABLE"
        print(f"      {r['mean_rank']:4.1f}  {r['feature']:30s}  std={r['rank_std']:.2f}  [{status}]")

    return {
        "n_bootstrap":      n_bootstrap,
        "sample_size":      sample_size,
        "features":         stability_results,
        "n_unstable":       len(unstable),
        "unstable_features": [r["feature"] for r in unstable],
        "overall_flag": "GREEN" if len(unstable) == 0 else ("YELLOW" if len(unstable) <= 2 else "RED"),
    }


# ── Test 3: Top Feature Flip Test ─────────────────────────────────────────────

def top_feature_flip_test(X, model, features, top_n=3):
    """
    Take the top N most important features and flip their values
    (swap low values with high values — imagine reversing rich/poor).

    If the model is genuinely using these features, flipping them should
    cause large changes in predicted risk scores.

    If flipping the #1 feature barely changes anything, the model isn't
    really using it — the SHAP importance was misleading.
    """
    print(f"\n  [Test 3] Top Feature Flip Test (top {top_n} features)")

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    mean_abs_shap  = np.mean(np.abs(shap_vals), axis=0)
    importance_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features   = [features[i] for i in importance_idx]

    baseline_probs = model.predict_proba(X)[:, 1]
    results = []

    for feat in top_features:
        X_flipped = X.copy()
        # Flip: replace each value with max + min - value (mirrors the distribution)
        feat_min = X[feat].min()
        feat_max = X[feat].max()
        X_flipped[feat] = feat_max + feat_min - X_flipped[feat]

        flipped_probs = model.predict_proba(X_flipped)[:, 1]
        mean_delta    = float(np.mean(np.abs(flipped_probs - baseline_probs)))
        avg_direction = float(np.mean(flipped_probs - baseline_probs))

        results.append({
            "feature":          feat,
            "mean_abs_delta":   round(mean_delta, 4),
            "avg_direction":    round(avg_direction, 4),
            "responsive":       mean_delta > 0.03,
        })
        print(f"    Flipping '{feat}': mean |delta| = {mean_delta:.4f}  "
              f"avg direction = {avg_direction:+.4f}  "
              f"{'RESPONSIVE' if mean_delta > 0.03 else 'NOT RESPONSIVE — investigate'}")

    return {"features_tested": results}


# ── Test 4: Local Explanation Samples ────────────────────────────────────────

def local_explanation_test(X, y, model, features, n_samples=5):
    """
    Pick 5 individual predictions and explain each one using SHAP.
    This is what a model risk officer or regulator would ask for:
    'explain why this specific person was rejected.'

    We pick:
    - 2 high-risk predictions (predicted default)
    - 2 low-risk predictions (predicted good)
    - 1 borderline prediction (near threshold)
    """
    print(f"\n  [Test 4] Local Explanations (sample predictions)")

    explainer  = shap.TreeExplainer(model)
    probs      = model.predict_proba(X)[:, 1]

    # Find representative samples
    high_risk_idx  = np.argsort(probs)[-2:]        # top 2 riskiest
    low_risk_idx   = np.argsort(probs)[:2]         # top 2 safest
    border_idx     = [np.argmin(np.abs(probs - 0.50))]  # closest to 0.5

    selected_idx = list(high_risk_idx) + list(low_risk_idx) + border_idx

    shap_vals  = explainer.shap_values(X.iloc[selected_idx])
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    base_value = float(explainer.expected_value[1] if isinstance(explainer.expected_value, list)
                       else explainer.expected_value)

    explanations = []
    labels = ["High-Risk", "High-Risk", "Low-Risk", "Low-Risk", "Borderline"]

    for i, (orig_idx, label) in enumerate(zip(selected_idx, labels)):
        sv = shap_vals[i]

        # Top 5 contributing features for this prediction
        top_contributors = sorted(
            [{"feature": features[j], "shap_value": round(float(sv[j]), 4)} for j in range(len(features))],
            key=lambda x: abs(x["shap_value"]), reverse=True
        )[:5]

        explanations.append({
            "sample_index":    int(orig_idx),
            "label":           label,
            "predicted_prob":  round(float(probs[orig_idx]), 4),
            "actual_target":   int(y.iloc[orig_idx]),
            "base_value":      round(base_value, 4),
            "top_contributors": top_contributors,
        })

        print(f"\n    Sample {orig_idx} [{label}]  pred={probs[orig_idx]:.4f}  actual={'DEFAULT' if y.iloc[orig_idx]==1 else 'GOOD'}")
        for contrib in top_contributors[:3]:
            direction = "↑risk" if contrib["shap_value"] > 0 else "↓risk"
            print(f"      {contrib['feature']:30s}  {contrib['shap_value']:+.4f} {direction}")

    return {"base_value": round(base_value, 4), "explanations": explanations}


# ── Test 5: Train vs OOT Importance Shift ────────────────────────────────────

def importance_shift_test(X_train, X_oot, model, features):
    """
    Compare global SHAP importance between training data and OOT data.

    If the top features are completely different between train and OOT,
    that means the model's decision logic has changed over time — a sign
    of concept drift or overfitting to training-specific patterns.

    We compute rank correlation between the two importance lists.
    Spearman correlation ~ 1.0 = same ranking, ~ 0.0 = completely different.
    """
    print("\n  [Test 5] Importance Shift (Train vs OOT)")

    explainer = shap.TreeExplainer(model)

    sv_train = explainer.shap_values(X_train.sample(min(200, len(X_train)), random_state=RANDOM_STATE))
    sv_oot   = explainer.shap_values(X_oot)

    if isinstance(sv_train, list): sv_train = sv_train[1]
    if isinstance(sv_oot, list):   sv_oot   = sv_oot[1]

    imp_train = pd.Series(np.mean(np.abs(sv_train), axis=0), index=features)
    imp_oot   = pd.Series(np.mean(np.abs(sv_oot),   axis=0), index=features)

    rank_train = imp_train.rank(ascending=False)
    rank_oot   = imp_oot.rank(ascending=False)

    spearman_corr = float(rank_train.corr(rank_oot, method="spearman"))
    flag = "GREEN" if spearman_corr > 0.80 else ("YELLOW" if spearman_corr > 0.60 else "RED")

    print(f"    Rank correlation (train vs OOT): {spearman_corr:.4f}  [{flag}]")
    print(f"    Interpretation: {'Feature importance is stable across time' if spearman_corr > 0.80 else 'Feature importance has shifted — concept drift suspected'}")

    comparison = pd.DataFrame({
        "train_rank": rank_train.astype(int),
        "oot_rank":   rank_oot.astype(int),
        "rank_change": (rank_oot - rank_train).astype(int),
    }).sort_values("train_rank")

    return {
        "spearman_rank_correlation": round(spearman_corr, 4),
        "flag": flag,
        "feature_comparison": comparison.reset_index().rename(columns={"index": "feature"}).to_dict(orient="records"),
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run_shap_validation():
    print("=" * 60)
    print("  VALIDATION MODULE 4: SHAP Feature Importance Stability")
    print("=" * 60)

    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    oot   = pd.read_csv(OOT_PATH)
    model = joblib.load(MODEL_PATH)

    with open(FEATURE_NAMES_PATH) as f:
        features = json.load(f)

    X_train, y_train = train[features], train[TARGET_COLUMN]
    X_test,  y_test  = test[features],  test[TARGET_COLUMN]
    X_oot,   y_oot   = oot[features],   oot[TARGET_COLUMN]

    report = {
        "global_importance_test": global_importance_test(X_test, model, features, "Test"),
        "global_importance_oot":  global_importance_test(X_oot,  model, features, "OOT"),
        "bootstrap_stability":    bootstrap_stability_test(X_test, model, features),
        "top_feature_flip":       top_feature_flip_test(X_test, model, features),
        "local_explanations":     local_explanation_test(X_oot, y_oot, model, features),
        "importance_shift":       importance_shift_test(X_train, X_oot, model, features),
    }

    # Remove raw shap_values arrays from JSON (too large — keep structured data only)
    report["global_importance_test"].pop("shap_values", None)
    report["global_importance_oot"].pop("shap_values", None)

    os.makedirs(os.path.dirname(SHAP_REPORT_PATH), exist_ok=True)
    with open(SHAP_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved → {SHAP_REPORT_PATH}")

    print("\n  FINDINGS SUMMARY:")
    print(f"    Bootstrap stability  : {report['bootstrap_stability']['overall_flag']}")
    print(f"    Importance shift     : {report['importance_shift']['flag']}")
    print(f"    Unstable features    : {report['bootstrap_stability']['unstable_features']}")

    return report


if __name__ == "__main__":
    run_shap_validation()
