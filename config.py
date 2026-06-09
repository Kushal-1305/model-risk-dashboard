"""
Central configuration for the Model Risk Dashboard.
All paths, thresholds, and model parameters live here.
"""

import os

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(ROOT_DIR, "data")
RAW_DATA_PATH   = os.path.join(DATA_DIR, "german_credit_raw.csv")
PROCESSED_PATH  = os.path.join(DATA_DIR, "german_credit_processed.csv")
TRAIN_PATH      = os.path.join(DATA_DIR, "train.csv")
TEST_PATH       = os.path.join(DATA_DIR, "test.csv")
OOT_PATH        = os.path.join(DATA_DIR, "oot.csv")   # Out-of-Time validation set

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR          = os.path.join(ROOT_DIR, "models")
MODEL_PATH          = os.path.join(MODELS_DIR, "xgb_credit_model.pkl")
FEATURE_NAMES_PATH  = os.path.join(MODELS_DIR, "feature_names.json")
BASELINE_METRICS    = os.path.join(MODELS_DIR, "baseline_metrics.json")

# ── Validation output paths ───────────────────────────────────────────────────
VALIDATION_DIR      = os.path.join(ROOT_DIR, "validation")
DRIFT_REPORT_PATH   = os.path.join(VALIDATION_DIR, "drift_report.json")
PERF_REPORT_PATH    = os.path.join(VALIDATION_DIR, "performance_report.json")
SHAP_REPORT_PATH    = os.path.join(VALIDATION_DIR, "shap_report.json")
ADV_REPORT_PATH     = os.path.join(VALIDATION_DIR, "adversarial_report.json")

# ── Report output ─────────────────────────────────────────────────────────────
REPORT_DIR          = os.path.join(ROOT_DIR, "report")
REPORT_OUTPUT_PATH  = os.path.join(REPORT_DIR, "sr117_validation_report.html")

# ── Model training parameters ─────────────────────────────────────────────────
RANDOM_STATE = 42

# Train / Test / OOT split ratios (must sum to 1.0)
# OOT = Out-of-Time: data held back to simulate future/unseen deployment
TRAIN_RATIO = 0.60
TEST_RATIO  = 0.20
OOT_RATIO   = 0.20

TARGET_COLUMN = "target"   # 1 = default (bad credit), 0 = good credit

XGBOOST_PARAMS = {
    "n_estimators":     200,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     RANDOM_STATE,
}

# ── Validation thresholds (SR 11-7 / industry standard) ──────────────────────
# PSI: Population Stability Index
#   < 0.10  → No significant change (GREEN)
#   0.10–0.20 → Moderate change, monitor (YELLOW)
#   > 0.20  → Significant shift, investigate (RED)
PSI_GREEN  = 0.10
PSI_YELLOW = 0.20

# AUC floor: if AUC drops below this on OOT, model needs revalidation
MIN_AUC_THRESHOLD = 0.65

# KS Statistic floor (how well model separates good vs bad)
MIN_KS_THRESHOLD = 0.20

# SHAP stability: max allowed rank change for top-5 features across bootstrap
MAX_SHAP_RANK_VARIANCE = 2.0
