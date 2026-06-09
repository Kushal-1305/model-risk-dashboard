"""
Phase 2 Runner — Execute this to run the complete validation suite.

Run: python run_phase2.py

What this does:
  Runs all 4 validation modules in sequence and prints a consolidated
  findings summary — exactly what a model risk officer would review.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "=" * 60)
print("  MODEL RISK DASHBOARD — Phase 2: Validation Suite")
print("=" * 60)

# Module 1: Performance Degradation
print("\n\nRUNNING MODULE 1 OF 4...\n")
from validation.performance import run_performance_validation
perf_report = run_performance_validation()

# Module 2: Data Drift Detection
print("\n\nRUNNING MODULE 2 OF 4...\n")
from validation.drift import run_drift_validation
drift_report = run_drift_validation()

# Module 3: Adversarial Input Testing
print("\n\nRUNNING MODULE 3 OF 4...\n")
from validation.adversarial import run_adversarial_validation
adv_report = run_adversarial_validation()

# Module 4: SHAP Feature Importance Stability
print("\n\nRUNNING MODULE 4 OF 4...\n")
from validation.shap_stability import run_shap_validation
shap_report = run_shap_validation()


# ── Consolidated Findings ─────────────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("  CONSOLIDATED VALIDATION FINDINGS")
print("=" * 60)

def color(flag):
    return {"RED": "[ RED  ]", "YELLOW": "[YELLOW]", "GREEN": "[GREEN ]"}.get(flag, f"[{flag}]")

print("\n  PERFORMANCE:")
print(f"    PSI (OOT score shift)        {color(perf_report['score_distribution']['psi_oot_flag'])}")
print(f"    Calibration (max gap)        {color(perf_report['calibration']['calibration_flag'])}")

print("\n  DATA DRIFT:")
psi_oot_flag = drift_report["feature_drift_oot"]["n_red"]
print(f"    RED features (OOT)           {drift_report['feature_drift_oot']['n_red']} features")
print(f"    Target drift (OOT)           {color(drift_report['target_drift']['flag_oot'])}")
print(f"    Covariate shift              {color(drift_report['covariate_shift']['flag'])}")

print("\n  ADVERSARIAL:")
print(f"    Boundary instability         {color(adv_report['boundary_probing']['flag'])}")
print(f"    Directional violations       {adv_report['directional_sanity']['n_fail']} failed")
print(f"    Recession stress increase    {adv_report['recession_stress']['relative_increase_pct']:.1%}  {color(adv_report['recession_stress']['flag'])}")

print("\n  EXPLAINABILITY:")
print(f"    SHAP bootstrap stability     {color(shap_report['bootstrap_stability']['overall_flag'])}")
print(f"    Importance shift (OOT)       {color(shap_report['importance_shift']['flag'])}")

print("\n" + "=" * 60)
print("  Phase 2 Complete. All validation reports saved to validation/")
print("  Ready for Phase 3: SR 11-7 Report Generation")
print("=" * 60 + "\n")
