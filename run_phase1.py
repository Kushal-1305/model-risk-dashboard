"""
Phase 1 Runner — Execute this to set up data and train the baseline model.

Run: python run_phase1.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "=" * 60)
print("  MODEL RISK DASHBOARD — Phase 1: Data + Baseline Model")
print("=" * 60 + "\n")

# Step 1: Download, preprocess, and split the data
from data.load_data import download_and_save_raw, preprocess, split_data, print_data_summary
df_raw  = download_and_save_raw()
df_proc = preprocess(df_raw)
split_data(df_proc)
print_data_summary(df_proc)

# Step 2: Train the model and save artifacts
from models.train_model import train
model, metrics = train()

print("\nAll Phase 1 artifacts saved:")
print("  data/train.csv")
print("  data/test.csv")
print("  data/oot.csv")
print("  models/xgb_credit_model.pkl")
print("  models/feature_names.json")
print("  models/baseline_metrics.json")
print("\nReady for Phase 2: Validation Suite")
